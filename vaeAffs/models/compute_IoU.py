from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from glob import glob
from .util import *
import numpy as np
from copy import deepcopy
from torch.nn.parallel.data_parallel import data_parallel

from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss

from inferno.extensions.layers.reshape import GlobalMeanPooling
from inferno.extensions.containers.graph import Identity

from vaeAffs.transforms import DownsampleAndCrop3D

from neurofire.models.unet.unet_3d import CONV_TYPES, Decoder, DecoderResidual, BaseResidual, Base, Output, Encoder, \
    EncoderResidual

import warnings

from speedrun.log_anywhere import log_image, log_embedding, log_scalar
from segmfriends.utils.various import parse_data_slice
from .losses import extract_patches_torch_new

from multiprocessing.pool import ThreadPool
from itertools import repeat

from segmfriends.utils.various import starmap_with_kwargs


class IoULoss(nn.Module):
    """Used as a regularizer in the training of patch-embeddings (they should be consistent)"""

    def __init__(self, model, loss_type="Dice", model_kwargs=None, devices=(0, 1),
                 min_random_offset_orig_res=(0,3,3)):
        super(IoULoss, self).__init__()

        if loss_type == "Dice":
            self.loss = SorensenDiceLoss()
        elif loss_type == "MSE":
            self.loss = nn.MSELoss()
        elif loss_type == "BCE":
            self.loss = nn.BCELoss()
        else:
            raise ValueError

        self.devices = devices
        self.model_kwargs = model_kwargs

        self.model = model
        assert isinstance(min_random_offset_orig_res, (tuple, list))
        self.min_random_offset_orig_res = min_random_offset_orig_res if isinstance(min_random_offset_orig_res, tuple) else tuple(min_random_offset_orig_res)

    def get_random_offset(self, patch_size, patch_dws_fact, size=1, min_random_offset_orig_res=None):
        assert isinstance(size, int)

        min_random_offset_orig_res = min_random_offset_orig_res if min_random_offset_orig_res is not None else self.min_random_offset_orig_res

        all_real_offsets = np.array([np.arange(-int(sz/2)*dws, (int(sz/2)+1)*dws, dws) for sz, dws in zip(patch_size, patch_dws_fact)])
        mask_allowed_offs = np.array([abs(all_real_offsets[i]) >= min for i, min in enumerate(min_random_offset_orig_res)])
        random_offs = [[np.random.choice(all_real_offsets[i][mask_allowed_offs[i]].flatten()).item() for i in range(3)] for _ in range(size)]

        return random_offs

    def forward(self, all_predictions, all_targets):
        mdl_kwargs = self.model_kwargs
        ptch_kwargs = mdl_kwargs["patchNet_kwargs"]
        nb_preds = len(all_predictions)

        loss = 0

        for nb_patch_net, pred in enumerate(all_predictions):
            kwargs = ptch_kwargs[nb_patch_net]

            if "IoU_kwargs" not in kwargs:
                continue

            IoU_kwargs = kwargs["IoU_kwargs"]

            # Collect options from config:
            patch_shape = kwargs.get("patch_size")
            assert all(i % 2 ==  1 for i in patch_shape), "Patch should be odd"
            patch_dws_fact = kwargs.get("patch_dws_fact", [1,1,1])
            pred_dws_fact = kwargs.get("pred_dws_fact", [1,1,1])

            # Process prediction and target:
            if isinstance(all_targets, (list, tuple)):
                assert "nb_target" in kwargs, "Multiple targets passed. Target should be specified"
                gt_segm = all_targets[kwargs["nb_target"]]
            else:
                gt_segm = all_targets

            precrop_pred = kwargs.get("precrop_pred", None)
            from segmfriends.utils.various import parse_data_slice
            if precrop_pred is not None:
                precrop_pred_slice = (slice(None), slice(None)) + parse_data_slice(precrop_pred)
            else:
                precrop_pred_slice = slice(None)
            processed_pred = pred[precrop_pred_slice]

            crop_slice_targets, crop_slice_prediction = get_slicing_crops(processed_pred.shape, gt_segm.shape, pred_dws_fact)
            processed_pred = processed_pred[crop_slice_prediction]
            processed_gt_segm = gt_segm[crop_slice_targets]
            dws_crop_slice = (slice(None), slice(None)) + tuple(slice(None, None, dws) for dws in pred_dws_fact)
            processed_gt_segm = processed_gt_segm[dws_crop_slice]

            # Get random offset:
            nb_random_IoU = IoU_kwargs.get("nb_random_IoU", 1)
            min_random_offset_orig_res = IoU_kwargs.get("min_random_offset", None)

            # TODO: we could be on a boundary...
            random_offsets = self.get_random_offset(patch_shape, patch_dws_fact, size=nb_random_IoU,
                                                    min_random_offset_orig_res=min_random_offset_orig_res)


            for offs in random_offsets:
                assert all([of%dws == 0 for of, dws in zip(offs, pred_dws_fact)]), "Pred. dws factor should be compatible with patch dws"
                stride = [abs(int(of/dws)) if of != 0 else 1 for of, dws in zip(offs, pred_dws_fact)]
                patches, crop_slice, nb_patches = extract_patches_torch_new(processed_pred, shape=(1, 1, 1), stride=stride,
                                                                   max_random_crop=stride)
                patches = data_parallel(self.model.models[-1].patch_models[nb_patch_net], patches[:, :, 0, 0, 0], self.devices)[
                          :, [0]]
                patches = patches.view(*nb_patches, *patch_shape)

                invert_masks = patch_dws_fact[1] <= 6
                IoU, valid_predictions = IoU_worker(patches, offs, patch_target_size=None, stride=stride, patch_dws_fact=patch_dws_fact, patch_shape=patch_shape,
                                                    invert_masks=invert_masks)

                # Compute targets:
                gt_labels, _, _ = extract_patches_torch_new(processed_gt_segm, shape=(1, 1, 1), stride=stride,
                                                                            crop_slice=crop_slice, limit_patches_to=nb_patches)
                gt_labels = gt_labels.view(*nb_patches)
                IoU_targets, valid_predictions_1 = compute_IoU_targets(gt_labels, offs, patch_target_size=None, stride=stride,
                                                                       patch_dws_fact=patch_dws_fact,
                                                                       patch_shape=patch_shape, ignore_label=0)

                # ------------ Compute loss ----------------
                # Invert targets (most common case is IoU == 0)
                valid_predictions = (valid_predictions*valid_predictions_1).float()
                IoU = (1. - IoU) * valid_predictions
                IoU_targets = (1. - IoU_targets.float()) * valid_predictions
                loss = loss + self.loss(IoU, IoU_targets)
        return loss



def get_slicing_crops(pred_shape, target_shape, pred_ds_factor):
    """
    Decide how to crop the prediction and targets to get consistent tensors
    """
    assert len(pred_shape) == 5 and len(target_shape) == 5
    pred_shape = pred_shape[2:]
    target_shape = target_shape[2:]
    # Compute new left crops:
    upscaled_pred_shape = [sh*fctr for sh, fctr in zip(pred_shape, pred_ds_factor)]

    shape_diff = [orig - trg for orig, trg in zip(target_shape, upscaled_pred_shape)]
    assert all([diff >= 0 for diff in shape_diff]), "Prediction should be smaller or equal to the targets!"
    assert all([diff % 2 == 0 for diff in shape_diff])
    padding = [int(diff/2) for diff in shape_diff]

    crop_slice_targets = [slice(None), slice(None)]
    crop_slice_prediction = [slice(None), slice(None)]
    for dim, pad in enumerate(padding):
        if pad > 0:
            # Crop targets
            crop_slice_targets.append(slice(pad, -pad))
            crop_slice_prediction.append(slice(None))
        else:
            # No need to crop:
            crop_slice_targets.append(slice(None))
            crop_slice_prediction.append(slice(None))

    return tuple(crop_slice_targets), tuple(crop_slice_prediction)




class IoULossOld(nn.Module):
    """MOST PROBABLY DEPRECATED"""
    def __init__(self, model, loss_type="Dice", model_kwargs=None, devices=(0, 1),
                 offset=(0, 5, 5),
                 stride=None, pre_crop_pred=None):
        super(IoULossOld, self).__init__()
        if loss_type == "Dice":
            self.loss = SorensenDiceLoss()
        elif loss_type == "MSE":
            self.loss = nn.MSELoss()
        elif loss_type == "BCE":
            self.loss = nn.BCELoss()
        else:
            raise ValueError

        self.devices = devices
        self.model_kwargs = model_kwargs
        self.MSE_loss = nn.MSELoss()
        self.smoothL1_loss = nn.SmoothL1Loss()
        # TODO: use nn.BCEWithLogitsLoss()
        self.BCE = nn.BCELoss()
        self.soresen_loss = SorensenDiceLoss()

        from vaeAffs.models.vanilla_vae import VAE_loss
        self.VAE_loss = VAE_loss()

        self.model = model

        # TODO: hack to adapt to stacked model:
        self.downscale_and_crop_targets = {}
        if hasattr(self.model, "collected_patchNet_kwargs"):
            self.model_kwargs["patchNet_kwargs"] = [kwargs for i, kwargs in
                                                    enumerate(self.model.collected_patchNet_kwargs) if
                                                    i in self.model.trained_patchNets]

            # FIXME: generalize to the non-stacked model (there I also have global in the keys...)
            for nb, kwargs in enumerate(self.model_kwargs["patchNet_kwargs"]):
                if "downscale_and_crop_target" in kwargs:
                    self.downscale_and_crop_targets[nb] = DownsampleAndCrop3D(**kwargs["downscale_and_crop_target"])

        # Specific arguments:
        self.stride = tuple(stride) if stride is not None else (1, 1, 1)
        if not isinstance(offset, tuple):
            assert isinstance(offset, list)
            offset = tuple(offset)
        assert len(offset) == 3
        self.offset = offset

        if pre_crop_pred is not None:
            assert isinstance(pre_crop_pred, str)
            pre_crop_pred = (slice(None), slice(None)) + parse_data_slice(pre_crop_pred)
        self.pre_crop_pred = pre_crop_pred

    def forward(self, all_predictions, target):
        mdl_kwargs = self.model_kwargs
        ptch_kwargs = mdl_kwargs["patchNet_kwargs"]
        # TODO: generalize
        kwargs = ptch_kwargs[0]
        nb_preds = len(all_predictions)

        # Collect options from config:
        patch_shape = kwargs.get("patch_size")
        assert all(i % 2 == 1 for i in patch_shape), "Patch should be odd"
        patch_dws_fact = kwargs.get("patch_dws_fact", [1, 1, 1])
        real_patch_shape = tuple(pt * fc for pt, fc in zip(patch_shape, patch_dws_fact))

        assert nb_preds == 1
        pred = all_predictions[0]
        # Pre-crop prediction:
        pred = pred[self.pre_crop_pred] if self.pre_crop_pred is not None else pred

        # Compute crop slice after rolling:
        left_crop = [off if off > 0 else 0 for off in self.offset]
        right_crop = [sh + off if off < 0 else None for off, sh in zip(self.offset, pred.shape[2:])]
        crop_slice = (slice(None), slice(None)) + tuple(slice(lft, rgt) for lft, rgt in zip(left_crop, right_crop))

        # ----
        # Roll axes and get some embeddings:
        # ----
        rolled_pred = pred.roll(shifts=tuple(-offs for offs in self.offset), dims=(2, 3, 4))
        rolled_pred = rolled_pred[crop_slice]
        pred = pred[crop_slice]
        embeddings_1, _, _ = extract_patches_torch_new(pred, shape=(1, 1, 1), stride=self.stride)
        embeddings_2, _, _ = extract_patches_torch_new(rolled_pred, shape=(1, 1, 1), stride=self.stride)

        # Compute the actual IoU scores by expanding the patches:
        with torch.no_grad():
            # TODO: generalize to different models and patchnets...
            patches_1 = data_parallel(self.model.models[1].patch_models[0], embeddings_1[:, :, 0, 0, 0], self.devices)[
                        :, [0]]
            patches_2 = data_parallel(self.model.models[1].patch_models[0], embeddings_2[:, :, 0, 0, 0], self.devices)[
                        :, [0]]
        assert all([offs%dws == 0 for dws, offs in zip(patch_dws_fact, self.offset)])
        patch_offset = tuple(int(offs/dws) for dws, offs in zip(patch_dws_fact, self.offset))
        # Get crop slices patch_1:
        left_crop = [off if off > 0 else 0 for off in patch_offset]
        right_crop = [sh + off if off < 0 else None for off, sh in zip(patch_offset, patch_shape)]
        crop_slice_patch1 = (slice(None), slice(None)) + tuple(slice(lft, rgt) for lft, rgt in zip(left_crop, right_crop))
        # Get crop slices patch_2:
        left_crop = [-off if off < 0 else 0 for off in patch_offset]
        right_crop = [sh - off if off > 0 else None for off, sh in zip(patch_offset, patch_shape)]
        crop_slice_patch2 = (slice(None), slice(None)) + tuple(slice(lft, rgt) for lft, rgt in zip(left_crop, right_crop))

        # Some plotting:
        log_image("IoU_patch1", 1.- patches_1)
        log_image("IoU_patch2", 1. - patches_2)

        # Crop and intersect masks:
        patches_1 = 1. - patches_1[crop_slice_patch1]
        patches_2 = 1. - patches_2[crop_slice_patch2]
        true_IoU = compute_intersection_over_union(patches_1, patches_2)

        binary_targets = true_IoU < 0.3
        ignored_targets = (true_IoU > 0.3) * (true_IoU < 0.7)

        log_image("IoU_patch1_cr", patches_1)
        log_image("IoU_patch2_cr", patches_2)

        # Reshape and predict IoU scores:
        predicted_IoU = data_parallel(self.model.IoU_module, (embeddings_1[:, :, :, 0, 0], embeddings_2[:, :, :, 0, 0]),
                              self.devices)[:,0,0]

        log_image("IoU_score", torch.ones_like(patches_2[[0]])*true_IoU[0])
        log_image("IoU_score_pred", torch.ones_like(patches_2[[0]])*predicted_IoU[0])

        binary_targets = binary_targets * (1 - ignored_targets)
        predicted_IoU = predicted_IoU * (1 - ignored_targets).float()

        with warnings.catch_warnings(record=True) as w:
            loss = data_parallel(self.loss, (predicted_IoU.unsqueeze(-1), binary_targets.float().unsqueeze(-1)),
                      self.devices).mean()
        log_scalar("avg_IoU", predicted_IoU.mean())
        log_scalar("std_IoU", predicted_IoU.std())
        log_scalar("nb_ignored", ignored_targets.float().sum())
        log_scalar("nb_total", ignored_targets.shape[0])


        return loss


def compute_intersection_over_union(mask1, mask2):
    # TODO: assert values between 0 and 1...
    if isinstance(mask1, np.ndarray):
        sum_kwargs = {'axis': -1}
    elif isinstance(mask1, torch.Tensor):
        sum_kwargs = {'dim': -1}
    intersection = (mask1 * mask2).sum(**sum_kwargs).sum(**sum_kwargs).sum(**sum_kwargs)
    union = mask1.sum(**sum_kwargs).sum(**sum_kwargs).sum(**sum_kwargs) + mask2.sum(**sum_kwargs).sum(**sum_kwargs).sum(**sum_kwargs) - intersection
    return intersection/union



class ComputeIoU(nn.Module):
    """MOST PROBABLY DEPRECATED"""
    def __init__(self, model, offsets, ptch_kwargs, loss_type="Dice", model_kwargs=None, devices=(0, 1),
                 stride=None, pre_crop_pred=None):
        super(ComputeIoU, self).__init__()
        if loss_type == "Dice":
            self.loss = SorensenDiceLoss()
        elif loss_type == "MSE":
            self.loss = nn.MSELoss()
        elif loss_type == "BCE":
            self.loss = nn.BCELoss()
        else:
            raise ValueError

        self.devices = devices
        self.model_kwargs = model_kwargs
        self.MSE_loss = nn.MSELoss()
        self.smoothL1_loss = nn.SmoothL1Loss()
        # TODO: use nn.BCEWithLogitsLoss()
        self.BCE = nn.BCELoss()
        self.soresen_loss = SorensenDiceLoss()

        from vaeAffs.models.vanilla_vae import VAE_loss
        self.VAE_loss = VAE_loss()

        self.model = model

        # TODO: hack to adapt to stacked model:
        self.downscale_and_crop_targets = {}
        if hasattr(self.model, "collected_patchNet_kwargs"):
            self.model_kwargs["patchNet_kwargs"] = [kwargs for i, kwargs in
                                                    enumerate(self.model.collected_patchNet_kwargs) if
                                                    i in self.model.trained_patchNets]

            # FIXME: generalize to the non-stacked model (there I also have global in the keys...)
            for nb, kwargs in enumerate(self.model_kwargs["patchNet_kwargs"]):
                if "downscale_and_crop_target" in kwargs:
                    self.downscale_and_crop_targets[nb] = DownsampleAndCrop3D(**kwargs["downscale_and_crop_target"])

        # Specific arguments:
        self.stride = tuple(stride) if stride is not None else (1, 1, 1)
        # if not isinstance(offset, tuple):
        #     assert isinstance(offset, list)
        #     offset = tuple(offset)
        # assert len(offset) == 3
        # TODO: assert offsets
        self.offsets = offsets
        self.ptch_kwargs = ptch_kwargs

        if pre_crop_pred is not None:
            assert isinstance(pre_crop_pred, str)
            pre_crop_pred = (slice(None), slice(None)) + parse_data_slice(pre_crop_pred)
        self.pre_crop_pred = pre_crop_pred

    def forward(self, pred, target):
        # FIXME:
        pred = pred[0]

        mdl_kwargs = self.model_kwargs
        kwargs = self.ptch_kwargs

        # Collect options from config:
        patch_shape = kwargs.get("patch_size")
        assert all(i % 2 == 1 for i in patch_shape), "Patch should be odd"
        patch_dws_fact = kwargs.get("patch_dws_fact", [1, 1, 1])
        real_patch_shape = tuple(pt * fc for pt, fc in zip(patch_shape, patch_dws_fact))

        # Pre-crop prediction:
        pred = pred[self.pre_crop_pred] if self.pre_crop_pred is not None else pred

        # FIXME: here we no longer crop, so we will get invalid values
        # # Compute crop slice after rolling:
        # left_crop = [off if off > 0 else 0 for off in self.offset]
        # right_crop = [sh + off if off < 0 else None for off, sh in zip(self.offset, pred.shape[2:])]
        # crop_slice = (slice(None), slice(None)) + tuple(slice(lft, rgt) for lft, rgt in zip(left_crop, right_crop))

        import time
        tick = time.time()
        patches, _, nb_patches = extract_patches_torch_new(pred, shape=(1, 1, 1), stride=self.stride)
        del pred
        with torch.no_grad():
            # TODO: generalize to different models and patchnets...
            patches = data_parallel(self.model.models[1].patch_models[0], patches[:, :, 0, 0, 0], self.devices)[
                        :, [0]]


        # TODO: reshape and bring to cpu:
        patches = patches.cpu().numpy()
        patches = patches.reshape(*nb_patches,*patches.shape[2:])
        # ----
        # Roll axes and compute IoU scores:
        # ----
        tock = time.time()
        print("Took", tock-tick)
        tick = time.time()
        results = []
        for offset in self.offsets:
            assert all([offs % strd == 0 for strd, offs in zip(self.stride, offset)])
            roll_offset = tuple(int(offs / dws) for dws, offs in zip(patch_dws_fact, offset))

            rolled_patches = np.roll(patches, shift=tuple(-offs for offs in roll_offset), axis=(0,1,2))

            assert all([offs%dws == 0 for dws, offs in zip(patch_dws_fact, offset)])
            patch_offset = tuple(int(offs/dws) for dws, offs in zip(patch_dws_fact, offset))
            # Get crop slices patch_1:
            left_crop = [off if off > 0 else 0 for off in patch_offset]
            right_crop = [sh + off if off < 0 else None for off, sh in zip(patch_offset, patch_shape)]
            crop_slice_patches = (slice(None), slice(None), slice(None)) + tuple(slice(lft, rgt) for lft, rgt in zip(left_crop, right_crop))
            # Get crop slices patch_2:
            left_crop = [-off if off < 0 else 0 for off in patch_offset]
            right_crop = [sh - off if off > 0 else None for off, sh in zip(patch_offset, patch_shape)]
            crop_slice_rolled_patches = (slice(None), slice(None), slice(None)) + tuple(slice(lft, rgt) for lft, rgt in zip(left_crop, right_crop))

            # Crop and compute IoU:
            IoU_scores = compute_intersection_over_union(1. - patches[crop_slice_patches],
                                                         1. - rolled_patches[crop_slice_rolled_patches])

            results.append(IoU_scores)

        tock = time.time()
        print("Took", tock-tick)
        print("Done, now reshape...")


from embeddingutils.models.unet import GeneralizedStackedPyramidUNet3D


class IntersectOverUnionUNet(GeneralizedStackedPyramidUNet3D):
    def __init__(self, offsets, num_IoU_workers=1,
                 number_patchNet=0,
                 pre_crop_pred=None,
                 patch_size_per_offset=None,
                 slicing_config=None,
                 IoU_on_GPU=False,
                 *super_args, **super_kwargs):
        super(IntersectOverUnionUNet, self).__init__(*super_args, **super_kwargs)

        # TODO: generalize
        self.ptch_kwargs = [kwargs for i, kwargs in
                                enumerate(self.collected_patchNet_kwargs) if
                                i in self.trained_patchNets]
        self.number_patchNet = number_patchNet
        self.ptch_kwargs = self.ptch_kwargs[number_patchNet]

        assert 'window_size' in slicing_config and slicing_config is not None
        slicing_config['stride'] = slicing_config['window_size']
        self.slicing_config = slicing_config

        # TODO: Assert
        self.IoU_on_GPU = IoU_on_GPU
        self.offsets = offsets
        if patch_size_per_offset is None:
            patch_size_per_offset = [None for _ in range(len(offsets))]
        else:
            assert len(patch_size_per_offset) == len(offsets)
        self.patch_size_per_offset = patch_size_per_offset
        self.num_IoU_workers = num_IoU_workers
        if pre_crop_pred is not None:
            assert isinstance(pre_crop_pred, str)
            pre_crop_pred = (slice(None), slice(None)) + parse_data_slice(pre_crop_pred)
        self.pre_crop_pred = pre_crop_pred

        # Deduce invalid affinity-padding from offsets (how much we should crop to get rid of invalid predictions)
        offset_arr = np.array(self.offsets)
        assert offset_arr.shape[1] == 3
        padding_left = np.abs(np.minimum(offset_arr, 0)).max(axis=0)
        padding_right = np.abs(np.maximum(offset_arr, 0)).max(axis=0)
        self.final_asym_crop_pred = [[lft, rgt] for lft, rgt in zip(padding_left, padding_right)]
        # self.invalid_crop = (slice(None),) + tuple(slice(pad, -pad) if pad !=0 else slice(None) for pad in padding)

    def forward(self, *inputs):
        pred = super(IntersectOverUnionUNet, self).forward(*inputs)

        def make_sliding_windows(volume_shape, window_size, stride, downsampling_ratio=None):
            from inferno.io.volumetric import volumetric_utils as vu
            assert isinstance(volume_shape, tuple)
            ndim = len(volume_shape)
            if downsampling_ratio is None:
                downsampling_ratio = [1] * ndim
            elif isinstance(downsampling_ratio, int):
                downsampling_ratio = [downsampling_ratio] * ndim
            elif isinstance(downsampling_ratio, (list, tuple)):
                # assert_(len(downsampling_ratio) == ndim, exception_type=ShapeError)
                downsampling_ratio = list(downsampling_ratio)
            else:
                raise NotImplementedError

            return list(vu.slidingwindowslices(shape=list(volume_shape),
                                               ds=downsampling_ratio,
                                               window_size=window_size,
                                               strides=stride,
                                               shuffle=False,
                                               add_overhanging=True))

        del inputs
        # torch.cuda.empty_cache()
        pred = pred[self.number_patchNet]
        assert pred.shape[0] == 1, "Only batch == 1 is supported atm"

        device = pred.get_device()


        sliding_windows = make_sliding_windows(pred.shape[2:], **self.slicing_config)

        kwargs = self.ptch_kwargs

        # Collect options from config:
        patch_shape = kwargs.get("patch_size")
        assert all(i % 2 == 1 for i in patch_shape), "Patch should be odd"
        patch_dws_fact = kwargs.get("patch_dws_fact", [1, 1, 1])
        real_patch_shape = tuple(pt * fc for pt, fc in zip(patch_shape, patch_dws_fact))

        # Pre-crop prediction:
        pred = pred[self.pre_crop_pred] if self.pre_crop_pred is not None else pred

        # return torch.cat((inputs[1][:,:,3:-3,75:-75, 75:-75], pred[:, :4]), dim=1)

        # FIXME: here we no longer crop, so we will get invalid values
        # # Compute crop slice after rolling:
        # left_crop = [off if off > 0 else 0 for off in self.offset]
        # right_crop = [sh + off if off < 0 else None for off, sh in zip(self.offset, pred.shape[2:])]
        # crop_slice = (slice(None), slice(None)) + tuple(slice(lft, rgt) for lft, rgt in zip(left_crop, right_crop))

        # if self.IoU_on_GPU:
        #     initial_shape = pred.shape
        #     pred = pred.reshape(*initial_shape[:2], -1) # Compress spatial dimensions
        #     pred = pred.permute(0,2,1)
        #     pred = pred.reshape(-1, initial_shape[1]) # Put them in the batch dimension
        #
        #     # Predict patches:
        #     pred = self.models[-1].patch_models[self.number_patchNet](pred)
        #     # Reshape:
        #     pred = pred.view(*initial_shape[-3:], *patch_shape)
        # else:
        patches_collected = None
        # print("Initial shape:", pred.shape)

        # Predict all patches:
        # print("Sliding windows: ", len(sliding_windows))
        for i, current_slice in enumerate(sliding_windows):
            # print("Iter {}".format(i))
            full_slice = (slice(None), slice(None),) + current_slice
            # TODO: Now this is simply doing a reshape...
            # print("Prima: ", pred[full_slice].shape)
            emb_vectors, _, nb_patches = extract_patches_torch_new(pred[full_slice], shape=(1, 1, 1), stride=(1,1,1))
            # print("Dopo: ", emb_vectors.shape)
            # TODO: Generalize to more models...?
            patches = self.models[-1].patch_models[self.number_patchNet](emb_vectors[:, :, 0, 0, 0])

            # From now on we can work on the CPU (too memory consuming):
            patch_shape = patches.shape[2:]
            if not self.IoU_on_GPU:
                patches = patches.cpu().numpy()
                patches = patches.reshape(*nb_patches, *patch_shape)
            else:
                patches = patches.view(*nb_patches, *patch_shape)


            if patches_collected is None:
                # Create output array with shape (volume_shape_z, ..., patch_shape_z, ...)
                if self.IoU_on_GPU:
                    patches_collected = torch.zeros(pred.shape[2:] + patch_shape).cuda(patches.get_device())
                else:
                    patches_collected = np.empty(pred.shape[2:] + patch_shape)

            patches_collected[current_slice] = patches
        pred = patches_collected

        # ----
        # Roll axes and compute affinities with IoU scores:
        # ----
        kwargs_pool = {"stride": (1, 1, 1),
                       "patch_dws_fact": patch_dws_fact,
                       "patch_shape": patch_shape}
        args_pool = zip(repeat(pred), self.offsets, self.patch_size_per_offset)
        pool = ThreadPool(processes=self.num_IoU_workers)
        results = starmap_with_kwargs(pool, IoU_worker, args_iter=args_pool,
                                      kwargs_iter=repeat(kwargs_pool))
        pool.close()
        pool.join()
        if self.IoU_on_GPU:
            return torch.stack([item[0] for item in results]).unsqueeze(0)
        else:
            affinities = np.stack([item[0] for item in results])
            # Get rid of invalid predictions and add batch dim:
            # valid_mask = np.stack([item[1] for item in results])
            return torch.from_numpy(np.expand_dims(affinities, axis=0)).cuda(device)





class IntersectOverUnionUNetOld(GeneralizedStackedPyramidUNet3D):
    def __init__(self, offsets, stride, num_IoU_workers=1,
                 pre_crop_pred=None,
                 patch_size_per_offset=None,
                 *super_args, **super_kwargs):
        super(IntersectOverUnionUNetOld, self).__init__(*super_args, **super_kwargs)

        # TODO: generalize
        self.ptch_kwargs = [kwargs for i, kwargs in
                                enumerate(self.collected_patchNet_kwargs) if
                                i in self.trained_patchNets]
        self.ptch_kwargs = self.ptch_kwargs[0]

        # TODO: Assert
        self.offsets = offsets
        if patch_size_per_offset is None:
            patch_size_per_offset = [None for _ in range(len(offsets))]
        else:
            assert len(patch_size_per_offset) == len(offsets)
        self.patch_size_per_offset = patch_size_per_offset
        self.stride = stride
        self.num_IoU_workers = num_IoU_workers
        if pre_crop_pred is not None:
            assert isinstance(pre_crop_pred, str)
            pre_crop_pred = (slice(None), slice(None)) + parse_data_slice(pre_crop_pred)
        self.pre_crop_pred = pre_crop_pred

    def forward(self, *inputs):
        pred = super(IntersectOverUnionUNetOld, self).forward(*inputs)

        # FIXME: delete again the inputs!
        del inputs
        # torch.cuda.empty_cache()
        assert len(pred) == 1
        pred = pred[0]
        assert pred.shape[0] == 1, "Only batch == 1 is supported atm"

        kwargs = self.ptch_kwargs

        # Collect options from config:
        patch_shape = kwargs.get("patch_size")
        assert all(i % 2 == 1 for i in patch_shape), "Patch should be odd"
        patch_dws_fact = kwargs.get("patch_dws_fact", [1, 1, 1])
        real_patch_shape = tuple(pt * fc for pt, fc in zip(patch_shape, patch_dws_fact))

        # Pre-crop prediction:
        pred = pred[self.pre_crop_pred] if self.pre_crop_pred is not None else pred

        # return torch.cat((inputs[1][:,:,3:-3,75:-75, 75:-75], pred[:, :4]), dim=1)

        # FIXME: here we no longer crop, so we will get invalid values
        # # Compute crop slice after rolling:
        # left_crop = [off if off > 0 else 0 for off in self.offset]
        # right_crop = [sh + off if off < 0 else None for off, sh in zip(self.offset, pred.shape[2:])]
        # crop_slice = (slice(None), slice(None)) + tuple(slice(lft, rgt) for lft, rgt in zip(left_crop, right_crop))

        # Depending on the stride, iterate:
        stride = self.stride
        stride_offsets = [0,0,0]
        results_collected = []
        while True:
            patches, _, nb_patches = extract_patches_torch_new(pred[:,:,stride_offsets[0]:,stride_offsets[1]:,stride_offsets[2]:],
                                                               shape=(1, 1, 1), stride=self.stride)
            # TODO: Generalize to more models...?
            patches = self.models[-1].patch_models[0](patches[:, :, 0, 0, 0])
            # patches = data_parallel(self.models[-1].patch_models[0], patches[:, :, 0, 0, 0], self.devices)[
            #             :, [0]]

            # From now on we can work on the CPU (too memory consuming):
            # patches = patches.cpu().numpy()
            patches = patches.reshape(*nb_patches, *patches.shape[2:])
            # ----
            # Roll axes and compute IoU scores:
            # ----
            kwargs_pool = {"stride": self.stride,
                           "patch_dws_fact": patch_dws_fact,
                           "patch_shape": patch_shape}
            args_pool = zip(repeat(patches), self.offsets, self.patch_size_per_offset)
            pool = ThreadPool(processes=self.num_IoU_workers)
            results = starmap_with_kwargs(pool, IoU_worker, args_iter=args_pool,
                                          kwargs_iter=repeat(kwargs_pool))
            pool.close()
            pool.join()
            predictions = [item[0] for item in results]
            masks = [item[1] for item in results]

            # results_collected.append([deepcopy(stride_offsets), np.stack(results)])
            results_collected.append([deepcopy(stride_offsets), torch.stack(predictions), np.stack(masks)])

            if stride_offsets[2]+1 == stride[2]:
                if stride_offsets[1]+1 == stride[1]:
                    if stride_offsets[0]+1 == stride[0]:
                        break
                    else:
                        stride_offsets[0] += 1
                        stride_offsets[1] = 0
                        stride_offsets[2] = 0
                else:
                    stride_offsets[1] += 1
                    stride_offsets[2] = 0
            else:
                stride_offsets[2] += 1

        # Merge everything in one output:
        final_output = torch.empty((1, len(self.offsets)) + pred.shape[2:], dtype=pred.dtype, device=pred.device)
        # final_output = np.empty((1, len(self.offsets)) + pred.shape[2:], dtype="float32")
        final_mask = torch.empty((1, len(self.offsets)) + pred.shape[2:], dtype=pred.dtype, device=pred.device)
        # final_mask = np.empty((1, len(self.offsets)) + pred.shape[2:], dtype="float32")
        for result in results_collected:
            write_slice = (slice(None), slice(None)) + tuple(slice(offs, None, strd) for offs, strd in zip(result[0], self.stride))
            final_output[write_slice] = result[1]
            final_mask[write_slice] = torch.from_numpy(result[2]).float().to(pred.device)
            # final_mask[write_slice] = result[2]

        # print("Took", time.time() - tick)
        # return [final_output, final_mask]
        return final_output, final_mask

        # return torch.from_numpy(final_output).cuda()

def IoU_worker(patches, offset, patch_target_size, stride, patch_dws_fact,
                  patch_shape, invert_masks=True):
    """
    Mask should be 1 where active
    """
    assert all([offs % strd == 0 for strd, offs in zip(stride, offset)])
    roll_offset = tuple(int(offs / strd) for strd, offs in zip(stride, offset))

    patch_shape = list(patch_shape) if isinstance(patch_shape, tuple) else patch_shape

    if not isinstance(patches, np.ndarray):
        rolled_patches = patches.roll(shifts=tuple(-offs for offs in roll_offset), dims=(0, 1, 2))
    else:
        rolled_patches = np.roll(patches, shift=tuple(-offs for offs in roll_offset), axis=(0, 1, 2))

    if not all([offs % dws == 0 for dws, offs in zip(patch_dws_fact, offset)]):
        print("Upsi daisy, offset should really be compatible with patch downscaling factor...")
        # assert all([offs % dws == 0 for dws, offs in zip(patch_dws_fact, offset)])
    patch_offset = tuple(int(offs / dws) for dws, offs in zip(patch_dws_fact, offset))

    # According to the passed passed patch_size, crop the patch accordingly:
    if patch_target_size is not None:
        assert len(patch_shape) == 3
        assert all([sh%2 != 0 or sh == 0 for sh in patch_target_size]), "Target patch-shape should be odd"
        crop_slice = [slice(None), slice(None), slice(None)]
        patch_shape = deepcopy(patch_shape)
        for dim in range(3):
            if patch_target_size[dim] == 0 or patch_target_size[dim] == patch_shape[dim]:
                crop_slice.append(slice(None))
            else:
                pad = int((patch_shape[dim] - patch_target_size[dim]) / 2)
                crop_slice.append(slice(pad,-pad))
                patch_shape[dim] = patch_target_size[dim]
        crop_slice = tuple(crop_slice)
        patches = patches[crop_slice]
        rolled_patches = rolled_patches[crop_slice]



    # Get crop slices patch_1:
    left_crop = [off if off > 0 else 0 for off in patch_offset]
    right_crop = [sh + off if off < 0 else None for off, sh in zip(patch_offset, patch_shape)]
    crop_slice_patches = (slice(None), slice(None), slice(None)) + tuple(
        slice(lft, rgt) for lft, rgt in zip(left_crop, right_crop))
    # Get crop slices patch_2:
    left_crop = [-off if off < 0 else 0 for off in patch_offset]
    right_crop = [sh - off if off > 0 else None for off, sh in zip(patch_offset, patch_shape)]
    crop_slice_rolled_patches = (slice(None), slice(None), slice(None)) + tuple(
        slice(lft, rgt) for lft, rgt in zip(left_crop, right_crop))


    # Crop and compute IoU:
    if invert_masks:
        output = compute_intersection_over_union(1. - patches[crop_slice_patches],
                                    1. - rolled_patches[crop_slice_rolled_patches])
    else:
        output = compute_intersection_over_union(patches[crop_slice_patches],
                                                 rolled_patches[crop_slice_rolled_patches])
    # if not isinstance(output, np.ndarray):
    #     output = output.cpu().numpy()

    left_crop = [-off if off < 0 else 0 for off in roll_offset]
    right_crop = [sh - off if off > 0 else None for off, sh in zip(roll_offset, patches.shape[:3])]
    valid_crop = tuple(slice(lft, rgt) for lft, rgt in zip(left_crop, right_crop))
    if isinstance(patches, np.ndarray):
        valid_predictions = np.zeros(patches.shape[:3], dtype='uint8')
    else:
        valid_predictions = torch.zeros(patches.shape[:3], dtype=torch.uint8).to(patches.get_device())

    valid_predictions[valid_crop] = 1

    return output, valid_predictions


def compute_IoU_targets(target_labels, offset, patch_target_size, stride, patch_dws_fact,
                        patch_shape, ignore_label=0):
    assert all([offs % strd == 0 for strd, offs in zip(stride, offset)])
    roll_offset = tuple(int(offs / strd) for strd, offs in zip(stride, offset))

    if not isinstance(target_labels, np.ndarray):
        rolled_labels = target_labels.roll(shifts=tuple(-offs for offs in roll_offset), dims=(0, 1, 2))
    else:
        rolled_labels = np.roll(target_labels, shift=tuple(-offs for offs in roll_offset), axis=(0, 1, 2))

    if not all([offs % dws == 0 for dws, offs in zip(patch_dws_fact, offset)]):
        print("Upsi daisy, offset should really be compatible with patch downscaling factor...")
        # assert all([offs % dws == 0 for dws, offs in zip(patch_dws_fact, offset)])

    output = target_labels == rolled_labels

    left_crop = [-off if off < 0 else 0 for off in roll_offset]
    right_crop = [sh - off if off > 0 else None for off, sh in zip(roll_offset, target_labels.shape[:3])]
    valid_crop = tuple(slice(lft, rgt) for lft, rgt in zip(left_crop, right_crop))
    if isinstance(target_labels, np.ndarray):
        valid_predictions = np.zeros(target_labels.shape[:3], dtype='uint8')
    else:
        valid_predictions = torch.zeros(target_labels.shape[:3] , dtype=torch.uint8).to(target_labels.get_device())
    valid_predictions[valid_crop] = 1

    valid_predictions[target_labels == ignore_label] = 0
    valid_predictions[rolled_labels == ignore_label] = 0

    return output, valid_predictions

