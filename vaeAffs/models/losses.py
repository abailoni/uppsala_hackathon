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


from neurofire.models.unet.unet_3d import CONV_TYPES, Decoder, DecoderResidual, BaseResidual, Base, Output, Encoder, \
    EncoderResidual

import warnings


class EncodingLoss(nn.Module):
    def __init__(self, path_autoencoder_model):
        super(EncodingLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.soresen_loss = SorensenDiceLoss()

        assert isinstance(path_autoencoder_model, str)
        AE_model = torch.load(path_autoencoder_model)
        # Freeze the auto-encoder model:
        self.AE_model = AE_model
        for param in self.AE_model.parameters():
            param.requires_grad = False


    def forward(self, predictions, target):
        # FIXME: atm I hard-coded the dimensions
        patch_slice = (slice(None), slice(None), slice(0,1), slice(27,54), slice(27,54))
        patch_slice = (slice(None), slice(None), slice(0,1), slice(27,54), slice(27,54))
        target_patch = target
        # with torch.no_grad():
        #     target_embedded_patch = self.AE_model.encode(target_patch)[...,0]
        predicted_embedded_patch = predictions[:, :, [0], 13, 13]

        # with torch.no_grad():
        assert predictions.shape[1] % 2 == 0
        emb_vect_size = int(predictions.shape[1] / 2)
        z = self.AE_model.reparameterize(predicted_embedded_patch[:,:emb_vect_size], predicted_embedded_patch[:,emb_vect_size:])
        self.emb_prediction = self.AE_model.decode(z)

        # target_embedded_patch = target_embedded_patch.repeat(27, 27, 1, 1)
        # target_embedded_patch = target_embedded_patch.permute(2,3,0,1)
        # MSE = self.loss(predictions[:,:,0], target_embedded_patch)
        loss = self.soresen_loss(self.emb_prediction, target)

        # # Generate a random patch:
        self.random_prediction = self.AE_model.decode(torch.randn(predicted_embedded_patch[:,:emb_vect_size].shape).cuda())


        return loss


class VAE_loss(nn.Module):
    def __init__(self, model_kwargs=None):
        super(VAE_loss, self).__init__()
        self.reconstruction_loss = nn.BCELoss(reduction="sum")
        self.model_kwargs = model_kwargs
        # self.reconstruction_function.size_average = False
        # self.reconstruction_function = SorensenDiceLoss()
        # self.reconstruction_function = nn.MSELoss()

        self.pre_maxpool = None
        if model_kwargs.get("pre_maxpool") is not None:
            pre_maxpool = model_kwargs.get("pre_maxpool")
            self.pre_maxpool = nn.MaxPool3d(kernel_size=pre_maxpool,
                                            stride=pre_maxpool,
                                            padding=0)

    def forward(self, predictions, target):
        # x = target[:, :, 0]
        recon_x, mu, logvar = predictions

        if self.pre_maxpool is not None:
            target = self.pre_maxpool(target)

        # Reconstruction loss:
        # BCE = 0
        BCE = self.reconstruction_loss(recon_x, target)

        # BCE = self.reconstruction_function(recon_x, target)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        log_scalar("VAE_BCE_loss", BCE)
        log_scalar("VAE_KLD_loss", KLD)
        log_image("maxpooled_target", target)

        return BCE + KLD

class AffLoss(nn.Module):
    def __init__(self, loss_type="Dice"):
        super(AffLoss, self).__init__()
        if loss_type == "Dice":
            self.loss = SorensenDiceLoss()
        elif loss_type == "MSE":
            self.loss = nn.MSELoss()
        elif loss_type == "BCE":
            self.loss = nn.BCELoss()
        else:
            raise ValueError

    def forward(self, affinities, target):
        # gt_segm = target[:,0]
        if isinstance(target, (tuple, list)):
            # TODO: avoid computing all others affinities
            # For the stacked architecture, only use the last affinities
            target = target[-1]

        all_affs = target[:,1:]
        nb_offsets = int(all_affs.shape[1] / 2)
        target_affs, ignore_affs_mask = all_affs[:,:nb_offsets], all_affs[:,nb_offsets:]

        # Apply ignore mask:
        target_affs = 1 - target_affs
        affinities = 1 - affinities
        ignore_affs_mask = ignore_affs_mask == 0
        affinities[ignore_affs_mask] = 0
        target_affs[ignore_affs_mask] = 0
        return self.loss(affinities, target_affs)



from speedrun.log_anywhere import log_image, log_embedding, log_scalar

class PatchLoss(nn.Module):
    def __init__(self, model, loss_type="Dice", devices=(0,1)):
        super(PatchLoss, self).__init__()
        if loss_type == "Dice":
            self.loss = SorensenDiceLoss()
        elif loss_type == "MSE":
            self.loss = nn.MSELoss()
        elif loss_type == "BCE":
            self.loss = nn.BCELoss()
        else:
            raise ValueError

        # TODO: generalize
        self.devices = devices
        self.MSE_loss = nn.MSELoss()
        self.smoothL1_loss = nn.SmoothL1Loss()
        # TODO: use nn.BCEWithLogitsLoss()
        self.BCE = nn.BCELoss()
        self.soresen_loss = SorensenDiceLoss()

        from vaeAffs.models.vanilla_vae import VAE_loss
        self.VAE_loss = VAE_loss()

        self.model = model
        # assert isinstance(path_autoencoder_model, str)
        # # FIXME: this should be moved to the model, otherwise it's not saved!
        # self.AE_model = [torch.load(path_autoencoder_model),
        #                    torch.load(path_autoencoder_model),
        #                    torch.load(path_autoencoder_model),]
        #
        # for i in range(3):
        #     self.AE_model[i].set_min_patch_shape((5,29,29))
        #     # Freeze the auto-encoder model:
        #     # for param in self.AE_model[i].parameters():
        #     #     param.requires_grad = False
        #
        # # from vaeAffs.models.vanilla_vae import AutoEncoder
        # # self.AE_model = AutoEncoder(**autoencoder_kwargs)



    def forward(self, all_predictions, target):
        # Crop some z-slice to assure some context:
        # TODO: remove
        # pred_affs = all_predictions[0]
        # pred_affs = pred_affs[:,:,1:-1]
        # feat_pyr = all_predictions[1:]
        feat_pyr = all_predictions
        # feat_pyr = [prd[:,:,1:-1] for prd in feat_pyr]
        # target = target[:,:,1:-1]

        # TODO: parametrize
        patch_shape_orig = (5,15,15)
        # Extract random patch:
        full_target_shape = target.shape[-3:]

        raw = target[:,[0]]
        gt_segm = target[:,1]
        all_affs = target[:,2:]

        nb_offsets = int(all_affs.shape[1] / 2)
        boundary_affs, target_affs, ignore_affs_mask = all_affs[:,:nb_offsets-1], all_affs[:,[nb_offsets-1]], all_affs[:,[-1]]

        # # First compute dense-affinity loss:
        # ignore_affs_mask = 1 - ignore_affs_mask # Zero when ignored
        # target_affs = target_affs * ignore_affs_mask
        # pred_affs = pred_affs * ignore_affs_mask[:,:,:,::2,::2]
        # loss = 0
        # LOSS_FACTOR = 0.0
        # # loss = self.soresen_loss(pred_affs, target_affs[:,:,:,::2,::2]) * LOSS_FACTOR

        # return loss

        # Now loss from random patches:
        boundary_mask, _ = boundary_affs[:,:4].max(dim=1, keepdim=True)
        boundary_mask_long, _ = boundary_affs[:,4:].max(dim=1, keepdim=True)

        assert all(i % 2 ==  1 for i in patch_shape_orig), "Patch should be odd"

        all_emb_vectors = {}
        all_target_patches = {}
        all_ignore_masks = {}
        all_raw_patches = {}

        all_patch_scale_fct = [(1,1,1), (1,2,2), (1,4,4)]
        all_prediction_scale_fct = [(1,1,1), (1,2,2), (1,4,4)]

        all_strides = [(1,15,15), (1,18,18), (1,18,18)]

        gt_segm = gt_segm.unsqueeze(1)


        for level, patch_scale_fct, prediction_scale_fct, pred, stride in zip(range(3), all_patch_scale_fct, all_prediction_scale_fct,
                                                                         [feat_pyr[0], feat_pyr[1], feat_pyr[2]], all_strides):
            patch_shape = tuple(pt*fc for pt, fc in zip(patch_shape_orig, patch_scale_fct))

            # TODO assert dims!
            assert all([i <= j for i, j in zip(patch_shape, full_target_shape)]), "Prediction is too small"

            gt_patches, crop_offsets, nb_patches = extract_patches_torch(gt_segm, patch_shape, stride=stride,
                                  max_random_crop=(0,5,5))
            crop_offsets_emb = tuple( off+int(sh/2) for off, sh in zip(crop_offsets, patch_shape))

            center_labels, _, _ = extract_patches_torch(gt_segm, (1,1,1), stride=stride,
                                                        fixed_crop=crop_offsets_emb,
                                                        limit_patches_to=nb_patches)
            selected_boundary_map = boundary_mask if level <= 1 else boundary_mask_long
            is_on_boundary, _, _ = extract_patches_torch(selected_boundary_map, (1, 1, 1), stride=stride,
                                                     fixed_crop=crop_offsets_emb,
                                                     limit_patches_to=nb_patches)

            center_labels_repeated = center_labels.repeat(1, 1, *patch_shape)
            me_masks = gt_patches != center_labels_repeated

            # max_offsets = [ j-i for i, j in zip(patch_shape, full_target_shape)]

            # Ignore some additional pixels:
            ignore_masks = (gt_patches == 0)

            # Reject some patches:
            valid_patches = (center_labels != 0) & (is_on_boundary != 1)
            valid_batch_indices = np.argwhere(valid_patches[:, 0, 0, 0, 0].cpu().detach().numpy())[:, 0]

            if valid_batch_indices.shape[0] == 0:
                continue

            # Downscale masks:
            # dw_slice = (slice(None), slice(None)) + tuple(slice(None,None,dw_fct) for dw_fct in patch_scale_fct)
            # Make sure that the me-mask is zero (better split than merge)
            if all(fctr == 1 for fctr in patch_scale_fct):
                maxpool = Identity()
            else:
                maxpool = nn.MaxPool3d(kernel_size=patch_scale_fct,
                     stride=patch_scale_fct,
                     padding=0)

            # Save data:
            all_target_patches[level] = maxpool(me_masks[valid_batch_indices].float()).float()
            all_ignore_masks[level] = maxpool(ignore_masks[valid_batch_indices].float())

            # Compute patches for prediction (possibly downscaled):
            # TODO: is there something smarter to do here? ROIAlign style
            # The problem is that the CNN does not know exactly which of the targets patches to expect
            # (but if we stay away from boundary it should be fine...)
            pred_crop_offsets = tuple(int(off/pred_fctr) for off, pred_fctr in zip(crop_offsets_emb, prediction_scale_fct))
            pred_strides = tuple(int(strd/pred_fctr) for strd, pred_fctr in zip(stride, prediction_scale_fct))
            pred_patches, _, _ = extract_patches_torch(pred, (1, 1, 1), stride=pred_strides,
                                                     fixed_crop=pred_crop_offsets,
                                                    limit_patches_to=nb_patches)
            all_emb_vectors[level] = pred_patches[valid_batch_indices]

            # TODO: avoid all raw patches for few figs...
            raw_patches, _, _ = extract_patches_torch(raw, patch_shape, stride=stride,
                                                                         fixed_crop=crop_offsets)
            all_raw_patches[level] = maxpool(raw_patches[valid_batch_indices])

        # print([all_emb_vectors[lvl].shape[0] for lvl in range(3)])
        assert feat_pyr[0].shape[1] % 2 == 0
        emb_vect_size = int(feat_pyr[0].shape[1]/2)
        # all_predicted_patches = [self.model.AE_model.decode(self.model.AE_model.reparameterize(vect[:,:emb_vect_size,:,0,0], vect[:,emb_vect_size:,:,0,0])) for vect in all_emb_vectors]

        # # Take only first channel, since now we predict masks:
        # mu =  [all_emb_vectors[lvl][:,:emb_vect_size, 0, 0] for lvl in range(3)]
        # log_var =  [all_emb_vectors[lvl][:,emb_vect_size:, 0, 0] for lvl in range(3)]


        # all_pred_patches = [self.model.AE_model[lvl].decode(self.model.AE_model[lvl].reparameterize(
        #     mu[lvl],
        #     log_var[lvl],
        # ))[:,[0]] for lvl in range(3)]

        # for lvl in range(3):
        #     data_parallel()

        all_pred_patches = [data_parallel(self.model.AE_model[lvl], all_emb_vectors[lvl][:,:, 0, 0], self.devices) for lvl in range(3)]
        all_pred_patches = [all_pred_patches[lvl][:, [0]] for lvl in range(3)]

        loss = 0
        for lvl in range(3):
            # with torch.no_grad():
            # cloned_trg = all_target_patches[lvl].clone()
            # outputs_VAE  = self.model.AE_model[lvl].forward(cloned_trg)

            # loss = loss + self.VAE_loss(outputs_VAE, cloned_trg.clone()) / all_target_patches[lvl].shape[0]

            #
            # loss1 = self.MSE_loss(mu[lvl], target_mu)
            # loss2 = self.MSE_loss(log_var[lvl], target_log_var)
            # loss += loss1 + loss2

            pred, trg, ign = all_pred_patches[lvl], all_target_patches[lvl], all_ignore_masks[lvl].byte()
            if lvl == 2 or lvl == 1:
                pred = 1 - pred
                trg = 1 - trg
            btch_slc = list(np.random.randint(trg.shape[0], size=4)) if trg.shape[0] >= 4 else slice(0, 1)
            # btch_slc = slice(0, 1)
            log_image("ptc_trg_l{}".format(lvl), trg[btch_slc][:, 0, 2])
            # log_image("ptc_pred_trg_l{}".format(lvl), outputs_VAE[0][btch_slc][:, 0, 2])
            log_image("ptc_pred_l{}".format(lvl), pred[btch_slc][:, 0, 2])
            log_image("ptc_raw_l{}".format(lvl), all_raw_patches[lvl][btch_slc][:, 0, 2])
            log_image("ptc_ign_l{}".format(lvl), ign[btch_slc][:, 0, 2])

            # Apply ignore mask:
            pred[ign] = 0
            trg[ign] = 0
            loss_unet = data_parallel(self.loss, (pred, trg.float()), self.devices).mean()
            loss += loss_unet
            # print(loss_unet)
            log_scalar("loss_l{}".format(lvl), loss_unet)
            # loss += self.MSE_loss(pred, trg.float())


        # loss = torch.stack([self.soresen_loss(pred, trg)  for pred, trg in zip(all_predicted_patches, all_target_patches)]).sum()

        # # # Generate a random patch:
        # self.random_prediction = self.AE_model.decode(torch.randn(predicted_embedded_patch[:,:emb_vect_size].shape).cuda())
        # FIXME: understand how embedding work (only accept 2D tensor...)
        # log_embedding("patch_target_new", all_target_patches[0][:,:,:])



        # # Make full prediction:
        # full_prediction = predict_full_image(feat_pyr[:1,:emb_vect_size,0], self.AE_model.decode)
        # log_image("full_pred", full_prediction[0])

        return loss


class StackedPyrHourGlassLoss(nn.Module):
    def __init__(self, model, loss_type="Dice", model_kwargs=None, devices=(0,1)):
        super(StackedPyrHourGlassLoss, self).__init__()
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



    def forward(self, all_predictions, all_targets):
        mdl_kwargs = self.model_kwargs
        ptch_kwargs = mdl_kwargs["scale_spec_patchNet_kwargs"]
        nb_stacked = mdl_kwargs["nb_stacked"]

        # FIXME: temp hadk
        all_predictions = [all_predictions[:3], all_predictions[3:6], all_predictions[6:]]

        # Collect some patches:
        loss = 0
        LIMIT_STACK = 5
        for lvl in range(3):
            pred_lvl = all_predictions[lvl]
            target = all_targets[lvl]

            full_target_shape = target.shape[-3:]
            gt_segm = target[:,[0]]
            boundary_affs = target[:,1:]
            boundary_affs = 1 - boundary_affs
            if lvl >= 1:
                # Thinner boundaries for the higher res output:
                boundary_mask = boundary_affs[:,:4].max(dim=1, keepdim=True)[0]
            else:
                boundary_mask = boundary_affs[:,4:].max(dim=1, keepdim=True)[0]

            # For the moment all the predictions are at full-res:
            patch_shape_orig = ptch_kwargs[lvl]["patch_size"]
            assert all(i % 2 ==  1 for i in patch_shape_orig), "Patch should be odd"
            stride = tuple(ptch_kwargs[lvl]["patch_stride"])


            patch_dws_fact = deepcopy(ptch_kwargs[lvl]["patch_dws_fact"])
            pred_dws_fact = [1, 1, 1]

            pred_lvl = pred_lvl if isinstance(pred_lvl, list) else [pred_lvl]
            DOWNSCALING_FACT_BETWEEN_DEPTHS = 2
            for depth_factor, pred in zip(range(1,len(pred_lvl)+1), pred_lvl):
                patch_shape = tuple(pt*fc for pt, fc in zip(patch_shape_orig, patch_dws_fact))

                # TODO assert dims!
                assert all([i <= j for i, j in zip(patch_shape, full_target_shape)]), "Prediction is too small"

                gt_patches, crop_offsets, nb_patches = extract_patches_torch(gt_segm, patch_shape, stride=stride,
                                      max_random_crop=(0,5,5))
                crop_offsets_emb = tuple( off+int(sh/2) for off, sh in zip(crop_offsets, patch_shape))

                center_labels, _, _ = extract_patches_torch(gt_segm, (1,1,1), stride=stride,
                                                            fixed_crop=crop_offsets_emb,
                                                            limit_patches_to=nb_patches)

                is_on_boundary, _, _ = extract_patches_torch(boundary_mask, (1, 1, 1), stride=stride,
                                                         fixed_crop=crop_offsets_emb,
                                                         limit_patches_to=nb_patches)

                center_labels_repeated = center_labels.repeat(1, 1, *patch_shape)
                me_masks = gt_patches != center_labels_repeated

                # max_offsets = [ j-i for i, j in zip(patch_shape, full_target_shape)]

                # Ignore some additional pixels:
                ignore_masks = (gt_patches == 0)

                # Reject some patches:
                valid_patches = (center_labels != 0) & (is_on_boundary != 1)
                valid_batch_indices = np.argwhere(valid_patches[:, 0, 0, 0, 0].cpu().detach().numpy())[:, 0]

                if valid_batch_indices.shape[0] == 0:
                    continue

                # Downscale masks:
                # Make sure that the me-mask is zero (better split than merge)
                if all(fctr == 1 for fctr in patch_dws_fact):
                    maxpool = Identity()
                else:
                    maxpool = nn.MaxPool3d(kernel_size=patch_dws_fact,
                         stride=patch_dws_fact,
                         padding=0)

                # Final targets:
                patch_targets = maxpool(me_masks[valid_batch_indices].float()).float()
                patch_ignore_masks = maxpool(ignore_masks[valid_batch_indices].float()).byte()

                # Compute patches for prediction (possibly downscaled):
                # TODO: is there something smarter to do here? ROIAlign style
                # The problem is that the CNN does not know exactly which of the targets patches to expect
                # (but if we stay away from boundary it should be fine...)
                pred_crop_offsets = tuple(int(off/pred_fctr) for off, pred_fctr in zip(crop_offsets_emb, pred_dws_fact))
                pred_strides = tuple(int(strd/pred_fctr) for strd, pred_fctr in zip(stride, pred_dws_fact))
                pred_patches, _, _ = extract_patches_torch(pred, (1, 1, 1), stride=pred_strides,
                                                         fixed_crop=pred_crop_offsets,
                                                        limit_patches_to=nb_patches)
                pred_embed = pred_patches[valid_batch_indices][:, :, 0, 0, 0]

                # raw_patches, _, _ = extract_patches_torch(raw, patch_shape, stride=stride,
                #                                                              fixed_crop=crop_offsets)
                # all_raw_patches[level] = maxpool(raw_patches[valid_batch_indices])

                # Compute predicted patches:
                pred_patches = data_parallel(self.model.patch_models[lvl], pred_embed, self.devices)[:, [0]]
                # pred_patches = self.model.patch_models[lvl](pred_embed)[:, [0]]

                if lvl == 0 and depth_factor > 1:
                    pred_patches = 1 - pred_patches
                    patch_targets = 1 - patch_targets

                if depth_factor == 1:
                    # btch_slc = list(np.random.randint(patch_targets.shape[0], size=4)) if patch_targets.shape[0] >= 4 else slice(0, 1)
                    log_image("ptc_trg_l{}".format(lvl), patch_targets)
                    log_image("ptc_pred_l{}".format(lvl), pred_patches)
                    log_image("ptc_ign_l{}".format(lvl), patch_ignore_masks)

                # Apply ignore mask:
                pred_patches[patch_ignore_masks] = 0
                patch_targets[patch_ignore_masks] = 0
                with warnings.catch_warnings(record=True) as w:
                    loss_unet = data_parallel(self.loss, (pred_patches, patch_targets.float()), self.devices).mean()
                    # loss_unet = self.loss(pred_patches, patch_targets.float())
                # if lvl == 0:
                loss += loss_unet
                # else:
                #     loss += 0.00000001 * loss_unet
                if depth_factor == 1:
                    log_scalar("loss_l{}".format(lvl), loss_unet)

                # Increase/decrease the size of the patches and the prediction for the next eventual
                # depth level in the pyramid:
                patch_dws_fact = (patch_dws_fact[0], patch_dws_fact[1]*DOWNSCALING_FACT_BETWEEN_DEPTHS, patch_dws_fact[2]*DOWNSCALING_FACT_BETWEEN_DEPTHS,)
                pred_dws_fact = (pred_dws_fact[0], pred_dws_fact[1]*DOWNSCALING_FACT_BETWEEN_DEPTHS,
                                  pred_dws_fact[2]*DOWNSCALING_FACT_BETWEEN_DEPTHS)

            if lvl + 1 >= LIMIT_STACK:
                break

        return loss


class RefinedPyrUNetLoss(nn.Module):
    def __init__(self, model, loss_type="Dice", model_kwargs=None, devices=(0,1)):
        super(RefinedPyrUNetLoss, self).__init__()
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



    def forward(self, all_predictions, target):
        mdl_kwargs = self.model_kwargs
        ptch_kwargs = mdl_kwargs["scale_spec_patchNet_kwargs"]
        nb_preds = len(all_predictions)

        full_target_shape = target.shape[-3:]
        gt_segm = target[:, [0]]
        boundary_affs = target[:, 1:]
        boundary_affs = 1 - boundary_affs

        # Collect loss from some random patches:
        loss = 0
        for nb_patch_net in range(nb_preds):
            pred = all_predictions[nb_patch_net]
            kwargs = ptch_kwargs[nb_patch_net]

            # Collect options from config:
            patch_shape_input = kwargs.get("patch_size")
            assert all(i % 2 ==  1 for i in patch_shape_input), "Patch should be odd"
            patch_dws_fact = kwargs.get("patch_dws_fact", [1,1,1])
            stride = tuple(kwargs.get("patch_stride", [1,1,1]))
            pred_dws_fact = kwargs.get("pred_dws_fact", [1,1,1])
            precrop_target = kwargs.get("crop_targets")

            # if patch_dws_fact[1] <= 2:
                # Thinner boundaries for the higher res output:
            boundary_mask = boundary_affs[:, :].max(dim=1, keepdim=True)[0]
            # else:
            #     boundary_mask = boundary_affs[:, 4:8].max(dim=1, keepdim=True)[0]


            real_patch_shape = tuple(pt*fc for pt, fc in zip(patch_shape_input, patch_dws_fact))

            # TODO assert dims!
            assert all([i <= j for i, j in zip(real_patch_shape, full_target_shape)]), "Real-sized patch is too large!"

            gt_patches, crop_slice_targets, nb_patches = extract_patches_torch_new(gt_segm, real_patch_shape, stride=stride,
                                  max_random_crop=(0,5,5),
                                                                         precrop_target=precrop_target)

            # Make sure to crop some additional border and get the embedding correctly:
            crop_slice_emb = (slice(None), slice(None)) + tuple(slice(slc.start+int(sh/2), slc.stop) for slc, sh in zip(crop_slice_targets[2:], real_patch_shape))

            center_labels, _, _ = extract_patches_torch_new(gt_segm, (1,1,1), stride=stride,
                                                        crop_slice=crop_slice_emb,
                                                        limit_patches_to=nb_patches)

            is_on_boundary, _, _ = extract_patches_torch_new(boundary_mask, (1, 1, 1), stride=stride,
                                                     crop_slice=crop_slice_emb,
                                                     limit_patches_to=nb_patches)

            center_labels_repeated = center_labels.repeat(1, 1, *real_patch_shape)
            me_masks = gt_patches != center_labels_repeated

            # max_offsets = [ j-i for i, j in zip(patch_shape, full_target_shape)]

            # Ignore some additional pixels:
            ignore_masks = (gt_patches == 0)

            # Reject some patches:
            valid_patches = (center_labels != 0) & (is_on_boundary != 1)
            valid_batch_indices = np.argwhere(valid_patches[:, 0, 0, 0, 0].cpu().detach().numpy())[:, 0]

            if valid_batch_indices.shape[0] == 0:
                raise ValueError("ZERO!!!")
                continue

            # Downscale masks:
            # Make sure that the me-mask is zero (better split than merge)
            if all(fctr == 1 for fctr in patch_dws_fact):
                maxpool = Identity()
            else:
                maxpool = nn.MaxPool3d(kernel_size=patch_dws_fact,
                     stride=patch_dws_fact,
                     padding=0)

            # Final targets:
            patch_targets = maxpool(me_masks[valid_batch_indices].float()).float()
            patch_ignore_masks = maxpool(ignore_masks[valid_batch_indices].float()).byte()

            # Compute patches for prediction (possibly downscaled):
            # TODO: is there something smarter to do here? ROIAlign style
            # The problem is that the CNN does not know exactly which of the targets patches to expect
            # (but if we stay away from boundary it should be fine...)
            crop_slice_pred = (slice(None), slice(None)) + tuple(
                slice(int(slc.start / pred_fctr), int(slc.stop / pred_fctr))
                for slc, pred_fctr in zip(crop_slice_emb[2:], pred_dws_fact))
            pred_strides = tuple(int(strd/pred_fctr) for strd, pred_fctr in zip(stride, pred_dws_fact))
            pred_patches, _, _ = extract_patches_torch_new(pred, (1, 1, 1), stride=pred_strides,
                                                     crop_slice=crop_slice_pred,
                                                    limit_patches_to=nb_patches)
            pred_embed = pred_patches[valid_batch_indices][:, :, 0, 0, 0]

            # raw_patches, _, _ = extract_patches_torch(raw, patch_shape, stride=stride,
            #                                                              fixed_crop=crop_offsets)
            # all_raw_patches[level] = maxpool(raw_patches[valid_batch_indices])

            # Compute predicted patches:
            pred_patches = data_parallel(self.model.patch_models[nb_patch_net], pred_embed, self.devices)[:, [0]]
            # pred_patches = self.model.patch_models[lvl](pred_embed)[:, [0]]

            # if lvl == 0 and depth_factor > 1:
            #     pred_patches = 1 - pred_patches
            #     patch_targets = 1 - patch_targets

            # btch_slc = list(np.random.randint(patch_targets.shape[0], size=4)) if patch_targets.shape[0] >= 4 else slice(0, 1)
            log_image("ptc_trg_l{}".format(nb_patch_net), patch_targets)
            log_image("ptc_pred_l{}".format(nb_patch_net), pred_patches)
            log_image("ptc_ign_l{}".format(nb_patch_net), patch_ignore_masks)

            # Apply ignore mask:
            pred_patches[patch_ignore_masks] = 0
            patch_targets[patch_ignore_masks] = 0
            with warnings.catch_warnings(record=True) as w:
                loss_unet = data_parallel(self.loss, (pred_patches, patch_targets.float()), self.devices).mean()
                # loss_unet = self.loss(pred_patches, patch_targets.float())

            # if lvl == 0:
            loss += loss_unet
            # else:
            #     loss += 0.00000001 * loss_unet
            log_scalar("loss_l{}".format(nb_patch_net), loss_unet)
            log_scalar("nb_patches_l{}".format(nb_patch_net), pred_patches.shape[0])
            # log_scalar("avg_targets_l{}".format(nb_patch_net), patch_targets.float().mean())


        return loss



def extract_patches_torch_new(tensor, shape, stride, precrop_target=None, max_random_crop=None,
                          # downscale_fctr=None,
                          crop_slice=None,
                          limit_patches_to=None,
                          reshape_to_batch_dim=True):
    assert tensor.dim() == 4 or tensor.dim() == 5
    dim = tensor.dim() - 2
    assert len(shape) == dim and len(stride) == dim
    if crop_slice is not None:
        assert max_random_crop is None and precrop_target is None
    if precrop_target is not None:
        assert len(precrop_target) == dim
        assert all([isinstance(sl, (tuple, list)) for sl in precrop_target]) and all([len(sl) == 2 for sl in precrop_target])
    else:
        precrop_target = [(0, 0) for _ in range(dim)]

    max_random_crop = [0 for _ in range(dim)] if max_random_crop is None else max_random_crop
    assert len(max_random_crop) == dim

    # if downscale_fct is not None:
    #     assert len(downscale_fct) == dim

    if limit_patches_to is not None:
        assert len(limit_patches_to) == dim

    # Pick a random crop:
    if crop_slice is None:
        rnd_crop = [np.random.randint(max_offs+1) for max_offs in max_random_crop]
        crop_slice = (slice(None), slice(None)) + tuple(slice(precrop[0]+off, full_shp-precrop[1]) for off, precrop, full_shp in zip(rnd_crop, precrop_target, tensor.shape[2:]))

    # Unfold it:
    tensor = tensor[crop_slice]
    N, C = tensor.shape[:2]
    for d in range(dim):
        tensor = tensor.unfold(d+2, size=shape[d],step=stride[d])
    # Reshape:
    nb_patches = tensor.shape[2:2+len(shape)]
    # Along each dimension, we make sure to keep only a specific number of patches (not more):
    # this assures compatibility with other patches already extracted
    if limit_patches_to is not None:
        actual_limits  = tuple( lim if lim<nb else nb for nb, lim in zip(nb_patches, limit_patches_to))
        valid_patch_slice = (slice(None), slice(None)) + tuple(slice(None,lim) for lim in actual_limits)
        tensor = tensor[valid_patch_slice]
        nb_patches = actual_limits
    # Reshape
    tensor = tensor.contiguous().view(N,C,-1,*shape)
    if reshape_to_batch_dim:
        tensor = tensor.permute(0,2,1,*range(3,3+dim)).contiguous().view(-1,C,*shape)
    else:
        tensor = tensor.permute(0, 1, *range(3, 3 + dim), 2).contiguous()

    # if downscale_fct is not None:
    #     # TODO: use MaxPool instead?
    #     for d, dw in enumerate(downscale_fct):
    #         slc = tuple(slice(None) for _ in range(2+d)) + (slice(None,None,dw),)
    #         tensor = tensor[slc]

    return tensor, crop_slice, nb_patches

def extract_patches_torch(tensor, shape, stride, fixed_crop=None, max_random_crop=None,
                          # downscale_fctr=None,
                          limit_patches_to=None,
                          reshape_to_batch_dim=True):
    assert tensor.dim() == 4 or tensor.dim() == 5
    dim = tensor.dim() - 2
    assert len(shape) == dim and len(stride) == dim
    if fixed_crop is not None:
        assert len(fixed_crop) == dim
        assert max_random_crop is None
    else:
        fixed_crop = tuple(0 for _ in range(dim)) if fixed_crop is None else fixed_crop


    max_random_crop = tuple(0 for _ in range(dim)) if max_random_crop is None else max_random_crop
    assert len(max_random_crop) == dim

    # if downscale_fct is not None:
    #     assert len(downscale_fct) == dim

    if limit_patches_to is not None:
        assert len(limit_patches_to) == dim

    # Pick a random crop:
    rnd_crop = tuple(fx_offs + np.random.randint(max_offs+1) for max_offs, fx_offs in zip(max_random_crop, fixed_crop))
    crop_slice = (slice(None), slice(None)) + tuple(slice(off,None) for off in rnd_crop)

    # Unfold it:
    tensor = tensor[crop_slice]
    N, C = tensor.shape[:2]
    for d in range(dim):
        tensor = tensor.unfold(d+2, size=shape[d],step=stride[d])
    # Reshape:
    nb_patches = tensor.shape[2:2+len(shape)]
    if limit_patches_to is not None:
        actual_limits  = tuple( lim if lim<nb else nb for nb, lim in zip(nb_patches, limit_patches_to))
        valid_patch_slice = (slice(None), slice(None)) + tuple(slice(None,lim) for lim in actual_limits)
        tensor = tensor[valid_patch_slice]
        nb_patches = actual_limits
    # Reshape
    tensor = tensor.contiguous().view(N,C,-1,*shape)
    if reshape_to_batch_dim:
        tensor = tensor.permute(0,2,1,*range(3,3+dim)).contiguous().view(-1,C,*shape)
    else:
        tensor = tensor.permute(0, 1, *range(3, 3 + dim), 2).contiguous()

    # if downscale_fct is not None:
    #     # TODO: use MaxPool instead?
    #     for d, dw in enumerate(downscale_fct):
    #         slc = tuple(slice(None) for _ in range(2+d)) + (slice(None,None,dw),)
    #         tensor = tensor[slc]

    return tensor, rnd_crop, nb_patches


def predict_full_image(embeddings, decoder, patch_shape=(27, 27), stride=3):
    assert len(patch_shape) == 2
    assert len(embeddings.shape) == 4
    pred_shape = embeddings.shape[-2:]

    assert all(i % 2 == 1 for i in patch_shape), "Patch should be odd"
    assert all([i <= j for i, j in zip(patch_shape, pred_shape)]), "Prediction is too small"

    def unfold_and_refold(input, normalize=True):
        # For the moment we do not overlap them:
        assert all([j % i == 0 for i, j in zip(patch_shape, pred_shape)]), "Patch should fit the image!"
        nb_channels = input.shape[1]
        unfolded = nn.functional.unfold(input[:,:,13:-13,13:-13], kernel_size=(1,1), stride=stride) # (N, C, nb_patches)
        batch_size = unfolded.shape[0]
        nb_patches = unfolded.shape[2]
        unfolded = unfolded.permute(0,2,1).contiguous().view(-1, nb_channels, 1)

        with torch.no_grad():
            decoded = decoder(unfolded)
        # Get rid of Z dim:
        decoded = decoded[:,:,0]
        decoded_shape = decoded.shape # (N * nb_patches, C, X, Y)
        # Bring it to the shape expected by fold:
        decoded_reshaped = decoded.view(batch_size, nb_patches, *decoded_shape[1:]).permute(0,2,3,4,1).view(batch_size,-1,nb_patches)
        refolded = nn.functional.fold(decoded_reshaped, kernel_size=(27, 27), output_size=pred_shape, stride=stride)

        if normalize:
            normalization = nn.functional.fold(torch.ones_like(decoded_reshaped), kernel_size=(27, 27), output_size=pred_shape, stride=stride)
            refolded /= normalization
        return refolded


    folded = unfold_and_refold(embeddings)
    return folded




def unfold_3d(x, *args, **kwargs):
    unfolded = []
    for z in range(x.shape[2]):
        unfolded_slice = nn.functional.unfold(x[:,:,z], *args, **kwargs)
        assert unfolded_slice.shape[-1] == 1, "kernel size does not match! Trying to unfold a patch of size {} with kernel {}".format(x.shape[-1], kwargs["kernel_size"])
        unfolded.append(unfolded_slice)
    x = torch.cat(unfolded, dim=2)
    return x

def fold_3d(x, *args, **kwargs):
    folded = []
    for z in range(x.shape[2]):
        folded_slice = nn.functional.fold(x[:,:,[z]], *args, **kwargs)
        folded.append(folded_slice)
    x = torch.stack(folded, dim=2)
    return x

# class AutoEncoderSkeleton(nn.Module):
#     def __init__(self, encoders, base, decoders, output,
#                  unfold_size=1,
#                  final_activation=None):
#         super(AutoEncoderSkeleton, self).__init__()
#         assert isinstance(encoders, list)
#         assert isinstance(decoders, list)
#
#         assert len(encoders) == len(decoders), "%i, %i" % (len(encoders), len(decoders))
#         assert isinstance(base, list)
#         self.encoders = nn.ModuleList(encoders)
#         self.decoders = nn.ModuleList(decoders)
#
#         self.base = nn.ModuleList(base)
#         self.output = output
#         if isinstance(final_activation, str):
#             self.final_activation = getattr(nn, final_activation)()
#         elif isinstance(final_activation, nn.Module):
#             self.final_activation = final_activation
#         elif final_activation is None:
#             self.final_activation = None
#         else:
#             raise NotImplementedError
#
#         assert isinstance(unfold_size, int)
#         self.unfold_size = unfold_size
#
#     def encode(self, input_):
#         x = input_
#         encoder_out = []
#         # apply encoders and remember their outputs
#         for encoder in self.encoders:
#             x = encoder(x)
#             encoder_out.append(x)
#
#         x = unfold_3d(x, kernel_size=self.unfold_size)
#
#         encoded_variable = self.base[0](x)
#
#         return encoded_variable
#
#     def decode(self, encoded_variable):
#         x = self.base[1](encoded_variable)
#         x = fold_3d(x, output_size=(self.unfold_size, self.unfold_size), kernel_size=self.unfold_size)
#
#         # apply decoders
#         max_level = len(self.decoders) - 1
#         for level, decoder in enumerate(self.decoders):
#             # the decoder gets input from the previous decoder and the encoder
#             # from the same level
#             x = decoder(x)
#
#         x = self.output(x)
#         if self.final_activation is not None:
#             x = self.final_activation(x)
#         return x
#
#     def forward(self, input_):
#         encoded_variable = self.encode(input_)
#         x = self.decode(encoded_variable)
#         return x
#
#
# class AutoEncoder(AutoEncoderSkeleton):
#     """
#     3D U-Net architecture.
#     """
#
#     def __init__(self,
#                  in_channels,
#                  initial_num_fmaps,
#                  fmap_growth,
#                  latent_variable_size=128,
#                  scale_factor=2,
#                  final_activation='auto',
#                  conv_type_key='vanilla',
#                  unfold_size=1,
#                  add_residual_connections=False):
#         """
#         Parameter:
#         ----------
#         in_channels (int): number of input channels
#         out_channels (int): number of output channels
#         initial_num_fmaps (int): number of feature maps of the first layer
#         fmap_growth (int): growth factor of the feature maps; the number of feature maps
#         in layer k is given by initial_num_fmaps * fmap_growth**k
#         final_activation:  final activation used
#         scale_factor (int or list / tuple): upscale / downscale factor (default: 2)
#         final_activation:  final activation used (default: 'auto')
#         conv_type_key: convolutin type
#         """
#         # validate conv-type
#         assert conv_type_key in CONV_TYPES, conv_type_key
#         conv_type = CONV_TYPES[conv_type_key]
#
#         # validate scale factor
#         assert isinstance(scale_factor, (int, list, tuple))
#         self.scale_factor = [scale_factor] * 3 if isinstance(scale_factor, int) else scale_factor
#         assert len(self.scale_factor) == 1
#         # NOTE individual scale factors can have multiple entries for anisotropic sampling
#         assert all(isinstance(sfactor, (int, list, tuple))
#                    for sfactor in self.scale_factor)
#
#         # Set attributes
#         self.in_channels = in_channels
#         self.out_channels = in_channels
#
#         conv_type = CONV_TYPES[conv_type_key]
#         decoder_type = DecoderResidual if add_residual_connections else Decoder
#         encoder_type = EncoderResidual if add_residual_connections else Encoder
#         base_type = BaseResidual if add_residual_connections else Base
#
#         # Build encoders with proper number of feature maps
#         f0e = initial_num_fmaps
#         f1e = initial_num_fmaps * fmap_growth
#         f2e = initial_num_fmaps * fmap_growth ** 2
#         encoders = [
#             encoder_type(in_channels, f0e, 3, self.scale_factor[0], conv_type=conv_type),
#             # encoder_type(f0e, f1e, 3, self.scale_factor[1], conv_type=conv_type),
#             # encoder_type(f1e, f2e, 3, self.scale_factor[2], conv_type=conv_type)
#         ]
#
#         # Build base
#         # number of base output feature maps
#         # f0b = initial_num_fmaps * fmap_growth ** 3
#         base_pre = nn.Conv1d(f0e * unfold_size * unfold_size, latent_variable_size, 1, 1, 0)
#         base_post = nn.Conv1d(latent_variable_size, f0e * unfold_size * unfold_size, 1, 1, 0)
#         base = [base_pre, base_post]
#
#         # Build decoders (same number of feature maps as MALA)
#         f2d = initial_num_fmaps * fmap_growth ** 2
#         f1d = initial_num_fmaps * fmap_growth
#         f0d = initial_num_fmaps
#         decoders = [
#             # decoder_type(f2e, f2d, 3, self.scale_factor[2], conv_type=conv_type),
#             # decoder_type(f1d, f1d, 3, self.scale_factor[1], conv_type=conv_type),
#             decoder_type(f0d, f0d, 3, self.scale_factor[0], conv_type=conv_type)
#         ]
#
#         # Build output
#         output = Output(f0d, in_channels, 3)
#         # Parse final activation
#         if final_activation == 'auto':
#             final_activation = nn.Sigmoid() if in_channels == 1 else nn.Softmax2d()
#
#         # Build the architecture
#         super(AutoEncoder, self).__init__(encoders=encoders,
#                                           base=base,
#                                           decoders=decoders,
#                                           output=output,
#                                           final_activation=final_activation,
#                                           unfold_size=unfold_size)
#
