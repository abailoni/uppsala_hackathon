import numpy as np

from inferno.io.transform import Transform
from vaeAffs.utils.affinitiy_utils import get_offset_locations
from scipy.ndimage import zoom

class SetVAETarget(Transform):
    def batch_function(self, batch):
        return [batch[0], np.copy(batch[0])]

class HackyHacky(Transform):
    # FIXME: super ugly, temp hack, add raw to target
    def batch_function(self, batch):
        return batch[:-1] + (np.concatenate([np.expand_dims(batch[0],0), batch[-1]], axis=0), )


class Downsample(Transform):
    def __init__(self, order=3,**super_kwargs):
        super(Downsample, self).__init__(**super_kwargs)
        self.order = order

    def image_function(self, image):
        # TODO: generalize factor
        image = zoom(image, 0.25, order=self.order)
        return image

class CreateMaskSeed(Transform):
    def batch_function(self, batch):
        assert len(batch) == 2
        raw, gt_segm = batch

        gt_shape = gt_segm.shape[-3:]
        # center_coord = [int(shp/2) for shp in gt_shape]
        seed_padding = (0,10,10)
        center_coord = [int(gt_shape[0]/2), seed_padding[1]+10, seed_padding[2]+10]

        # Look for point far away from boundary:
        # FIXME: change this shit to some DT stuff
        from copy import deepcopy
        current_pos = deepcopy(center_coord)
        while True:
            gt_slice = tuple(slice(pos-pad, pos+pad+1) for pos, pad in zip(current_pos, seed_padding))
            labels = np.unique(gt_segm[gt_slice])
            # TODO: generalize ignore label
            if labels.shape[0] == 1 and labels[0] != 0:
                break
            else:
                current_pos[1] += 1
                if current_pos[1]+seed_padding[1] > gt_shape[1]:
                    current_pos[1] = center_coord[1]
                    current_pos[2] += 1
                    if current_pos[2] + seed_padding[2] > gt_shape[2]:
                        raise RuntimeError("Seed not available...")

        input_mask = np.zeros_like(raw)
        gt_mask = np.zeros_like(raw)
        ignore_mask = np.zeros_like(raw)
        input_mask[gt_slice] = 1

        # Create GT Mask:
        gt_mask[gt_segm == labels[0]] = 1
        ignore_mask[gt_segm == 0] = 1
        return [raw, input_mask, gt_mask, ignore_mask]

class ApplyIgnoreMask(Transform):
    def batch_function(self, inputs_):
        pred, targets = inputs_
        assert len(targets) == 2
        mask = 1 - targets[1]
        pred = pred * mask
        targets[0] = targets[0] * mask
        return pred, targets[0]




class RandomZCrop(Transform):
    def build_random_variables(self):
        np.random.seed()
        self.set_random_variable('gamma',
                                 np.random.uniform(low=self.gamma_between[0],
                                                   high=self.gamma_between[1]))

    def batch_function(self, batch):
        return [self.random_crop(b) for b in batch]

class ComputeMeMask(Transform):
    def tensor_function(self, tensor):
        assert tensor.ndim == 3
        patch_shape = tensor.shape

        center_coord = tuple(
            int(patch_shape[i] / 2) for i in
            range(3))
        center_label = tensor[center_coord]
        # center_labels_repeated = np.tile(center_labels, patch_shape)
        me_masks = tensor != center_label

        return me_masks.astype(np.float32)

class RemoveThirdDimension(Transform):
    def tensor_function(self, tensor):
        return tensor[:,0]

class RemoveInvalidAffs(Transform):
    def tensor_function(self, tensor):
        nb_offsets = int(tensor.shape[0] / 2)
        return (1-tensor[:nb_offsets])*tensor[nb_offsets:]


class InvertTargets(Transform):
    def tensor_function(self, tensor):
        nb_offsets = int(tensor.shape[0] / 2)
        return (1-tensor[:nb_offsets])*tensor[nb_offsets:]


class RandomlyDownscale(Transform):
    def __init__(self, final_shape=(29,29),
                 downscale_factors=(1,2,4),
                 *super_args, **super_kwargs):
        # TODO: atm only working for 2d downscaling
        super(RandomlyDownscale, self).__init__(*super_args, **super_kwargs)
        self.dw_fact = downscale_factors
        self.final_shape = final_shape

    def build_random_variables(self, **kwargs):
        # self.set_random_variable("DS", np.random.randint(2) == 0)
        self.set_random_variable("DS", np.random.choice(list(self.dw_fact)))

    def batch_function(self, batch):
        assert len(batch) == 1
        self.build_random_variables()
        DS = self.get_random_variable("DS")
        # assert batch[0].shape[-1] % self.dw_fact == 0
        # assert batch[0].shape[-2] % self.dw_fact == 0
        new_batch = batch[0][...,::DS,::DS]
        shape0 = new_batch.shape[-2]
        shape1 = new_batch.shape[-1]
        out_shape = self.final_shape
        assert shape0 >= out_shape[0]
        assert shape1 >= out_shape[1]
        if shape0 > out_shape[0] or shape1 > out_shape[1]:
            off0 = np.random.randint(shape0 - out_shape[0])
            off1 = np.random.randint(shape1 - out_shape[1])
            new_batch = new_batch[..., off0:int(off0+out_shape[0]), off1:int(off1+out_shape[1])]

        if DS == 1:
            new_batch[:4]
        return [new_batch]
