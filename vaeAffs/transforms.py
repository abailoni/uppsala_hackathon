import numpy as np

from inferno.io.transform import Transform
from vaeAffs.utils.affinitiy_utils import get_offset_locations


class SetVAETarget(Transform):
    def batch_function(self, batch):
        return [batch[0], np.copy(batch[0])]

class HackyHacky(Transform):
    def batch_function(self, batch):
        return [batch[0], np.concatenate([np.expand_dims(batch[0],0), batch[1]], axis=0)]

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
    def __init__(self, downscale_factor=2, *super_args, **super_kwargs):
        # TODO: atm only working for 2d downscaling
        super(RandomlyDownscale, self).__init__(*super_args, **super_kwargs)
        self.dw_fact = downscale_factor

    def build_random_variables(self, **kwargs):
        self.set_random_variable("DS", np.random.randint(2) == 0)

    def batch_function(self, batch):
        assert len(batch) == 1
        self.build_random_variables()
        assert batch[0].shape[-1] % self.dw_fact == 0
        assert batch[0].shape[-2] % self.dw_fact == 0
        if self.get_random_variable("DS"):
            new_batch = [batch[0][...,::self.dw_fact,::self.dw_fact]]
        else:
            shape0 = batch[0].shape[-2]
            shape1 = batch[0].shape[-1]
            off0 = np.random.randint(shape0 - shape0/self.dw_fact)
            off1 = np.random.randint(shape1 - shape1/self.dw_fact)
            new_batch = [batch[0][..., off0:int(off0+shape0/self.dw_fact), off1:int(off1+shape1/self.dw_fact)]]
        return new_batch
