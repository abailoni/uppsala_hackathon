import numpy as np

from inferno.io.transform import Transform
from vaeAffs.utils.affinitiy_utils import get_offset_locations


class SetVAETarget(Transform):
    def batch_function(self, batch):
        return [batch[0], np.copy(batch[0])]


class RemoveThirdDimension(Transform):
    def tensor_function(self, tensor):
        return tensor[:,0]

class RemoveInvalidAffs(Transform):
    def tensor_function(self, tensor):
        nb_offsets = int(tensor.shape[0] / 2)
        return (1-tensor[:nb_offsets])*tensor[nb_offsets:]
