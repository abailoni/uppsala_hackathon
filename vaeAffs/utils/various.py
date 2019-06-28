from scipy.ndimage import zoom
import torch

def torch_tensor_zoom(array, *args,**kwargs):
    zoomed_array = zoom(array, *args, **kwargs)
    return torch.from_numpy(zoomed_array)
