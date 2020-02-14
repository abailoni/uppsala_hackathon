import numpy as np
from nifty.external import generate_opensimplex_noise

from scipy.ndimage import zoom

import sys
import os

from cremi import Annotations, Volume
from cremi.io import CremiFile


# Import some cremi tools for back-aligning and preparing the submission:
from ..utils.path_utils import get_abailoni_hci_home_path
sys.path += [
    os.path.join(get_abailoni_hci_home_path(), "python_libraries/cremi_tools"), ]
from cremi_tools.alignment import backalign_segmentation
from cremi_tools.alignment.backalign import bounding_boxes as magic_bboxes



import segmfriends.utils.various as segm_utils

shape_padded_aligned_datasets = {
    "A+": (200, 3727, 3505),
    "B+": (200, 3832, 5455),
    "C+": (200, 3465, 3668)
}


def prepare_submission(sample, path_segm, inner_path_segm,
                       path_bbox_slice, ds_factor=None):
    """

    :param path_segm:
    :param inner_path_segm:
    :param path_bbox_slice: path to the csv file
    :param ds_factor: for example (1, 2, 2)
    """


    segm = segm_utils.readHDF5(path_segm, inner_path_segm)

    bbox_data = np.genfromtxt(path_bbox_slice, delimiter=';', dtype='int')
    assert bbox_data.shape[0] == segm.ndim and bbox_data.shape[1] == 2
    # bbox_slice = tuple(slice(b_data[0], b_data[1]) for b_data in bbox_data)

    if ds_factor is not None:
        assert len(ds_factor) == segm.ndim
        segm = zoom(segm, ds_factor, order=0)

    padding = tuple((slc[0], shp - slc[1]) for slc, shp in zip(bbox_data, shape_padded_aligned_datasets[sample]))
    padded_segm = np.pad(segm, pad_width=padding, mode="constant")

    # Apply Constantin crop and then backalign:
    cropped_segm = padded_segm[magic_bboxes[sample]]
    tmp_file = path_segm.replace(".h5", "_submission_temp.hdf")
    backalign_segmentation(sample, cropped_segm, tmp_file,
                           key="temp_data",
                           postprocess=False)

    # Create a CREMI-style file ready to submit:
    final_submission_path = path_segm.replace(".h5", "_submission.hdf")
    file = CremiFile(final_submission_path, "w")

    # Write volumes representing the neuron and synaptic cleft segmentation.
    backaligned_segm = segm_utils.readHDF5(tmp_file, "temp_data")
    neuron_ids = Volume(backaligned_segm.astype('uint64'), resolution=(40.0, 4.0, 4.0),
                        comment="Emb-submission")

    file.write_neuron_ids(neuron_ids)
    file.close()

    os.remove(tmp_file)


def add_opensimplex_noise_to_affs(affinities, scale_factor,
                            mod='merge-biased',
                            target_affs='all',
                            seed=None
                            ):
    affinities = affinities.copy()

    if target_affs == 'short':
        noise_slc = slice(0, 3)
    elif target_affs == 'long':
        noise_slc = slice(3, None)
    elif target_affs == "all":
        noise_slc = slice(None)
    else:
        raise ValueError


    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def logit(x, clip=True):
        if clip:
            x = add_epsilon(x)
        return np.log(x / (1. - x))

    def add_epsilon(affs, eps=1e-2):
        p_min = eps
        p_max = 1. - eps
        return (p_max - p_min) * affs + p_min

    # Generate noise:
    shape = affinities[noise_slc].shape

    large_ft_size = np.array((1., 3., 50., 50.))
    large_scale_noise = (generate_opensimplex_noise(shape, seed=seed, features_size=large_ft_size, number_of_threads=8)
                         +1.0) / 2.0
    fine_ft_size = np.array((1., 3., 20., 20.))
    fine_scale_noise = (generate_opensimplex_noise(shape, seed=seed, features_size=fine_ft_size, number_of_threads=8)
                        + 1.0) / 2.0

    # Combine large and fine features:
    # TODO: more or simplify?
    large_scale, fine_scale = 10, 5
    simplex_noise = (large_scale_noise * large_scale + fine_scale_noise * fine_scale) / (large_scale + fine_scale)

    if mod == "merge-biased":
        noisy_affs = sigmoid(logit(affinities[noise_slc]) + scale_factor * logit(np.maximum(simplex_noise, 0.5)))
    elif mod == "split-biased":
        noisy_affs = sigmoid(logit(affinities[noise_slc]) + scale_factor * logit(np.minimum(simplex_noise, 0.5)))
    elif mod == "unbiased":
        noisy_affs = sigmoid(logit(affinities[noise_slc]) + scale_factor * logit(simplex_noise))
    else:
        raise ValueError("Accepted mods are add or subtract")

    affinities[noise_slc] = noisy_affs

    return affinities



