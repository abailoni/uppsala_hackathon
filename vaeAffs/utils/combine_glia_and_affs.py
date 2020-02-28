import vigra
import numpy as np

# TODO:
# - first we compute boundary by combining offsets (-1, 0) and (0, -1)
# - we need to make sure to ignore the boundary with label 0 if possible
# - then we erode these segments
# - call this as a specific boundary-label

"""
The difference with just computing a boundary-patch with the usual method is that in this way also the other pixels inside segments are
trained to predict a conservative boundary margin.

This could create an effect with thin processes, but apparently it doesn't
"""

# -----------
# LOAD data
# -----------
from vaeAffs.utils.path_utils import get_abailoni_hci_home_path, get_trendytukan_drive_path
from segmfriends.utils.various import parse_data_slice, readHDF5, writeHDF5
import os
import json
import h5py

from scipy.ndimage import zoom



project_dir = os.path.join(get_trendytukan_drive_path(),"projects/pixel_embeddings")

EXP_NAMES = [
    ("v4_onlySparseAffs_eff", True),
    ("v4_addSparseAffs_eff", True)
]

glia_mask_exp = "v4_onlyTrainGlia_eff"
offsets_file_name = "sparse_affs_v4.json"
combination_method = 2



def compute_boundary_mask_from_label_image(affinities,
                                           glia_mask,
                                           offsets,
                                           invert_affs=False,
                                           combination_method=1,
                                           pad_mode='constant',
                                           pad_constant_values=0,
                                           set_invalid_values_to=-1):
    """
    Faster than the nifty version, but does not check the actual connectivity of the segments (no rag is
    built). A non-local edge could be cut, but it could also connect not-neighboring segments.
b
    It returns a boundary mask (1 on boundaries, 0 otherwise). To get affinities reverse it.

    :param offsets: numpy array
        Example: [ [0,1,0], [0,0,1] ]

    :param return_boundary_affinities:
        if True, the output shape is (len(axes, z, x, y)
        if False, the shape is       (z, x, y)

    :param channel_affs: accepted options are 0 or -1

    :param background_value: if either one of the two pixels is equal to background_value, then the edge is
            labelled as boundary
    """
    assert affinities.ndim == 4
    if glia_mask.ndim > 3:
        assert glia_mask.shape[0] == 1 and glia_mask.ndim == 4
        glia_mask = glia_mask[0]
    assert glia_mask.ndim == 3
    assert len(offsets) == affinities.shape[0]

    # Get mask of invalid values:
    invalid_affs_mask = np.logical_or(affinities < 0., affinities > 1.)
    invalid_glia_mask = np.logical_or(glia_mask < 0., glia_mask > 1.)

    # Set invalid values temporaly to zero:
    affinities[invalid_affs_mask] = 0
    glia_mask[invalid_glia_mask] = 0

    if invert_affs:
        affinities = 1. - affinities

    if isinstance(offsets, list):
        offsets = np.array(offsets)

    print("Averaged glia mask (should be close to zero): ", glia_mask.mean())

    padding = [[0,0] for _ in range(3)]
    for ax in range(3):
        padding[ax][0] = np.abs(np.minimum(offsets[:, ax].min(), 0))
        padding[ax][1] = np.maximum(offsets[:,ax].max(), 0)

    if pad_mode == 'edge':
        padded_glia_mask = np.pad(glia_mask, pad_width=padding, mode=pad_mode)
    elif pad_mode == 'constant':
        padded_glia_mask = np.pad(glia_mask, pad_width=padding, mode=pad_mode, constant_values=pad_constant_values)
    else:
        raise NotImplementedError
    crop_slices = tuple([slice(padding[ax][0], padded_glia_mask.shape[ax]-padding[ax][1]) for ax in range(3)])

    out_affs = []
    for nb_off, offset in enumerate(offsets):
        print("{} ".format(nb_off), end="", flush=True)
        rolled_glia_mask = padded_glia_mask.copy()
        for ax, offset_ax in enumerate(offset):
            if offset_ax!=0:
                rolled_glia_mask = np.roll(rolled_glia_mask, -offset_ax, axis=ax)
        minim_glia = np.minimum(rolled_glia_mask, padded_glia_mask)[crop_slices]
        if combination_method == 1:
            out_affs.append(minim_glia**2 + affinities[nb_off] * (1.-minim_glia))
        elif combination_method == 2:
            out_affs.append((minim_glia ** 2 + affinities[nb_off]) / (1 + minim_glia))
        else:
            raise ValueError

        # Reset invalid values:
        if invert_affs:
            out_affs[-1] = 1. - out_affs[-1]
        out_affs[-1][invalid_glia_mask] = set_invalid_values_to

    print("")
    out_affs = np.stack(out_affs)

    # Reset invalid values:
    out_affs[invalid_affs_mask] = set_invalid_values_to

    return out_affs


def load_offsets():
    offsets_path = os.path.join(get_abailoni_hci_home_path(),
                              "pyCharm_projects/uppsala_hackathon/experiments/cremi/offsets",
                              offsets_file_name)
    assert os.path.exists(offsets_path)
    with open(offsets_path, 'r') as f:
        return json.load(f)

offsets = load_offsets()

for sample in ["C"]:
    glia_prediction_path =os.path.join(project_dir, glia_mask_exp, "predictions_sample_{}.h5".format(sample))
    print("Loading glia for sample ", sample)
    glia_mask = readHDF5(glia_prediction_path, "glia_mask")

    for exp_name, invert in EXP_NAMES:
        pred_dir = os.path.join(project_dir, exp_name)

        for item in os.listdir(pred_dir):
            if os.path.isfile(os.path.join(pred_dir, item)):
                filename = item
                if not filename.endswith(".h5") or filename.startswith(".") or not filename.startswith("predictions_sample"):
                    continue
                pred_file = os.path.join(pred_dir, filename)

                # Load glia mask and predictions:
                # TODO: add crop slice
                print("Loading affs for ", exp_name)
                affs = readHDF5(pred_file, "data")

                print("Computing...")
                new_affs = compute_boundary_mask_from_label_image(affs,
                                                       glia_mask,
                                                      invert_affs=invert,
                                                       offsets=offsets,
                                                       combination_method=combination_method,
                                                      set_invalid_values_to=-1)


                print("Saving...")
                writeHDF5(new_affs, pred_file, "affs_plus_glia_{}".format(combination_method))



