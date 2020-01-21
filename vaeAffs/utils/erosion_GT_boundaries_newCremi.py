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
from segmfriends.utils.various import parse_data_slice
import os
import h5py

defected_slices = {
    "A": [],
    "B": ["23:25", "52:54"],
    "C": ["22:23", "82:83",
          "94:95, 785:, :500",
          "94:95, 1360:, 500:741", ],
    "0": ["76:77, :,  1070:", "122:123"],
    "1": ["122:123"],
    "2": []
}

copy_from_previous = {
    "A": [],
    "B": [],
    "C": [22, 82],
    "0": [122],
    "1": [122],
    "2": [],
}

# for sample in ["A", "B", "C"]:
for sample in ["C", "0", "1", "2"]:
    print("Sample", sample)

    data_path = os.path.join(get_trendytukan_drive_path(), "datasets/new_cremi/sample{}.h5".format(sample))

    with h5py.File(data_path, 'r') as f:
        print([atr for atr in f['volumes/labels']])
        #     glia = f['volumes/labels/glia'][:]
        # raw = f['volumes/raw'][:]
        GT = f['volumes/labels/neuron_ids'][:]
        glia = f['volumes/labels/glia'][:]

    from affogato.affinities import compute_affinities

    offsets = [
        [0, 1, 0],
        [0, 0, 1],
    ]
    print(GT.max())

    # affs: 0 boundary, 1 segment;
    # valid_mask: 1 is valid
    affs, affs_valid_mask = compute_affinities(GT.astype('int64'), offsets,
                                                  ignore_label=0,
                                                  have_ignore_label=True)

    # Where it is not valid, we should not predict a boundary label:
    affs[affs_valid_mask==0] = 1

    # Combine left and right affinities:
    segment_mask = np.logical_and(affs[0], affs[1])

    # This functions erode binary out_mask (segments 1, boundary 0)
    eroded_segment_mask = segment_mask.copy()
    for z in range(eroded_segment_mask.shape[0]):
        eroded_segment_mask[z] = vigra.filters.multiBinaryErosion(segment_mask[z], radius=2.)
    boundary_mask = np.logical_not(eroded_segment_mask)

    BOUNDARY_LABEL = 2
    DEFECTED_LABEL = 3
    out_mask = glia.copy()
    out_mask[boundary_mask] = BOUNDARY_LABEL

    # Mask defected slices:
    for slc in defected_slices[sample]:
        out_mask[parse_data_slice(slc)] = DEFECTED_LABEL


    # Copy GT from previous (to avoid weird connected components problems):
    for z in copy_from_previous[sample]:
        GT[z] = GT[z-1]

    print("Now writing...")

    from segmfriends.utils.various import writeHDF5
    writeHDF5(out_mask, data_path, 'volumes/labels/various_masks')
    writeHDF5(GT, data_path, 'volumes/labels/neuron_ids_fixed')
