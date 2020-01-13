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
import os
import h5py

for sample in ["A", "B", "C"]:

    data_path = os.path.join(get_abailoni_hci_home_path(), "../ialgpu1_local_home/datasets/cremi/SOA_affinities/sample{}_train.h5".format(sample))

    old_raw = True
    with h5py.File(data_path, 'r') as f:
        if sample == "B" and old_raw:
            GT = f['segmentations/groundtruth_fixed_OLD'][:]
            # raw = f['raw_old'][:]
        else:
            GT = f['segmentations/groundtruth_fixed'][:]
            # raw = f['raw'][:]

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

    # This functions erode binary mask (segments 1, boundary 0)
    eroded_segment_mask = segment_mask.copy()
    for z in range(eroded_segment_mask.shape[0]):
        eroded_segment_mask[z] = vigra.filters.multiBinaryErosion(segment_mask[z], radius=2.)
    boundary_mask = np.logical_not(eroded_segment_mask)

    # Max int32 value:
    BOUNDARY_VALUE = 2147483647

    modified_GT = GT.copy()
    modified_GT[boundary_mask] = BOUNDARY_VALUE

    print("Done")

    from segmfriends.utils.various import writeHDF5
    legacy_mod = "_OLD" if sample == "B" and old_raw else ""
    writeHDF5(modified_GT, data_path, 'segmentations/groundtruth_plus_boundary2'+legacy_mod)
