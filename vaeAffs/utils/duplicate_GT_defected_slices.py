import vigra
import numpy as np

"""
For each detected slice we use the GT of the slice before (in the case of two in a row, both has the same GT).
As na hack, we denote the pixel in the top left corner with 2147483646 to recognise them later (we should avoid to 
train without the previous slice)

"""

from vaeAffs.utils.path_utils import get_abailoni_hci_home_path, get_trendytukan_drive_path
import os
import h5py

defected_slices = {
    "A": [],
    "B": [15, 16, 44, 45],
    "C": [14, 74],
}

for sample in ["A", "B", "C"]:
    # -----------
    # LOAD data
    # -----------

    data_path = os.path.join(get_abailoni_hci_home_path(), "../ialgpu1_local_home/datasets/cremi/SOA_affinities/sample{}_train.h5".format(sample))

    old_raw = True
    legacy_mod = "_OLD" if sample == "B" and old_raw else ""
    with h5py.File(data_path, 'r') as f:
        GT = f['segmentations/groundtruth_plus_boundary'+legacy_mod][:]


    # Max int32 value - 1:
    DEFECTED_VALUE = 2147483647 - 1

    for slc_idx in defected_slices[sample]:
        GT[slc_idx] = DEFECTED_VALUE

    from segmfriends.utils.various import writeHDF5

    writeHDF5(GT, data_path, 'segmentations/groundtruth_plus_boundary_defectMod'+legacy_mod)
