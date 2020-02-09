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

from scipy.ndimage import zoom

crp_slices = {
    "B": ":,:-1,:",
    "C": ":,:,:-1"
}




for sample in ["A", "B", "C", "0", "1", "2"]:
    print("Sample", sample)

    data_path = os.path.join(get_abailoni_hci_home_path(), "datasets/new_cremi/sample{}.h5".format(sample))

    if sample in crp_slices:
        crp_slc = parse_data_slice(crp_slices[sample])
    else:
        crp_slc = slice(None)

    with h5py.File(data_path, 'r+') as f:
        print([atr for atr in f['volumes/labels']])
        #     glia = f['volumes/labels/glia'][:]
        raw = f['volumes/raw'][crp_slc]
        GT = f['volumes/labels/neuron_ids_fixed'][crp_slc]
        # various_masks = f['volumes/labels/various_masks'][crp_slc]
        various_masks = f['volumes/labels/various_masks_noDefects'][crp_slc]

        print("Now writing...")
        # f['volumes/raw_2x'] = zoom(raw, (1, 0.5, 0.5), order=3)
        # f['volumes/labels/neuron_ids_fixed_2x'] = zoom(GT, (1, 0.5, 0.5), order=0)
        if 'volumes/labels/various_masks_noDefects_2x' in f:
            del f['volumes/labels/various_masks_noDefects_2x']
        f['volumes/labels/various_masks_noDefects_2x'] = zoom(various_masks, (1, 0.5, 0.5), order=0)

