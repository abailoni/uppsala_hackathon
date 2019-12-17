import vaeAffs

import nifty
import numpy as np
import vigra



import os
from vaeAffs.utils.path_utils import get_abailoni_hci_home_path
import h5py
sample = "A"
data_path = os.path.join(get_abailoni_hci_home_path(), "../ialgpu1_local_home/datasets/cremi/SOA_affinities/sample{}_train.h5".format(sample))
crop_slice = ":,20:30,300:600,300:600"
from segmfriends.utils.various import parse_data_slice

crop_slice = parse_data_slice(crop_slice)
with h5py.File(data_path, 'r') as f:
    affs = f['predictions']['full_affs'][crop_slice]
    raw = f['raw'][crop_slice[1:]]

offsets = [[-1, 0, 0],
  [0, -1, 0],
  [0, 0, -1],
  [-2, 0, 0],
  [0, -3, 0],
  [0, 0, -3],
  [-3, 0, 0],
  [0, -9, 0],
  [0, 0, -9],
  [-4, 0, 0],
  [0, -27, 0],
  [0, 0, -27]]



from affogato.affinities import compute_multiscale_affinities, compute_affinities
from affogato.segmentation import compute_mws_segmentation

_ ,mask = compute_affinities(np.zeros_like(raw, dtype='int64'), offsets)

print("Total valid edges: ", mask.sum())

# Invert affs:
affs[3:] = 1. - affs[3:]
import time
tick = time.time()
final_segm = compute_mws_segmentation(affs, offsets, number_of_attractive_channels=3,
                             strides=None, randomize_strides=False,
                                      algorithm='seeded',
                             mask=None, initial_coordinate=(4,150,150))
print("Time seeded: ", time.time() - tick)

# tick = time.time()
# _ = compute_mws_segmentation(affs, offsets, number_of_attractive_channels=3,
#                              strides=None, randomize_strides=False,
#                              mask=None, initial_coordinate=(0,0,70))
# print("Time normal: ", time.time() - tick)



import segmfriends.vis as vis
fig, ax = vis.get_figure(1,2, figsize=(10,20))
vis.plot_segm(ax[0],final_segm, z_slice=4,background=raw, highlight_boundaries=False)
vis.plot_output_affin(ax[1], affs, nb_offset=-1, z_slice=4)
# vis.plot_output_affin(ax[2], duplicate_affs, nb_offset=-1, z_slice=0)
# vis.plot_segm(ax[3],active_nodes, z_slice=0,background=raw, highlight_boundaries=False, alpha_labels=0.7)
fig.savefig(os.path.join(get_abailoni_hci_home_path(), "../seeded_MWS_v2.png"))
