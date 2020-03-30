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
from segmfriends.utils.various import parse_data_slice, readHDF5, writeHDF5, readHDF5_from_volume_config
import os
import json
import h5py

from scipy.ndimage import zoom
import nifty.graph.rag as nrag




import segmfriends.vis as vis
from matplotlib import pyplot as plt

sample = "C"
# slices = {'A': ":,:,:,:", 'B': ":, :, 90:, 580: 1900", 'C': ":, :, 70:1450, 95:1425"}
slices = {'A': ":,:,:,:", 'B': ":,:,:,:", 'C': ":,70:-6,50:-50,50:-50",
         "0": ":,:,:,:",
         "1": ":,:,:,:",
         "2": ":,:,:,:",
         "3": ":,:,:,:",}

parsed_slice = parse_data_slice(slices[sample])
data_path = os.path.join(get_abailoni_hci_home_path(), "datasets/new_cremi/sample{}.h5".format(sample))
# data_path = os.path.join(get_trendytukan_drive_path(), "datasets/new_cremi/fib25/sample{}.h5".format(sample))
with h5py.File(data_path, 'r') as f:
    print([atr for atr in f['volumes/labels']])
#     glia = f['volumes/labels/glia'][:]
    raw = f['volumes/raw'][parsed_slice[1:]]
    GT = f['volumes/labels/neuron_ids_fixed'][parsed_slice[1:]]

# Load affs:
from segmfriends.utils.various import writeHDF5, readHDF5
affs_path = os.path.join(get_trendytukan_drive_path(), "projects/pixel_embeddings/{}/predictions_sample_{}.h5".format("v4_addSparseAffs_eff", sample))
affs_dice = 1. - readHDF5(affs_path, "data", crop_slice="3:5,70:-6,25:-25,25:-25")
affs_path = os.path.join(get_trendytukan_drive_path(), "projects/pixel_embeddings/{}/predictions_sample_{}.h5".format("v4_addSparseAffs_avgDirectVar", sample))
affs_avg = readHDF5(affs_path, "data", crop_slice="3:5,70:-6,25:-25,25:-25")


# Load segm:
folder_path = os.path.join(get_trendytukan_drive_path(), "projects/pixel_embeddings/")
segm_path = os.path.join(folder_path,"{}/out_segms/{}.h5".format("v4_addSparseAffs_avgDirectVar","{}__MutexWatershed___affs_withLR_z".format(sample)))
segm = readHDF5(segm_path, "segm_WS")


# # 127, 358, 34
#
# subcrop = parse_data_slice(":,34:35,300:450,100:250")
# subcrop_no_dws = parse_data_slice(":,34:35,600:900:2,200:500:2")
subcrop = parse_data_slice(":,34:35,300:400,100:250")
subcrop_no_dws = parse_data_slice(":,34:35,600:800:2,200:500:2")

#
#
#
# # # All together:
# # fig, axes = vis.get_figure(2,3, figsize=(18,27))
# # vis.plot_gray_image(axes[0,0], raw[subcrop_no_dws[1:]])
# # vis.plot_segm(axes[0,1], GT[subcrop_no_dws[1:]], background=raw[subcrop_no_dws[1:]], alpha_labels=0.6,
# #               alpha_boundary=0.7)
# # vis.plot_output_affin(axes[1,0], affs_dice[subcrop].astype('float32'), nb_offset=0)
# # vis.plot_output_affin(axes[2,0], affs_dice[subcrop].astype('float32'), nb_offset=1)
# # vis.plot_output_affin(axes[1,1], affs_avg[subcrop].astype('float32'), nb_offset=0)
# # vis.plot_output_affin(axes[2,1], affs_avg[subcrop].astype('float32'), nb_offset=1)
# # fig.savefig(os.path.join(get_trendytukan_drive_path(), "test_plot.pdf"), format='pdf')
#
# Single images:
fig, axes = vis.get_figure(1,1, figsize=(12,12))
vis.plot_gray_image(axes, raw[subcrop_no_dws[1:]])
plt.tight_layout()
fig.savefig(os.path.join(get_trendytukan_drive_path(), "raw.pdf"), format='pdf')

fig, axes = vis.get_figure(1,1, figsize=(12,12))
vis.plot_segm(axes, GT[subcrop_no_dws[1:]], background=raw[subcrop_no_dws[1:]], alpha_labels=0.6,
              alpha_boundary=0.7)
plt.tight_layout()
fig.savefig(os.path.join(get_trendytukan_drive_path(), "GT.pdf"), format='pdf')

fig, axes = vis.get_figure(1,1, figsize=(12,12))
vis.plot_output_affin(axes, affs_dice[subcrop].astype('float32'), nb_offset=0)
plt.tight_layout()
fig.savefig(os.path.join(get_trendytukan_drive_path(), "affs1.pdf"), format='pdf')

fig, axes = vis.get_figure(1,1, figsize=(12,12))
vis.plot_output_affin(axes, affs_dice[subcrop].astype('float32'), nb_offset=1)
plt.tight_layout()
fig.savefig(os.path.join(get_trendytukan_drive_path(), "affs2.pdf"), format='pdf')

fig, axes = vis.get_figure(1,1, figsize=(12,12))
vis.plot_output_affin(axes, affs_avg[subcrop].astype('float32'), nb_offset=0)
plt.tight_layout()
fig.savefig(os.path.join(get_trendytukan_drive_path(), "affs3.pdf"), format='pdf')

fig, axes = vis.get_figure(1,1, figsize=(12,12))
vis.plot_output_affin(axes, affs_avg[subcrop].astype('float32'), nb_offset=1)
plt.tight_layout()
fig.savefig(os.path.join(get_trendytukan_drive_path(), "affs4.pdf"), format='pdf')



# # Full segmentation:
#
# subcrop = parse_data_slice(":,30:31,130:560,150:630")
# subcrop_no_dws = parse_data_slice(":,30:31,260:1120:2,300:1260:2")
#
# fig, axes = vis.get_figure(1,1, figsize=(20,20))
# vis.plot_segm(axes, segm[subcrop[1:]], background=raw[subcrop_no_dws[1:]], alpha_labels=0.6,
#               alpha_boundary=0.7)
# fig.savefig(os.path.join(get_trendytukan_drive_path(), "MWS_segm.pdf"), format='pdf')
