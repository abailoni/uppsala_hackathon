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




from matplotlib import pyplot as plt
import yaml
import matplotlib
matplotlib.rcParams.update({'font.size': 25})

sample = "C"
# slices = {'A': ":,:,:,:", 'B': ":, :, 90:, 580: 1900", 'C': ":, :, 70:1450, 95:1425"}
slices = {'A': ":,:,:,:", 'B': ":,:,:,:", 'C': ":,70:-6,50:-50,50:-50",
         "0": ":,:,:,:",
         "1": ":,:,:,:",
         "2": ":,:,:,:",
         "3": ":,:,:,:",}

parsed_slice = parse_data_slice(slices[sample])
conf_folder_path = os.path.join(get_trendytukan_drive_path(), "projects/pixel_embeddings/v4_addSparseAffs_avgDirectVar/scores")


prefix_math = [
    "C__MEAN___",
    # "C__MutexWatershed___",
    "C__MWS__stride10___"
]


labels = {
    "C__MEAN___": "Gasp Average",
    "C__MWS__stride10___": "Mutex Watershed"

}

keys = [
    "nb_nodes",
    "nb_edges",
    "starting_mem",
    "final_mem",
    "agglo_mem",
    "agglo_runtime",
    "graph_mem",
    "graph_runtime",
]

axis_labels = [
    "Number of nodes",
    "Number of edges",
    "Init memory (GB)",
    "Total memory script (GB)",
    "Memory allocated during agglomeration (GB)",
    "Runtime agglomeration (min)",
    "Memory allocated during graph creation (GB)",
    "Runtime graph creation (min)",
]

scaling_factors = [
    1,
    1,
    1024,
    1024,
    1024,
    60,
    1024,
    60,
]


collected_data = {key: [] for key in prefix_math}
for prefix in prefix_math:
    collected_configs = {}
    for item in os.listdir(conf_folder_path):
        if os.path.isfile(os.path.join(conf_folder_path, item)):
            filename = item
            if not filename.endswith(".yml") or filename.startswith("."):
                continue
            if filename.startswith(prefix):
                nb_slices = filename.replace(prefix, "").replace(".yml", "")
                try:
                    nb_slices = int(nb_slices)
                except ValueError:
                    continue
                with open(os.path.join(conf_folder_path, item), 'rb') as f:
                    config = yaml.load(f)
                new_point = []
                for key in keys:
                    value = config[key] if key in config else 0.
                    new_point.append(value)
                collected_data[prefix].append(new_point)


# print(collected_data)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
x_axis_quantity = 1
for quantity in [2,3,4,5,6,7]:
    f, ax = plt.subplots(ncols=1, nrows=1,
                         figsize=(13, 13))
    for idx_pref, prefix in enumerate(prefix_math):
        collected_data[prefix] = np.array(collected_data[prefix])
    # ax.plot(collected_data[prefix][:,0], collected_data[prefix][:,4])
        argsort = np.argsort(collected_data[prefix][:, x_axis_quantity])
        x_data = (collected_data[prefix][:, x_axis_quantity] / scaling_factors[x_axis_quantity])[argsort]
        y_data = (collected_data[prefix][:, quantity] / scaling_factors[quantity])[argsort]
        # if (quantity == 5) and prefix == 'C__MEAN___':
        #     p = np.poly1d(np.polyfit(x_data, y_data, 2))
        #     fake_xdata = np.linspace(0, 1e9, 20)
        #     ax.plot(fake_xdata, p(fake_xdata), color='black')
        # if (quantity == 4 or quantity == 6) and prefix == 'C__MEAN___':
        #     p = np.poly1d(np.polyfit(x_data, y_data, 1))
        #     fake_xdata = np.linspace(0, 1e9, 20)
        #     ax.plot(fake_xdata, p(fake_xdata), color='black')
        ax.plot(x_data, y_data, label=labels[prefix], color=colors[idx_pref])
        ax.scatter(x_data, y_data, color=colors[idx_pref])
        ax.set_xlabel(axis_labels[x_axis_quantity])
        ax.set_ylabel(axis_labels[quantity])
        ax.legend(loc='best')

    f.savefig("debug_{}.pdf".format(keys[quantity]), format='pdf')

# for prefix in prefix_math:
#     collected_data[prefix] = np.array(collected_data[prefix])
#     f, ax = plt.subplots(ncols=1, nrows=1,
#                              figsize=(13,13))
#     # ax.plot(collected_data[prefix][:,0], collected_data[prefix][:,4])
#     ax.scatter(collected_data[prefix][:, 0], collected_data[prefix][:, 5])
#     f.savefig("test_runtime_{}.pdf".format(prefix), format='pdf')
