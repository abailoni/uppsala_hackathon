import matplotlib
matplotlib.use('Agg')

# FIXME: temporary hack to add all dependencies to PYTHON_PATH

from .utils.path_utils import get_abailoni_hci_home_path
import sys
import os

sys.path += [
os.path.join(get_abailoni_hci_home_path(), "python_libraries/nifty/python"),
os.path.join(get_abailoni_hci_home_path(), "python_libraries/cremi_python"),
os.path.join(get_abailoni_hci_home_path(), "python_libraries/affogato/python"),
os.path.join(get_abailoni_hci_home_path(), "pyCharm_projects/inferno"),
os.path.join(get_abailoni_hci_home_path(), "pyCharm_projects/constrained_mst"),
os.path.join(get_abailoni_hci_home_path(), "pyCharm_projects/neuro-skunkworks"),
os.path.join(get_abailoni_hci_home_path(), "pyCharm_projects/segmfriends"),
os.path.join(get_abailoni_hci_home_path(), "pyCharm_projects/speedrun"),
os.path.join(get_abailoni_hci_home_path(), "pyCharm_projects/embeddingutils"),
os.path.join(get_abailoni_hci_home_path(), "pyCharm_projects/firelight"),
# os.path.join(get_abailoni_hci_home_path(), "pyCharm_projects/hc_segmentation"),
os.path.join(get_abailoni_hci_home_path(), "pyCharm_projects/neurofire"),]
