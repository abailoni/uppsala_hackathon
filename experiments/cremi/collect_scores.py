import os
from copy import deepcopy
from vaeAffs.utils.path_utils import get_trendytukan_drive_path, get_abailoni_hci_home_path


from segmfriends.utils.config_utils import assign_color_to_table_value, return_recursive_key_in_dict
import json
import numpy as np
from segmfriends.utils.various import yaml2dict
# -----------------------
# Script options:
# -----------------------

project_dir = os.path.join(get_trendytukan_drive_path(),"projects/pixel_embeddings")

EXP_NAMES = ["ignoreGlia_trainedAffs", "v2_ignoreGlia_trainedAffs_thinBound", ]

LATEX_OUTPUT = False

sorting_column_idx = 0

# -------------------------------------------------------






keys_to_collect = [
    ['score_WS', 'cremi-score'],
    ['score_WS', 'adapted-rand'],
    ['score_WS', 'vi-merge'],
    ['score_WS', 'vi-split'],
    # ['runtime'],
    # ['energy']
]

nb_flt_digits = [
    3,
    3,
    3,
    3,
    2,
    # 2,
]
nb_formats = [
    'f',
    'f',
    'f',
    'f',
    'e',
    # 'e',
]

# label_names = {
#     'MutexWatershed': "Abs Max",
#     'mean': "Average",
#     "max": "Max",
#     "min": "Min",
#     "sum": "Sum",
# }

collected_results = []
# energies, ARAND = [], []
# SEL_PROB = 0.1


for exp_name in EXP_NAMES:
    os.path.join(project_dir, exp_name)
    scores_path = os.path.join(project_dir, exp_name, "scores")

    # Get all the configs:
    for item in os.listdir(scores_path):
        if os.path.isfile(os.path.join(scores_path, item)):
            filename = item
            if not filename.endswith(".yml") or filename.startswith("."):
                continue
            result_file = os.path.join(scores_path, filename)
            config = yaml2dict(result_file)

            new_table_entrance = ["{}_{}".format(exp_name, filename.replace(".yml", ""))]

            for j, key in enumerate(keys_to_collect):
                cell_value = return_recursive_key_in_dict(config, key)
                # if key[-1] == 'adapted-rand':
                #     new_table_entrance.append("{0:.{prec}{type}}".format(1. - cell_value, prec=nb_flt_digits[j],
                #                                                          type=nb_formats[j]))
                # else:
                new_table_entrance.append("{0:.{prec}{type}}".format(cell_value, prec=nb_flt_digits[j],
                                                                     type=nb_formats[j]))

            collected_results.append(new_table_entrance)

collected_results = np.array(collected_results)
collected_results = collected_results[collected_results[:, sorting_column_idx + 1].argsort()]
ID = np.random.randint(255000)
print(ID)
from segmfriends.utils.various import check_dir_and_create
export_dir = os.path.join(project_dir, "collected_scores")
check_dir_and_create(export_dir)
if LATEX_OUTPUT:
    np.savetxt(os.path.join(export_dir, "collected_cremi_{}.csv".format(ID)), collected_results, delimiter=' & ',
           fmt='%s',
           newline=' \\\\\n')
else:
    np.savetxt(os.path.join(export_dir, "collected_cremi_{}.csv".format(ID)), collected_results, delimiter=';',
               fmt='%s',
               newline=' \n')
