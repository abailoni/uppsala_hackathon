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

EXP_NAMES = [
    "v2_main_trainedAffs_thinBound",
    "v2_diceAffs_trainedAffs_thinBound",
    "v2_ignoreGlia_trainedAffs_thinBound",
    "v2_ignoreGlia_trainedAffs",
    "v2_diceAffs_trainedAffs",
    "v2_main_trainedAffs"
]

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


# for exp_name in EXP_NAMES:
#     os.path.join(project_dir, exp_name)
#     scores_path = os.path.join(project_dir, exp_name, "out_segms")
#     import shutil
#
#     # Get all the configs:
#     for item in os.listdir(scores_path):
#         if os.path.isfile(os.path.join(scores_path, item)):
#             filename = item
#             if not filename.endswith(".h5") or filename.startswith("."):
#                 continue
#             result_file = os.path.join(scores_path, filename)
#             new_filename = result_file.replace("_fullGT.", "__fullGT.")
#             new_filename = new_filename.replace("_ignoreGlia.", "__ignoreGlia.")
#             shutil.move(result_file, new_filename)


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

            # print(filename.replace(".yml", "").split("__"))
            # new_table_entrance = [exp_name + "__" + filename.replace(".yml", "")]
            new_table_entrance = [exp_name] + \
                                 ["{}".format(spl) for spl in filename.replace(".yml", "").split("__")]
            nb_first_columns = len(new_table_entrance)


            for j, key in enumerate(keys_to_collect):
                cell_value = return_recursive_key_in_dict(config, key)
                # if key[-1] == 'adapted-rand':
                #     new_table_entrance.append("{0:.{prec}{type}}".format(1. - cell_value, prec=nb_flt_digits[j],
                #                                                          type=nb_formats[j]))
                # else:
                new_table_entrance.append("{0:.{prec}{type}}".format(cell_value, prec=nb_flt_digits[j],
                                                                     type=nb_formats[j]))

            collected_results.append(new_table_entrance)

# nb_col, nb_rows = len(collected_results[0]), len(collected_results)
#
# collected_array = np.empty((nb_rows, nb_col), dtype="str")
# for r in range(nb_rows):
#     for c in range(nb_col):
#         collected_array[r, c] = collected_results[r][c]

# collected_results = np.array([np.array(item, dtype="str") for item in collected_results], dtype="str")
collected_results = np.array(collected_results, dtype="str")
collected_results = collected_results[collected_results[:, sorting_column_idx + nb_first_columns].argsort()]
ID = np.random.randint(255000)
print(ID)
from segmfriends.utils.various import check_dir_and_create
export_dir = os.path.join(project_dir, "collected_scores")
check_dir_and_create(export_dir)
# print(collected_results)
if LATEX_OUTPUT:
    np.savetxt(os.path.join(export_dir, "collected_cremi_{}.csv".format(ID)), collected_results, delimiter=' & ',
           fmt='%s',
           newline=' \\\\\n')
else:
    np.savetxt(os.path.join(export_dir, "collected_cremi_{}.csv".format(ID)), collected_results, delimiter=';',
               fmt='%s',
               newline=' \n')
