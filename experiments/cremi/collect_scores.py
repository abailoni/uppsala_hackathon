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
    "v4_addSparseAffs_eff",
    "v4_onlySparseAffs_eff",
    "v4_main_avgDirectVar",
    "v4_addSparseAffs_avgDirectVar",
    # "v3_diceAffs_noTrainGlia_direct",
    # "v3_main_noTrainGlia_avgDirectVarCropped",
    # "v3_diceAffs_direct",
    # "v3_noMultiScale_small_avgDirectVar",
]

REQUIRED_STRINGS = [
    # "C__"
]

EXCLUDE_STRINGS = [
    "multicut_kerLin",
]

INCLUDE_STRINGS = [
]


LATEX_OUTPUT = False

sorting_column_idx = 1

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

max_nb_columns = 0

for exp_name in EXP_NAMES:
    os.path.join(project_dir, exp_name)
    scores_path = os.path.join(project_dir, exp_name, "scores")

    # Get all the configs:
    for item in os.listdir(scores_path):
        if os.path.isfile(os.path.join(scores_path, item)):
            filename = item
            if not filename.endswith(".yml") or filename.startswith("."):
                continue
            skip = False
            for char in REQUIRED_STRINGS:
                if char not in filename:
                    skip = True
                    break
            if not skip:
                for excl_string in EXCLUDE_STRINGS:
                    if excl_string in filename:
                        skip = True
                        break
                for excl_string in INCLUDE_STRINGS:
                    if excl_string in filename:
                        skip = False
                        break
            if skip:
                continue
            result_file = os.path.join(scores_path, filename)
            config = yaml2dict(result_file)

            # print(filename.replace(".yml", "").split("__"))
            # new_table_entrance = [exp_name + "__" + filename.replace(".yml", "")]
            new_table_entrance = [exp_name] + \
                                 ["{}".format(spl) for spl in filename.replace(".yml", "").split("__")]
            nb_first_columns = len(new_table_entrance)
            if nb_first_columns > max_nb_columns:
                # Add empty columns to all previous rows:
                cols_to_add = nb_first_columns - max_nb_columns
                for i, row in enumerate(collected_results):
                    collected_results[i] = row[:max_nb_columns] + ["" for _ in range(cols_to_add)] + \
                                           row[max_nb_columns:]
                max_nb_columns = nb_first_columns
            elif nb_first_columns < max_nb_columns:
                # Add empty columns only to this row:
                cols_to_add = max_nb_columns - nb_first_columns
                new_table_entrance += ["" for _ in range(cols_to_add)]

            for j, key in enumerate(keys_to_collect):
                # print(result_file)
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

if len(collected_results) == 0:
    raise ValueError("No scores collected")
assert all(len(row) == len(collected_results[0]) for row in collected_results)
# if any(len(row) != len(collected_results[0]) for row in collected_results):
#     # Collapse first columns?
#     for i, row in enumerate(collected_results):
#         collected_results[i] = ['__'.join(row[:nb_first_columns])] + row[nb_first_columns:]
#     nb_first_columns = 1

collected_results = np.array(collected_results, dtype="str")
collected_results = collected_results[collected_results[:, sorting_column_idx + max_nb_columns].argsort()]
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
