import os
from copy import deepcopy
import numpy as np

# -----------------------
# Script options:
# -----------------------



type = "postproc"
CUDA = "CUDA_VISIBLE_DEVICES=0"

# WSDT superpixels plus GASP:

# list_of_args = [
#     (["--"], ["deb_infer"]),
#     (["--inherit"], [
#         "debug.yml",
#       ]),
#     # (["--config.experiment_name", "--config.offsets_file_name"],
#     #  ["mainFullTrain_cls", "bigUNet_cls", "main_classic", "clsDefct_cls", "noSideLoss_cls", "noGlia_cls", "main_dice", "2patches_cls"],
#     #  ["default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "dice_affs.json", "two_patches_only.json"],
#     #  ),
#     ([
#          "--config.experiment_name",
#          "--config.offsets_file_name",
#          "--config.postproc_config.invert_affinities"
#      ],
#      [
#          "v2_ignoreGlia_trainedAffs_thinBound",
#          # "v2_ignoreGlia_trainedAffs",
#          # "v2_main_trainedAffs_thinBound",
#          # "v2_main_trainedAffs",
#          # "v2_diceAffs_trainedAffs_thinBound",
#          # "v2_diceAffs_trainedAffs",
#      ],
#      [
#          "trainedAffs_from_patch.json",
#          # "trainedAffs_from_patch.json",
#          # "trainedAffs_from_patch.json",
#          # "trainedAffs_from_patch.json",
#          # "dice_affs.json",
#          # "dice_affs.json",
#      ],
#      [
#          "True",
#          # "True",
#          # "True",
#          # "True",
#          # "True",
#          # "True",
#      ],
#      ),
#
#     (["--config.postproc_config.save_name_postfix",
#       "--config.volume_config.ignore_glia"],
#      [
#          "fullGT",
#          # "ignoreGlia"
#      ],
#      [
#          "False",
#          # "True"
#      ]),
#     # (["--config.postproc_config.iterated_options.preset"], ["MEAN"]),
#     # (["--config.postproc_config.iterated_options.sample"], [
#     #     ["B", "C", "A"],
#     #     ["0", "1", "2"],
#     #     # "C"
#     # ]),
# ]

list_of_args = [
    # (["-m"], ["memory_profiler"]),
    (["--"], ["deb_infer"]),
    (["--inherit"], [
        "main_config.yml",
      ]),
    (["--update0"], ["validation_crop.yml",]),
    # (["--config.experiment_name", "--config.offsets_file_name"],
    #  ["mainFullTrain_cls", "bigUNet_cls", "main_classic", "clsDefct_cls", "noSideLoss_cls", "noGlia_cls", "main_dice", "2patches_cls"],
    #  ["default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "dice_affs.json", "two_patches_only.json"],
    #  ),
    ([
         "--config.experiment_name",
         "--update2",
        "--config.offsets_file_name",
        "--config.postproc_config.invert_affinities"
     ],
     [
         # "v4_addSparseAffs_eff",
         # "v4_onlySparseAffs_eff",
         "v4_addSparseAffs_avgDirectVar",
         # "v4_main_avgDirectVar",
         # "v4_main_eff",
     ],[
         # "empty_config.yml",
         # "empty_config.yml",
         "crop_avg_affs.yml",
         # "crop_avg_affs.yml",
         # "empty_config.yml",
     ], [
        # "dice_affs_v3.json",
        # "dice_affs_v3.json",
        "aggr_affs_v4.json",
        # "aggr_affs_v4.json",
        # "dice_affs_v3.json",
     ], [
        # "True",
        # "True",
        "False",
        # "False",
        # "True",
     ]
     ),

    (["--update1"], [
        # "empty_config.yml",
        "MWS_from_pix.yml",
        "GASP_from_pix_MWS.yml",
        "GASP_from_pix.yml",
        # "GASP_from_pix.yml",
        # "longRange_DWST.yml",
        # "multicut.yml",
        # "multicut_longR.yml",
    ]),
    (["--config.volume_config.affinities.inner_path"], ["data"]),
    (["--config.postproc_config.iterated_options.sub_crop_slice", "--config.postproc_config.save_name_postfix",
      # "--config.postproc_config.iterated_options.edge_prob"
      ],
     [
         # ":,70:75,25:-25,25:-25",
         # ":,70:80,25:-25,25:-25",
         # ":,70:85,25:-25,25:-25",
         # ":,70:90,25:-25,25:-25",
         # ":,70:95,25:-25,25:-25",
         # ":,70:100,25:-25,25:-25",
         # ":,70:105,25:-25,25:-25",
         # ":,70:110,25:-25,25:-25",
         # ":,70:115,25:-25,25:-25",
         ":,70:120,25:-25,25:-25",
         ":,70:125,25:-25,25:-25",
         ":,70:130,25:-25,25:-25",
         ":,70:135,25:-25,25:-25",
         ":,70:140,25:-25,25:-25",
         ":,70:145,25:-25,25:-25",
         ":,70:150,25:-25,25:-25",
         ":,70:155,25:-25,25:-25",
         ":,70:160,25:-25,25:-25",
     ],
     [
         # "_5",
         # "_10",
         # "_15",
         # "_20",
         # "_25",
         # "_30",
         # "_35",
         # "_40",
         # "_45",
         "_50",
         "_55",
         "_60",
         "_65",
         "_70",
         "_75",
         "_80",
         "_85",
         "_90",
         # "_affs_withLR_z",
         # "_affs_noLR",
         # "plusGliaMask2",
     ],
     # [
     #    "0.1",
     #    "0.0"
     # ]
     ),
    (["--config.volume_config.ignore_glia"], ["False"]),
    (["--update3"], ["memory_test_crop.yml",]),
    (["--config.postproc_config.iterated_options.sample"], [
        "C",
    ]),
    # (["> "], ["mem_test_results_{}.txt".format(np.random.randint(50000)),]),
]

"""
- MWS and DTWS+GASP (local)
- direct and averaged affs (not for dice..)
- partial sample C, 0, 1, 2 (all)
- ignore glia (all)
- offsets

"""


# -----------------------
# Compose list of commands to execute:
# -----------------------

cmd_base_string = CUDA
if type == "infer":
    cmd_base_string += " ipython experiments/cremi/infer_IoU.py"
elif type == "postproc":
    cmd_base_string += " ipython experiments/cremi/post_process.py"
else:
    raise ValueError


def recursively_get_cmd(current_cmd, accumulated_cmds):
    current_arg_spec_indx = len(current_cmd)
    if current_arg_spec_indx == len(list_of_args):

        # We are done, compose the command:
        new_cmd = cmd_base_string
        for i, arg_spec in enumerate(list_of_args):
            for nb_arg, arg_name in enumerate(arg_spec[0]):
                new_arg_str = current_cmd[i][nb_arg]
                if "///" in new_arg_str:
                    new_arg_str = new_arg_str.split("///")[0]

                new_cmd += " {} {}".format(arg_name, new_arg_str)
        accumulated_cmds.append(new_cmd)

    elif current_arg_spec_indx < len(list_of_args):
        # Here we add all options at current level and then recursively go deeper:
        current_arg_spec = list_of_args[current_arg_spec_indx]
        total_current_options = len(current_arg_spec[1])

        for nb_option in range(total_current_options):
            new_cmd_entry = []
            for arg in current_arg_spec[1:]:
                assert len(arg) == total_current_options, "All args  passed in the same entry should have the same number of options! {}, {}".format(arg, total_current_options)
                if isinstance(arg[nb_option], str):
                    # Here we simply append the string:
                    new_cmd_entry.append(arg[nb_option])
                else:
                    # Format the string from previously chosen options:
                    assert isinstance(arg[nb_option], tuple)
                    assert len(arg[nb_option]) >= 2

                    collected_format_args = []
                    for format_args in arg[nb_option][1:]:
                        indx1, indx2 = format_args.split(":")
                        assert int(indx1) < current_arg_spec_indx
                        collected_str = current_cmd[int(indx1)][int(indx2)]
                        if "///" in collected_str:
                            collected_str = collected_str.split("///")[1]
                        collected_format_args.append(collected_str)

                    # Compose new command entry:
                    new_cmd_entry.append(arg[nb_option][0].format(*collected_format_args))
            # Recursively go deeper:
            accumulated_cmds = recursively_get_cmd(current_cmd+[new_cmd_entry], accumulated_cmds)
    else:
        raise ValueError("Something went wrong")

    return accumulated_cmds

cmds_to_run = recursively_get_cmd([], [])
print("Number of commands to run: {}".format(len(cmds_to_run)))

for cmd in cmds_to_run:
    print("\n\n\n\n{}\n\n".format(cmd))
    os.system(cmd)
