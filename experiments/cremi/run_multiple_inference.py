import os
from copy import deepcopy

# -----------------------
# Script options:
# -----------------------

type = "infer"
CUDA = "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
# CUDA = "CUDA_VISIBLE_DEVICES=0"

# -----------
# Old?
# -----------


# list_of_args = [
#     (["--"], ["deb_infer"]),
#     (["--inherit"], [
#         "newCremi_v2_main.yml",
#       ]),
#     (["--update0", "--config.model.model_kwargs.path_backbone"], [
#         "newCremi_v2_trainAffsFromPatches.yml///main",
#         "newCremi_v2_ignoreGlia.yml///ignoreGlia",
#         # "newCremi_v2_diceAffs.yml///diceAffs",
#     ], [
#         "RUNS__HOME/v2_main_b/checkpoint.pytorch",
#         "RUNS__HOME/v2_ignoreGlia/checkpoint.pytorch",
#         # "placeholder",
#     ]),
#     (["--config.model.model_kwargs.loadfrom", "--update1"], [
#         ("RUNS__HOME/v2_{}_trainAffs/checkpoint.pytorch///", "2:0"),
#         ("RUNS__HOME/v2_{}_trainAffs_thinBound/checkpoint.pytorch///_thinBound", "2:0"),
#         # "RUNS__HOME/v2_main_trainAffs_b/checkpoint.pytorch",
#         # "RUNS__HOME/v2_main_trainAffs_b/checkpoint.pytorch",
#     ], [
#         "newCremi_v2_trainAffsFromPatches.yml",
#         "newCremi_v2_trainAffsFromPatches_thinBound.yml",
#     ]),
#     (["--config.name_experiment"], [ ("v2_{}_trainedAffs{}", "2:0", "3:0") ]),
#     (["--config.loaders.infer.loader_config.batch_size"], ["8"]),
#     # (["--config.loaders.infer.loader_config.batch_size"], ["1"]),
#     (["--config.loaders.infer.loader_config.num_workers"], ["20"]),
#     (["--config.loaders.infer.name"], [
#         "B",
#         "C",
#         "A",
#         "0",
#         "1",
#         "2",
#     ]),
# ]


# -----------
# Thin trained boundary
# -----------


list_of_args = [
    (["--"], ["deb_infer"]),
    (["--inherit"], [
        "newCremi_v2_main.yml",
      ]),
    (["--update0",
      "--config.model.model_kwargs.path_backbone"
      ], [
        "newCremi_v2_trainAffsFromPatches.yml///main",
        # "newCremi_v2_ignoreGlia.yml///ignoreGlia",
        # "newCremi_v2_diceAffs.yml///diceAffs",
    ],
     [
        "RUNS__HOME/v2_main_b/checkpoint.pytorch",
    #     # "RUNS__HOME/v2_ignoreGlia/checkpoint.pytorch",
    #     "RUNS__HOME/v2_diceAffs_b/checkpoint.pytorch",
    ]
     ),
    (["--config.model.model_kwargs.loadfrom", "--update1"], [
        ("RUNS__HOME/v2_{}_trainAffs/checkpoint.pytorch///", "2:0"),
        # ("RUNS__HOME/v2_{}_trainAffs_thinBound/checkpoint.pytorch///_thinBound", "2:0"),
        # "RUNS__HOME/v2_diceAffs_b/checkpoint.pytorch///",
        # "RUNS__HOME/v2_diceAffs_thinBound/checkpoint.pytorch///_thinBound",
    ], [
        "newCremi_v2_trainAffsFromPatches.yml",
        # "newCremi_v2_trainAffsFromPatches_thinBound.yml",
        # "empty_config.yml",
        # "newCremi_v2_finetuneDiceLoss.yml",
    ]),
    (["--config.name_experiment"], [ ("v2_{}_trainedAffs{}", "2:0", "3:0") ]),
    (["--config.loaders.infer.loader_config.batch_size"], ["8"]),
    # (["--config.loaders.infer.loader_config.batch_size"], ["1"]),
    (["--config.loaders.infer.loader_config.num_workers"], ["20"]),
    (["--config.loaders.infer.name"], [
        "B",
        # "C",
        # "A",
        # "0",
        # "1",
        # "2",
    ]),
]

# -----------
# Infer glia:
# -----------

#
# list_of_args = [
#     (["--"], ["deb_infer"]),
#     (["--inherit"], [
#         "newCremi_v2_main.yml",
#       ]),
#     (["--update0",
#       ], [
#         "empty_config.yml///main",
#         # "newCremi_v2_ignoreGlia.yml///ignoreGlia",
#         "newCremi_v2_diceAffs.yml///diceAffs",
#     ],
#      ),
#     (["--update1"], [
#         "newCremi_v2_inferGlia.yml"
#     ]),
#     (["--config.name_experiment"], [ ("v2_{}_trainedAffs", "2:0") ]),
#     (["--config.loaders.infer.loader_config.batch_size"], ["8"]),
#     # (["--config.loaders.infer.loader_config.batch_size"], ["1"]),
#     (["--config.inner_path_output"], ["glia"]),
#     (["--config.loaders.infer.loader_config.num_workers"], ["20"]),
#     (["--config.loaders.infer.name"], [
#         "B",
#         "C",
#         "A",
#         "0",
#         "1",
#         "2",
#     ]),
# ]


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
                    actual_string = arg[nb_option][0]
                    # if "///" in actual_string:
                    #     actual_string = actual_string.split("///")[0]
                    new_cmd_entry.append(actual_string.format(*collected_format_args))
            # Recursively go deeper:
            accumulated_cmds = recursively_get_cmd(current_cmd+[new_cmd_entry], accumulated_cmds)
    else:
        raise ValueError("Something went wrong")

    return accumulated_cmds

cmds_to_run = recursively_get_cmd([], [])

for cmd in cmds_to_run:
    print("\n\n\n\n{}\n\n".format(cmd))
    os.system(cmd)