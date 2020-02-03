import os
from copy import deepcopy

# -----------------------
# Script options:
# -----------------------

# type = "infer"
# CUDA = "CUDA_VISIBLE_DEVICES=0,1,2,5"
#
# list_of_args = [
#     (["--"], ["deb_infer"]),
#     (["--config.model_shortcuts.affinity_mode"],
#      [
#         # "classic///cls",
#         "probabilistic///probs07"]),
#     (["--inherit"], [
#         # "newCremi_comparison_main_diceAffs.yml///main_dice",
#         "newCremi_comparison_main.yml///main",
#         "newCremi_comparison_classicDefcAugm.yml///clsDefct",
#         "newCremi_comparison_main_noGlia.yml///noGlia",
#         "newCremi_comparison_main_2patches.yml///2patches",
#         "newCremi_comparison_wo_sideLoss.yml///noSideLoss",
#         "newCremi_comparison_BIG.yml///biUNet",
#         "newCremi_comparison_fullMain.yml///mainFullTrain",
#         "newCremi_comparison_noExtraPad.yml///noExtaPad",
#       ]),
#     (["--config.name_experiment"], [ ("{}_{}", "2:0", "1:0") ]),
#     (["--config.model_shortcuts.patch_threshold"], ["0.7"]),
#     (["--config.loaders.infer.loader_config.batch_size"], ["4"]),
#     (["--config.loaders.infer.loader_config.num_workers"], ["20"]),
#     (["--config.loaders.infer.name"], ["B", "C", "0", "1", "2"]),
# ]





type = "postproc"
CUDA = "CUDA_VISIBLE_DEVICES=0"

list_of_args = [
    (["--"], ["deb_infer"]),
    (["--inherit"], [
        "debug.yml",
      ]),
    # (["--config.experiment_name", "--config.offsets_file_name"],
    #  ["mainFullTrain_cls", "bigUNet_cls", "main_classic", "clsDefct_cls", "noSideLoss_cls", "noGlia_cls", "main_dice", "2patches_cls"],
    #  ["default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "default_from_patch.json", "dice_affs.json", "two_patches_only.json"],
    #  ),
    (["--config.experiment_name", "--config.offsets_file_name",
      "--config.postproc_config.invert_affinities"],
     ["main_dice",
      "2patches_cls"],
     ["dice_affs.json", "two_patches_only.json"],
     ["True", "False"],
     ),

    # (["--config.postproc_config.iterated_options.preset"], ["MEAN"]),
    # (["--config.postproc_config.iterated_options.sample"], ["B", "C"]),
]


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

for cmd in cmds_to_run:
    print("\n\n\n\n{}\n\n".format(cmd))
    os.system(cmd)
