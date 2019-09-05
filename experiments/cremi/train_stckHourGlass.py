import vaeAffs


from vaeAffs.utils.path_utils import change_paths_config_file

from speedrun import BaseExperiment, TensorboardMixin, InfernoMixin, FirelightLogger
from speedrun.log_anywhere import register_logger, log_image, log_scalar
from speedrun.py_utils import create_instance
from speedrun.py_utils import locate

from copy import deepcopy
import vaeAffs

import os
import torch
import torch.nn as nn

# from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore
from neurofire.criteria.loss_wrapper import LossWrapper
from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
from inferno.extensions.layers.convolutional import Conv3D
from inferno.trainers.callbacks import Callback
from inferno.io.transform.base import Compose

from embeddingutils.loss import WeightedLoss, SumLoss
from segmfriends.utils.config_utils import recursive_dict_update

from shutil import copyfile
import sys

from inferno.extensions.layers.convolutional import ConvELU3D, Conv3D, BNReLUConv3D

from neurofire.criteria.loss_wrapper import LossWrapper
from neurofire.criteria.loss_transforms import ApplyAndRemoveMask
from neurofire.criteria.loss_transforms import RemoveSegmentationFromTarget
from neurofire.criteria.loss_transforms import InvertTarget

from vaeAffs.datasets.cremi_stackedHourGlass import get_cremi_loader
from vaeAffs.utils.path_utils import get_source_dir



class BaseCremiExperiment(BaseExperiment, InfernoMixin, TensorboardMixin):
    def __init__(self, experiment_directory=None, config=None):
        super(BaseCremiExperiment, self).__init__(experiment_directory)
        # Privates
        self._device = None
        self._meta_config['exclude_attrs_from_save'] = ['data_loader', '_device']
        if config is not None:
            self.read_config_file(config)


        self.DEFAULT_DISPATCH = 'train'
        self.auto_setup()

        # register_logger(FirelightLogger, "image")
        register_logger(self, 'scalars')


        # offsets = self.get_boundary_offsets()
        # self.set('global/offsets', offsets)
        # self.set('loaders/general/volume_config/segmentation/affinity_config/offsets', offsets)

        self.model_class = list(self.get('model').keys())[0]

        if self.get("loaders/general/master_config/downscale_and_crop") is not None:
            # TODO: improve this...
            # FIXME: bug when replicate_targets is missing...
            ds_config = self.get("loaders/general/master_config/downscale_and_crop")
            nb_targets = len(ds_config) - 1
            self.set("trainer/num_targets", nb_targets)
        else:
            self.set("trainer/num_targets", 1)

        self.set_devices()


    def build_model(self, model_config=None):
        model_config = self.get('model') if model_config is None else model_config
        # return super(BaseCremiExperiment, self).build_model(model_config) #parse_model(model_config)

        model_path = model_config[next(iter(model_config.keys()))].pop('loadfrom', None)
        stacked_models_path = model_config[next(iter(model_config.keys()))].pop('load_stacked_models_from', None)
        model = create_instance(model_config, self.MODEL_LOCATIONS)

        if model_path is not None:
            print(f"loading model from {model_path}")
            state_dict = torch.load(model_path)["_model"].state_dict()
            model.load_state_dict(state_dict)
        if stacked_models_path is not None:
            for mdl in stacked_models_path:
                stck_mdl_path = stacked_models_path[mdl]
                print("loading stacked model {} from {}".format(mdl, stck_mdl_path))
                mdl_state_dict = torch.load(stck_mdl_path)["_model"].models[mdl].state_dict()
                model.models[mdl].load_state_dict(mdl_state_dict)


        return model

    def set_devices(self):
        # n_gpus = torch.cuda.device_count()
        # gpu_list = range(n_gpus)
        # self.set("gpu_list", gpu_list)
        # self.trainer.cuda(gpu_list)
        self.set("gpu_list", [0])
        self.trainer.cuda([0])

    def inferno_build_criterion(self):
        print("Building criterion")
        # path = self.get("autoencoder/path")
        loss_kwargs = self.get("trainer/criterion/kwargs", {})
        # from vaeAffs.models.losses import EncodingLoss, PatchLoss, PatchBasedLoss, StackedAffinityLoss
        loss_name = self.get("trainer/criterion/loss_name", "vaeAffs.models.losses.PatchBasedLoss")
        loss_config = {loss_name: loss_kwargs}
        loss_config[loss_name]['model'] = self.model
        model_kwargs = self.get('model/{}'.format(self.model_class))
        loss_config[loss_name]['model_kwargs'] = model_kwargs
        loss_config[loss_name]['devices'] = tuple(self.get("gpu_list"))

        loss = create_instance(loss_config, self.CRITERION_LOCATIONS)
        self._trainer.build_criterion(loss)
        self._trainer.build_validation_criterion(loss)

    def build_train_loader(self):
        kwargs = recursive_dict_update(self.get('loaders/train'), deepcopy(self.get('loaders/general')))
        return get_cremi_loader(kwargs)

    def build_val_loader(self):
        kwargs = recursive_dict_update(self.get('loaders/val'), deepcopy(self.get('loaders/general')))
        return get_cremi_loader(kwargs)


if __name__ == '__main__':
    print(sys.argv[1])

    source_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(source_path, 'configs')
    experiments_path = os.path.join(source_path, 'runs')

    sys.argv[1] = os.path.join(experiments_path, sys.argv[1])
    if '--inherit' in sys.argv:
        i = sys.argv.index('--inherit') + 1
        if sys.argv[i].endswith(('.yml', '.yaml')):
            sys.argv[i] = change_paths_config_file(os.path.join(config_path, sys.argv[i]))
        else:
            sys.argv[i] = os.path.join(experiments_path, sys.argv[i])
    if '--update' in sys.argv:
        i = sys.argv.index('--update') + 1
        sys.argv[i] = change_paths_config_file(os.path.join(config_path, sys.argv[i]))
    i = 0
    while True:
        if f'--update{i}' in sys.argv:
            ind = sys.argv.index(f'--update{i}') + 1
            sys.argv[ind] = os.path.join(config_path, sys.argv[ind])
            i += 1
        else:
            break
    cls = BaseCremiExperiment
    cls().run()

