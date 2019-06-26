import vaeAffs

from speedrun import BaseExperiment, TensorboardMixin, InfernoMixin
from speedrun.log_anywhere import register_logger, log_image, log_scalar
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

from vaeAffs.datasets.cremi_Unet import get_cremi_loader
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

        register_logger(self, 'scalars')
        register_logger(self, 'embedding')
        register_logger(self, 'image')

        offsets = self.get_default_offsets()
        self.set('global/offsets', offsets)
        self.set('loaders/general/volume_config/segmentation/affinity_config/offsets', offsets)

    def get_default_offsets(self):
        return [[0, -4, +4],
                [0, -4, -4], [0, -4, 0], [0, 0, -4]]

    def build_model(self, model_config=None):
        model_config = self.get('model') if model_config is None else model_config
        model_class = list(model_config.keys())[0]
        out_channels = self.get("autoencoder/latent_variable_size") * 2
        model_config[model_class]['out_channels'] = out_channels
        self.set('model/{}/out_channels'.format(model_class), out_channels)

        # self.build_final_activation(model_config)
        return super(BaseCremiExperiment, self).build_model(model_config) #parse_model(model_config)

    # def build_final_activation(self, model_config=None):
    #     model_config = self.get('model') if model_config is None else model_config
    #     model_class = list(model_config.keys())[0]
    #
    #
    #     final_activation = model_config[model_class].pop('final_activation', None)
    #     if final_activation is None:
    #         return
    #     if isinstance(final_activation, str):
    #         final_activation = locate(
    #                 final_activation, ['torch.nn'])
    #         model_config[model_class]['final_activation'] = \
    #             final_activation()
    #         return
    #     model_config[model_class]['final_activation'] = \
    #         final_activation


    def inferno_build_criterion(self):
        print("Building criterion")
        path = self.get("autoencoder/path")
        from vaeAffs.models.modified_unet import EncodingLoss
        loss = EncodingLoss(path_autoencoder_model=path)
        self._trainer.build_criterion(loss)
        self._trainer.build_validation_criterion(loss)
    #
    # def inferno_build_metric(self):
    #     metric_config = self.get('trainer/metric')
    #     frequency = metric_config.pop('evaluate_every', (25, 'iterations'))
    #
    #     self.trainer.evaluate_metric_every(frequency)
    #     if metric_config:
    #         assert len(metric_config) == 1
    #         for class_name, kwargs in metric_config.items():
    #             cls = locate(class_name)
    #             #kwargs['offsets'] = self.get('global/offsets')
    #             #kwargs['z_direction'] = self.get(
    #             #    'loaders/general/master_config/compute_directions/z_direction')
    #             print(f'Building metric of class "{cls.__name__}"')
    #             metric = cls(**kwargs)
    #             self.trainer.build_metric(metric)
    #     self.set('trainer/metric/evaluate_every', frequency)

    def build_train_loader(self):
        return get_cremi_loader(recursive_dict_update(self.get('loaders/train'), deepcopy(self.get('loaders/general'))))

    def build_val_loader(self):
        return get_cremi_loader(recursive_dict_update(self.get('loaders/val'), deepcopy(self.get('loaders/general'))))


if __name__ == '__main__':
    print(sys.argv[1])

    source_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(source_path, 'configs')
    experiments_path = os.path.join(source_path, 'runs')

    sys.argv[1] = os.path.join(experiments_path, sys.argv[1])
    if '--inherit' in sys.argv:
        i = sys.argv.index('--inherit') + 1
        if sys.argv[i].endswith(('.yml', '.yaml')):
            sys.argv[i] = os.path.join(config_path, sys.argv[i])
        else:
            sys.argv[i] = os.path.join(experiments_path, sys.argv[i])
    if '--update' in sys.argv:
        i = sys.argv.index('--update') + 1
        sys.argv[i] = os.path.join(config_path, sys.argv[i])
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

