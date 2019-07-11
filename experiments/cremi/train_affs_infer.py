import vaeAffs

from vaeAffs.utils.path_utils import change_paths_config_file, get_trendytukan_drive_path

from speedrun import BaseExperiment, TensorboardMixin, AffinityInferenceMixin
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

from vaeAffs.datasets.cremi_stackedHourGlass import get_cremi_loader
from vaeAffs.utils.path_utils import get_source_dir


class BaseCremiExperiment(BaseExperiment, AffinityInferenceMixin):
    def __init__(self, experiment_directory=None, config=None):
        super(BaseCremiExperiment, self).__init__(experiment_directory)
        # Privates
        self._device = None
        self._meta_config['exclude_attrs_from_save'] = ['data_loader', '_device']
        if config is not None:
            self.read_config_file(config)

        self.DEFAULT_DISPATCH = 'train'
        self.auto_setup()

        offsets = self.get_default_offsets()
        self.set('global/offsets', offsets)
        self.set('loaders/general/volume_config/segmentation/affinity_config/offsets', offsets)

        self.set_devices()

        self.build_infer_loader()

    def get_default_offsets(self):
        return [[0, -3, 0],
                [0, 0, -3],
                [0, -9, 0],
                [0, 0, -9],
                [0, -6, -7],
                [0, -6, +7],
                [0, -18, 0],
                [0, 0, -18],
                [-1, 0, 0],
                [-2, 0, 0],
                [+1, -1, -1],
                [+2, -1, -1],
                ]

    def build_model(self, model_config=None):
        model_config = self.get('model') if model_config is None else model_config
        model_class = list(model_config.keys())[0]
        # out_channels = self.get("autoencoder/latent_variable_size") * 2
        # model_config[model_class]['out_channels'] = out_channels
        # self.set('model/{}/out_channels'.format(model_class), out_channels)

        # self.build_final_activation(model_config)
        return super(BaseCremiExperiment, self).build_model(model_config)  # parse_model(model_config)

    def set_devices(self):
        n_gpus = torch.cuda.device_count()
        gpu_list = range(n_gpus)
        self.set("gpu_list", gpu_list)
        self.trainer.cuda(gpu_list)
        # self.trainer.cuda()

    def inferno_build_criterion(self):
        print("Building criterion")
        # path = self.get("autoencoder/path")
        loss_kwargs = self.get("trainer/criterion/kwargs")
        from vaeAffs.models.modified_unet import EncodingLoss, PatchLoss, AffLoss
        loss = AffLoss(**loss_kwargs)
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
        scaling_factors = self.get("stack_scaling_factors_2")
        kwargs = recursive_dict_update(self.get('loaders/train'), deepcopy(self.get('loaders/general')))
        kwargs["volume_config"]["scaling_factors"] = scaling_factors
        return get_cremi_loader(kwargs)

    def build_val_loader(self):
        scaling_factors = self.get("stack_scaling_factors_2")
        kwargs = recursive_dict_update(self.get('loaders/val'), deepcopy(self.get('loaders/general')))
        kwargs["volume_config"]["scaling_factors"] = scaling_factors
        return get_cremi_loader(kwargs)


    def build_infer_loader(self):
        # scaling_factors = self.get("stack_scaling_factors_2")
        kwargs = deepcopy(self.get('loaders/infer'))
        # kwargs["volume_config"]["scaling_factors"] = scaling_factors
        loader = get_cremi_loader(kwargs)
        self.set("full_volume_infer_shape", loader.dataset.volume.shape)
        self.set("inference/global_padding", loader.dataset.padding)
        self.set("inference/global_ds_ratio", loader.dataset.downsampling_ratio)

        return loader

    def save_infer_output(self, output):
        import h5py
        import numpy as np
        print("Saving....")
        with h5py.File(os.path.join(get_trendytukan_drive_path(), "projects/pixel_embeddings/stacked_hour_glass/predictions_sample_{}.h5".format(self.get("loaders/infer/name"))), 'w') as f:
            f.create_dataset('data', data=output.astype(np.float16), compression='gzip')


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
    cls().infer()
