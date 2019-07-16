from copy import deepcopy

from inferno.io.core import ZipReject, Concatenate
from inferno.io.transform import Compose
from inferno.io.transform.generic import AsTorchBatch
from inferno.io.transform.volume import RandomFlip3D, VolumeAsymmetricCrop
from inferno.io.transform.image import RandomRotate, ElasticTransform
from inferno.utils.io_utils import yaml2dict

from torch.utils.data.dataloader import DataLoader, default_collate

from neurofire.datasets.loader import RawVolume, SegmentationVolume, RawVolumeWithDefectAugmentation
from neurofire.transform.affinities import Segmentation2AffinitiesDynamicOffsets, affinity_config_to_transform
from neurofire.transform.artifact_source import RejectNonZeroThreshold
from neurofire.transform.volume import RandomSlide

from ..transforms import ComputeVAETarget, RemoveThirdDimension, RemoveInvalidAffs, HackyHacky, DownsampleAndCrop3D, \
    ReplicateBatch
import numpy as np

from neurofire.criteria.loss_transforms import InvertTarget


class RejectSingleLabelVolumes(object):
    def __init__(self, threshold, threshold_zero_label=1.):
        """
        :param threshold: If the biggest segment takes more than 'threshold', batch is rejected
        : param threshold_zero_label: if the percentage of non-zero-labels is less than this, reject
        """
        self.threshold = threshold
        self.threshold_zero_label = threshold_zero_label

    def __call__(self, fetched):
        _, counts = np.unique(fetched, return_counts=True)
        # Check if we should reject:
        return ((float(np.max(counts)) / fetched.size) > self.threshold) or (
                    (np.count_nonzero(fetched) / fetched.size) < self.threshold_zero_label)


class CremiDataset(ZipReject):
    def __init__(self, name, volume_config, slicing_config,
                 defect_augmentation_config, master_config=None):
        assert isinstance(volume_config, dict)
        assert isinstance(slicing_config, dict)
        assert isinstance(defect_augmentation_config, dict)
        assert 'raw' in volume_config
        assert 'segmentation' in volume_config

        volume_config = deepcopy(volume_config)
        self.scaling_factors = volume_config.pop("scaling_factors")

        # Get kwargs for raw volume
        raw_volume_kwargs = dict(volume_config.get('raw'))

        # check if we have special dict entries for names in the defect augmentation
        # slicing config
        augmentation_config = deepcopy(defect_augmentation_config)
        for slicing_key, slicing_item in augmentation_config['artifact_source']['slicing_config'].items():
            if isinstance(slicing_item, dict):
                new_item = augmentation_config['artifact_source']['slicing_config'][slicing_key][name]
                augmentation_config['artifact_source']['slicing_config'][slicing_key] = new_item

        raw_volume_kwargs.update({'defect_augmentation_config': augmentation_config})
        raw_volume_kwargs.update(slicing_config)
        # Build raw volume
        self.raw_volume = RawVolumeWithDefectAugmentation(name=name, **raw_volume_kwargs)

        # Get kwargs for segmentation volume
        segmentation_volume_kwargs = dict(volume_config.get('segmentation'))
        segmentation_volume_kwargs.update(slicing_config)
        self.affinity_config = segmentation_volume_kwargs.pop('affinity_config', None)
        # Build segmentation volume
        self.segmentation_volume = SegmentationVolume(name=name,
                                                      **segmentation_volume_kwargs)

        rejection_threshold = volume_config.get('rejection_threshold', 0.92)
        super().__init__(self.raw_volume, self.segmentation_volume,
                         sync=True, rejection_dataset_indices=1,
                         rejection_criterion=RejectSingleLabelVolumes(1.0, 0.95))
        # Set master config (for transforms)
        self.master_config = {} if master_config is None else master_config
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose()

        if self.master_config.get('random_flip', False):
            transforms.add(RandomFlip3D())
            transforms.add(RandomRotate())

        # # Elastic transforms can be skipped by
        # # setting elastic_transform to false in the
        # # yaml config file.
        # if self.master_config.get('elastic_transform'):
        #     elastic_transform_config = self.master_config.get('elastic_transform')
        #     if elastic_transform_config.get('apply', False):
        #         transforms.add(ElasticTransform(alpha=elastic_transform_config.get('alpha', 2000.),
        #                                         sigma=elastic_transform_config.get('sigma', 50.),
        #                                         order=elastic_transform_config.get('order', 0)))

        # random slide augmentation
        if self.master_config.get('random_slides', False):
            # TODO slide probability
            ouput_shape = self.master_config.get('shape_after_slide', None)
            max_misalign = self.master_config.get('max_misalign', None)
            transforms.add(RandomSlide(output_image_size=ouput_shape, max_misalign=max_misalign))

        # affinity transforms for affinity targets
        # we apply the affinity target calculation only to the segmentation (1)
        assert self.affinity_config is not None
        transforms.add(affinity_config_to_transform(apply_to=[1], **self.affinity_config))

        # crop invalid affinity labels and elastic augment reflection padding assymetrically
        crop_config = self.master_config.get('crop_after_target', {})
        if crop_config:
            # One might need to crop after elastic transform to avoid edge artefacts of affinity
            # computation being warped into the FOV.
            transforms.add(VolumeAsymmetricCrop(**crop_config))

        from vaeAffs.transforms import PassGTBoundaries_HackyHackyReloaded
        transforms.add(PassGTBoundaries_HackyHackyReloaded())

        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        name = config.get('dataset_name')
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        defect_augmentation_config = config.get('defect_augmentation_config')
        master_config = config.get('master_config')
        return cls(name, volume_config=volume_config,
                   slicing_config=slicing_config,
                   defect_augmentation_config=defect_augmentation_config,
                   master_config=master_config)


class CremiDatasets(Concatenate):
    def __init__(self, names,
                 volume_config,
                 slicing_config,
                 defect_augmentation_config,
                 master_config=None):
        # Make datasets and concatenate
        if names is None:
            datasets = [CremiDataset(name=None,
                                     volume_config=volume_config,
                                     slicing_config=slicing_config,
                                     defect_augmentation_config=defect_augmentation_config,
                                     master_config=master_config)]
        else:
            datasets = [CremiDataset(name=name,
                                     volume_config=volume_config,
                                     slicing_config=slicing_config,
                                     defect_augmentation_config=defect_augmentation_config,
                                     master_config=master_config)
                        for name in names]
        super().__init__(*datasets)
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = AsTorchBatch(3)
        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        names = config.get('names')
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        defect_augmentation_config = config.get('defect_augmentation_config')
        master_config = config.get('master_config')
        return cls(names=names, volume_config=volume_config,
                   defect_augmentation_config=defect_augmentation_config,
                   slicing_config=slicing_config, master_config=master_config)


class CremiDatasetInference(RawVolume):
    def __init__(self, transform_config, **super_kwargs):
        super(CremiDatasetInference, self).__init__(return_index_spec=True,
                                                    **super_kwargs)
        self.transforms = self.get_additional_transforms(transform_config)

    def get_additional_transforms(self, transform_config):
        transforms = self.transforms if self.transforms is not None else Compose()

        stack_scaling_factors = transform_config["stack_scaling_factors"]

        # Replicate and downscale batch:
        num_inputs = len(stack_scaling_factors)
        input_indices = list(range(num_inputs))

        transforms.add(ReplicateBatch(num_inputs))
        inv_scaling_facts = deepcopy(stack_scaling_factors)
        inv_scaling_facts.reverse()
        for in_idx, dws_fact, crop_fact in zip(input_indices, stack_scaling_factors,
                                                         inv_scaling_facts):
            transforms.add(DownsampleAndCrop3D(apply_to=[in_idx], order=2, zoom_factor=dws_fact, crop_factor=crop_fact))

        transforms.add(AsTorchBatch(3))

        return transforms

class CremiDatasetsInference(Concatenate):
    def __init__(self, names,
                 transform_config,
                 raw_volume_kwargs):
        names = [None] if names is None else names

        datasets = [CremiDatasetInference(transform_config,
                                  name=name,
                                  **raw_volume_kwargs)
                for name in names]
        super().__init__(*datasets)
        # self.transforms = self.get_transforms()
    #
    # def get_transforms(self):
    #     FIXME: avoid to apply to index...
        # transforms = AsTorchBatch(3, apply_to=[0])
        # return transforms


def get_cremi_loader(config):
    """
    Get Cremi loader given a the path to a configuration file.

    Parameters
    ----------
    config : str or dict
        (Path to) Data configuration.

    Returns
    -------
    torch.utils.data.dataloader.DataLoader
        Data loader built as configured.
    """
    config = yaml2dict(config)
    loader_config = config.get('loader_config')
    inference_mode = config.get('inference_mode', False)

    if inference_mode:
        datasets = CremiDatasetInference(
            config.get("transforms_config"),
            name=config.get('name'),
            **config.get('raw_volume_kwargs'))
        # Avoid to wrap arrays into tensors:
        loader_config["collate_fn"] = collate_indices
    else:
        datasets = CremiDatasets.from_config(config)
    # Don't wrap stuff in tensors:
    loader = DataLoader(datasets, **loader_config)
    return loader


def collate_indices(batch):
    tensor_list = [itm[0] for itm in batch]
    indices_list = [itm[1] for itm in batch]
    return default_collate(tensor_list), indices_list
