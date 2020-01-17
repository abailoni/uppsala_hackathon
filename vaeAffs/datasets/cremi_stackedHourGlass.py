from copy import deepcopy

from inferno.io.core import ZipReject, Concatenate
from inferno.io.transform import Compose, Transform
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
    def __init__(self, threshold, threshold_zero_label=1.,
                 defected_label=None):
        """
        :param threshold: If the biggest segment takes more than 'threshold', batch is rejected
        :param threshold_zero_label: if the percentage of non-zero-labels is less than this, reject
        """
        self.threshold = threshold
        self.threshold_zero_label = threshold_zero_label
        self.defected_label = defected_label

    def __call__(self, fetched):
        # # Check if we have a defected slice at the beginning of the batch:
        # if self.defected_label is not None:
        #     if ((fetched[0].astype('int64') == self.defected_label).sum() != 0) or \
        #             (fetched[-1].astype('int64') == self.defected_label).sum() != 0:
        #         # Check if we should reject:
        #         print("!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!)")
        #         print("Batch rejected because of defected slice first/last!")
        #         return True

        _, counts = np.unique(fetched, return_counts=True)
        # Check if we should reject:
        return ((float(np.max(counts)) / fetched.size) > self.threshold) or (
                    (np.count_nonzero(fetched) / fetched.size) < self.threshold_zero_label)

class DuplicateGtDefectedSlices(Transform):
    def __init__(self, defected_label=-1, keep_defected_mask=False, **super_kwargs):
        self.defected_label = defected_label
        self.keep_defected_mask = keep_defected_mask
        super(DuplicateGtDefectedSlices, self).__init__(**super_kwargs)

    def batch_function(self, batch):
        assert len(batch) == 3
        raw, defected_mask, GT = batch

        new_GT = GT.copy()

        # Check for defected slices in the dataset: (entirely labelled with defected_label)
        for z_idx in range(new_GT.shape[0]):
            if (new_GT[z_idx].astype('int64') == self.defected_label).sum() != 0:
                if z_idx == 0:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!! WARNING 1 !!!!!!!!!!!!!!!!!!!!!!!!!!)")
                    continue
                # assert z_idx != 0, "The first slice should never be defected"
                # Copy ground truth from previous slice:
                new_GT[z_idx] = new_GT[z_idx-1]

        # Modify new_GT for defect-augmented slices:
        counter = 0
        for z_idx, is_defected in enumerate(defected_mask[:,0,0]):
            if is_defected:
                if z_idx == 0:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!! WARNING 2 !!!!!!!!!!!!!!!!!!!!!!!!!!)")
                    continue

                counter += 1
                # assert z_idx != 0, "The first slice should never be defected"
                # Copy ground truth from previous slice:
                new_GT[z_idx] = new_GT[z_idx-1]

        # if defected_mask.max() or counter != 0:
        #     print("Defected: {}; number: {}".format(defected_mask.max(), counter))


        if self.keep_defected_mask:
            return (raw, defected_mask.astype('int64'), new_GT.astype('int64'))
        else:
            return (raw, new_GT.astype('int64'))

class AdjustBatch(Transform):

    def batch_function(self, batch):
        assert len(batch) == 2
        raw_inputs, GT = batch
        assert len(raw_inputs) == 2
        raw, defected_mask = raw_inputs
        return (raw, defected_mask, GT)


class CremiDataset(ZipReject):
    def __init__(self, name, volume_config, slicing_config,
                 defect_augmentation_config, master_config=None):
        assert isinstance(volume_config, dict)
        assert isinstance(slicing_config, dict)
        assert isinstance(defect_augmentation_config, dict)
        assert 'raw' in volume_config
        assert 'segmentation' in volume_config

        volume_config = deepcopy(volume_config)

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

        rejection_threshold = volume_config.get('rejection_threshold', 0.5)
        super().__init__(self.raw_volume, self.segmentation_volume,
                         sync=True, rejection_dataset_indices=1,
                         rejection_criterion=RejectSingleLabelVolumes(1.0, rejection_threshold,
                                                                      defected_label=master_config.get('duplicate_GT_defected_slices', {}).get('defect_label', -1)))
        # Set master config (for transforms)
        self.master_config = {} if master_config is None else deepcopy(master_config)
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose()

        if self.master_config.get('random_flip', False):
            transforms.add(AdjustBatch())
            transforms.add(RandomFlip3D())
            transforms.add(RandomRotate())

        if self.master_config.get('duplicate_GT_defected_slices', False):
            transforms.add(DuplicateGtDefectedSlices(**self.master_config['duplicate_GT_defected_slices']))


        # Elastic transforms can be skipped by
        # setting elastic_transform to false in the
        # yaml config file.
        if self.master_config.get('elastic_transform'):
            elastic_transform_config = self.master_config.get('elastic_transform')
            if elastic_transform_config.get('apply', False):
                transforms.add(ElasticTransform(alpha=elastic_transform_config.get('alpha', 2000.),
                                                sigma=elastic_transform_config.get('sigma', 50.),
                                                order=elastic_transform_config.get('order', 0)))

        # random slide augmentation
        if self.master_config.get('random_slides') is not None:
            # TODO slide probability
            random_slides_config = deepcopy(self.master_config.get('random_slides'))
            ouput_shape = random_slides_config.pop('shape_after_slide', None)
            max_misalign = random_slides_config.pop('max_misalign', None)
            transforms.add(RandomSlide(output_image_size=ouput_shape, max_misalign=max_misalign,
                                       index_mask_defected_slices=1 if self.master_config["duplicate_GT_defected_slices"].get("keep_defected_mask", False) else None,
                                       **random_slides_config))

        # Replicate and downscale batch:
        input_indices, target_indices = [0], [1]
        num_inputs = 1
        if self.master_config.get("downscale_and_crop") is not None:
            ds_config = self.master_config.get("downscale_and_crop")
            replicate_targets = ds_config.pop("replicate_targets", False)
            assert len(ds_config) >= 1
            num_inputs = len(ds_config)
            input_indices = list(range(num_inputs))
            target_indices = list(range(num_inputs, 2*num_inputs)) if replicate_targets else [num_inputs]

            transforms.add(ReplicateBatch(num_inputs, replicate_targets=replicate_targets))
            for i, in_idx in enumerate(input_indices):
                kwargs = ds_config[in_idx]
                transforms.add(DownsampleAndCrop3D(apply_to=[in_idx], order=2, **kwargs))
                if replicate_targets:
                    transforms.add(
                        DownsampleAndCrop3D(apply_to=[target_indices[i]], order=0, **kwargs))

        # # affinity transforms for affinity targets
        # # we apply the affinity target calculation only to the segmentation (1)
        if self.master_config.get("affinity_config") is not None:
            affs_config = deepcopy(self.master_config.get("affinity_config"))
            global_kwargs = affs_config.pop("global", {})

            use_dynamic_offsets = affs_config.pop("use_dynamic_offsets", False)
            aff_transform = Segmentation2AffinitiesDynamicOffsets if use_dynamic_offsets else affinity_config_to_transform

            for input_index in affs_config:
                affs_kwargs = deepcopy(global_kwargs)
                affs_kwargs.update(affs_config[input_index])
                transforms.add(aff_transform(apply_to=[input_index+num_inputs], **affs_kwargs))

        # crop invalid affinity labels and elastic augment reflection padding assymetrically
        crop_config = self.master_config.get('crop_after_target', {})
        if crop_config:
            # One might need to crop after elastic transform to avoid edge artefacts of affinity
            # computation being warped into the FOV.
            transforms.add(VolumeAsymmetricCrop(**crop_config))

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


class CheckBatchAndChannelDim(Transform):
    def __init__(self, dimensionality, *super_args, **super_kwargs):
        super(CheckBatchAndChannelDim, self).__init__(*super_args, **super_kwargs)
        self.dimensionality = dimensionality

    def batch_function(self, batch):
        output_batch = []
        for tensor in batch:
            if tensor.ndimension() == self.dimensionality:
                output_batch.append(tensor.unsqueeze(0).unsqueeze(0))
            elif tensor.ndimension() == self.dimensionality + 1:
                output_batch.append(tensor.unsqueeze(0))
            elif tensor.ndimension() == self.dimensionality + 2:
                output_batch.append(tensor)
            else:
                raise ValueError
        return tuple(output_batch)


class CremiDatasetInference(RawVolume):
    def __init__(self, master_config, **super_kwargs):
        super(CremiDatasetInference, self).__init__(return_index_spec=True,
                                                    **super_kwargs)
        self.transforms = self.get_additional_transforms(master_config)

    def get_additional_transforms(self, master_config):
        transforms = self.transforms if self.transforms is not None else Compose()

        master_config = {} if master_config is None else master_config
        # TODO: somehow merge with the trainer loader...

        # Replicate and downscale batch:
        input_indices, target_indices = [0], [1]
        if master_config.get("downscale_and_crop") is not None:
            ds_config = master_config.get("downscale_and_crop")
            replicate_targets = ds_config.pop("replicate_targets", False)
            assert len(ds_config) >= 1
            num_inputs = len(ds_config)
            input_indices = list(range(num_inputs))
            target_indices = list(range(num_inputs, 2*num_inputs)) if replicate_targets else [num_inputs]

            transforms.add(ReplicateBatch(num_inputs, replicate_targets=replicate_targets))
            for i, in_idx in enumerate(input_indices):
                kwargs = ds_config[in_idx]
                transforms.add(DownsampleAndCrop3D(apply_to=[in_idx], order=2, **kwargs))
                if replicate_targets:
                    transforms.add(
                        DownsampleAndCrop3D(apply_to=[target_indices[i]], order=0, **kwargs))

        # # # affinity transforms for affinity targets
        # # # we apply the affinity target calculation only to the segmentation (1)
        # if master_config.get("affinity_config") is not None:
        #     affs_config = master_config.get("affinity_config")
        #     global_kwargs = affs_config.pop("global", {})
        #     # TODO: define computed affs not in this way, but with a variable in config...
        #     nb_affs = len(affs_config)
        #     assert nb_affs == num_inputs
        #     # all_affs_kwargs = [deepcopy(global_kwargs) for _ in range(nb_affs)]
        #     for input_index in affs_config:
        #         affs_kwargs = deepcopy(global_kwargs)
        #         affs_kwargs.update(affs_config[input_index])
        #         transforms.add(affinity_config_to_transform(apply_to=[input_index+num_inputs], **affs_kwargs))

        # crop invalid affinity labels and elastic augment reflection padding assymetrically
        crop_config = master_config.get('crop_after_target', {})
        if crop_config:
            # One might need to crop after elastic transform to avoid edge artefacts of affinity
            # computation being warped into the FOV.
            transforms.add(VolumeAsymmetricCrop(**crop_config))

        transforms.add(AsTorchBatch(3, add_channel_axis_if_necessary=True))

        # transforms.add(CheckBatchAndChannelDim(3))

        return transforms

# class CremiDatasetsInference(Concatenate):
#     def __init__(self, names,
#                  transform_config,
#                  raw_volume_kwargs):
#         names = [None] if names is None else names
#
#         datasets = [CremiDatasetInference(transform_config,
#                                   name=name,
#                                   **raw_volume_kwargs)
#                 for name in names]
#         super().__init__(*datasets)
#         # self.transforms = self.get_transforms()
#     #
#     # def get_transforms(self):
#     #     FIXME: avoid to apply to index...
#         # transforms = AsTorchBatch(3, apply_to=[0])
#         # return transforms


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
            config.get("master_config"),
            name=config.get('name'),
            **config.get('volume_config'))
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
