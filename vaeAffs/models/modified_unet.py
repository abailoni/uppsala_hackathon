from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from glob import glob
from .util import *
import numpy as np
from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss

from inferno.extensions.layers.reshape import GlobalMeanPooling


from neurofire.models.unet.unet_3d import CONV_TYPES, Decoder, DecoderResidual, BaseResidual, Base, Output, Encoder, \
    EncoderResidual


class EncodingLoss(nn.Module):
    def __init__(self, path_autoencoder_model):
        super(EncodingLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.soresen_loss = SorensenDiceLoss()

        assert isinstance(path_autoencoder_model, str)
        AE_model = torch.load(path_autoencoder_model)
        # Freeze the auto-encoder model:
        self.AE_model = AE_model
        for param in self.AE_model.parameters():
            param.requires_grad = False


    def forward(self, predictions, target):
        # FIXME: atm I hard-coded the dimensions
        patch_slice = (slice(None), slice(None), slice(0,1), slice(27,54), slice(27,54))
        patch_slice = (slice(None), slice(None), slice(0,1), slice(27,54), slice(27,54))
        target_patch = target
        with torch.no_grad():
            target_embedded_patch = self.AE_model.encode(target_patch)[...,0]
        predicted_embedded_patch = predictions[:, :, 0, 13, 13]

        # with torch.no_grad():
        self.emb_prediction = self.AE_model.decode(predicted_embedded_patch.unsqueeze(2))

        # target_embedded_patch = target_embedded_patch.repeat(27, 27, 1, 1)
        # target_embedded_patch = target_embedded_patch.permute(2,3,0,1)
        # MSE = self.loss(predictions[:,:,0], target_embedded_patch)
        loss = self.soresen_loss(self.emb_prediction, target)

        return loss


def unfold_3d(x, *args, **kwargs):
    unfolded = []
    for z in range(x.shape[2]):
        unfolded_slice = nn.functional.unfold(x[:,:,z], *args, **kwargs)
        assert unfolded_slice.shape[-1] == 1, "kernel size does not match! Trying to unfold a patch of size {} with kernel {}".format(x.shape[-1], kwargs["kernel_size"])
        unfolded.append(unfolded_slice)
    x = torch.cat(unfolded, dim=2)
    return x

def fold_3d(x, *args, **kwargs):
    folded = []
    for z in range(x.shape[2]):
        folded_slice = nn.functional.fold(x[:,:,[z]], *args, **kwargs)
        folded.append(folded_slice)
    x = torch.stack(folded, dim=2)
    return x

class AutoEncoderSkeleton(nn.Module):
    def __init__(self, encoders, base, decoders, output,
                 unfold_size=1,
                 final_activation=None):
        super(AutoEncoderSkeleton, self).__init__()
        assert isinstance(encoders, list)
        assert isinstance(decoders, list)

        assert len(encoders) == len(decoders), "%i, %i" % (len(encoders), len(decoders))
        assert isinstance(base, list)
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)

        self.base = nn.ModuleList(base)
        self.output = output
        if isinstance(final_activation, str):
            self.final_activation = getattr(nn, final_activation)()
        elif isinstance(final_activation, nn.Module):
            self.final_activation = final_activation
        elif final_activation is None:
            self.final_activation = None
        else:
            raise NotImplementedError

        assert isinstance(unfold_size, int)
        self.unfold_size = unfold_size

    def encode(self, input_):
        x = input_
        encoder_out = []
        # apply encoders and remember their outputs
        for encoder in self.encoders:
            x = encoder(x)
            encoder_out.append(x)

        x = unfold_3d(x, kernel_size=self.unfold_size)

        encoded_variable = self.base[0](x)

        return encoded_variable

    def decode(self, encoded_variable):
        x = self.base[1](encoded_variable)
        x = fold_3d(x, output_size=(self.unfold_size, self.unfold_size), kernel_size=self.unfold_size)

        # apply decoders
        max_level = len(self.decoders) - 1
        for level, decoder in enumerate(self.decoders):
            # the decoder gets input from the previous decoder and the encoder
            # from the same level
            x = decoder(x)

        x = self.output(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x

    def forward(self, input_):
        encoded_variable = self.encode(input_)
        x = self.decode(encoded_variable)
        return x


class AutoEncoder(AutoEncoderSkeleton):
    """
    3D U-Net architecture.
    """

    def __init__(self,
                 in_channels,
                 initial_num_fmaps,
                 fmap_growth,
                 latent_variable_size=128,
                 scale_factor=2,
                 final_activation='auto',
                 conv_type_key='vanilla',
                 unfold_size=1,
                 add_residual_connections=False):
        """
        Parameter:
        ----------
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        initial_num_fmaps (int): number of feature maps of the first layer
        fmap_growth (int): growth factor of the feature maps; the number of feature maps
        in layer k is given by initial_num_fmaps * fmap_growth**k
        final_activation:  final activation used
        scale_factor (int or list / tuple): upscale / downscale factor (default: 2)
        final_activation:  final activation used (default: 'auto')
        conv_type_key: convolutin type
        """
        # validate conv-type
        assert conv_type_key in CONV_TYPES, conv_type_key
        conv_type = CONV_TYPES[conv_type_key]

        # validate scale factor
        assert isinstance(scale_factor, (int, list, tuple))
        self.scale_factor = [scale_factor] * 3 if isinstance(scale_factor, int) else scale_factor
        assert len(self.scale_factor) == 1
        # NOTE individual scale factors can have multiple entries for anisotropic sampling
        assert all(isinstance(sfactor, (int, list, tuple))
                   for sfactor in self.scale_factor)

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = in_channels

        conv_type = CONV_TYPES[conv_type_key]
        decoder_type = DecoderResidual if add_residual_connections else Decoder
        encoder_type = EncoderResidual if add_residual_connections else Encoder
        base_type = BaseResidual if add_residual_connections else Base

        # Build encoders with proper number of feature maps
        f0e = initial_num_fmaps
        f1e = initial_num_fmaps * fmap_growth
        f2e = initial_num_fmaps * fmap_growth ** 2
        encoders = [
            encoder_type(in_channels, f0e, 3, self.scale_factor[0], conv_type=conv_type),
            # encoder_type(f0e, f1e, 3, self.scale_factor[1], conv_type=conv_type),
            # encoder_type(f1e, f2e, 3, self.scale_factor[2], conv_type=conv_type)
        ]

        # Build base
        # number of base output feature maps
        # f0b = initial_num_fmaps * fmap_growth ** 3
        base_pre = nn.Conv1d(f0e * unfold_size * unfold_size, latent_variable_size, 1, 1, 0)
        base_post = nn.Conv1d(latent_variable_size, f0e * unfold_size * unfold_size, 1, 1, 0)
        base = [base_pre, base_post]

        # Build decoders (same number of feature maps as MALA)
        f2d = initial_num_fmaps * fmap_growth ** 2
        f1d = initial_num_fmaps * fmap_growth
        f0d = initial_num_fmaps
        decoders = [
            # decoder_type(f2e, f2d, 3, self.scale_factor[2], conv_type=conv_type),
            # decoder_type(f1d, f1d, 3, self.scale_factor[1], conv_type=conv_type),
            decoder_type(f0d, f0d, 3, self.scale_factor[0], conv_type=conv_type)
        ]

        # Build output
        output = Output(f0d, in_channels, 3)
        # Parse final activation
        if final_activation == 'auto':
            final_activation = nn.Sigmoid() if in_channels == 1 else nn.Softmax2d()

        # Build the architecture
        super(AutoEncoder, self).__init__(encoders=encoders,
                                          base=base,
                                          decoders=decoders,
                                          output=output,
                                          final_activation=final_activation,
                                          unfold_size=unfold_size)

