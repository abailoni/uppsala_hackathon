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


class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(VAE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        self.e3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf * 4)

        self.e4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        self.e5 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf * 8)
        self.global_pool = GlobalMeanPooling()

        self.fc1 = nn.Conv2d(ndf * 8, latent_variable_size, 1, 1, 0)
        self.fc2 = nn.Conv2d(ndf * 8, latent_variable_size, 1, 1, 0)

        # decoder
        self.d1 = nn.Conv2d(latent_variable_size, ngf * 8 * 2, 1, 1, 0)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf * 8 * 2, ngf * 8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf * 8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf * 8, ngf * 4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf * 4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf * 4, ngf * 2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf * 2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf * 2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.e5(h4))

        spatial_size = h5.size()[2]

        # h5 = self.global_pool(h5)

        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # if args.cuda:
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        # else:
        #     eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def check_dim(self, x):
        if x.dim() != 4:
            assert x.dim() == 5
            # Get rid of third dimension:
            x = x[:, :, 0]
        return x

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        # h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

    def get_latent_var(self, x):
        x = self.check_dim(x)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        x_shape = x.size()
        x = self.check_dim(x)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        res = res.view(*x_shape)
        return [res, mu, logvar]


class VAE_debug(VAE):
    def __init__(self, nc, ngf, ndf, latent_variable_size, unfold_size=6):
        super(VAE_debug, self).__init__(nc, ngf, ndf, latent_variable_size)

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, kernel_size=3, stride=1, dilation=2)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, dilation=2)
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        # self.e3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=1, dilation=2)
        # self.bn3 = nn.BatchNorm2d(ndf*4)
        #
        # self.e4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=3, stride=1, dilation=2)
        # self.bn4 = nn.BatchNorm2d(ndf*8)
        #
        # self.e5 = nn.Conv2d(ndf*8, ndf*8, kernel_size=3, stride=1, dilation=2)
        # self.bn5 = nn.BatchNorm2d(ndf*8)

        self.unfold = nn.Unfold(kernel_size=unfold_size)

        self.fc1 = nn.Conv1d(ndf * 2 * unfold_size * unfold_size, latent_variable_size, 1, 1, 0)
        self.fc2 = nn.Conv1d(ndf * 2 * unfold_size * unfold_size, latent_variable_size, 1, 1, 0)

        # decoder
        self.d1 = nn.Conv1d(latent_variable_size, ngf * 2 * unfold_size * unfold_size, 1, 1, 0)

        # self.up1 = nn.functional.interpolate(scale_factor=2, mode='bilinear')
        # self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.fold = nn.Fold(output_size=(unfold_size, unfold_size), kernel_size=unfold_size)

        # self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=0, dilation=2)
        self.bn6 = nn.BatchNorm2d(ngf, 1.e-3)

        # self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.ConvTranspose2d(ngf, nc, kernel_size=3, stride=1, padding=0, dilation=2)
        # self.bn7 = nn.BatchNorm2d(ngf, 1.e-3)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))

        unfolded = self.unfold(h2)

        return self.fc1(unfolded), self.fc2(unfolded)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        nb_blocks = h1.size()[2]
        assert nb_blocks == 1

        h1 = self.fold(h1)
        # h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(h1)))
        h3 = self.d3(h2)
        # h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        # h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(h3)

    def forward(self, x):
        x_shape = x.size()
        x = self.check_dim(x)
        mu, logvar = self.encode(x)
        # z = self.reparametrize(mu, logvar)
        res = self.decode(mu)
        res = res.view(*x_shape)
        return [res, mu, logvar]


class LeakyBatchNormConv2D(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size=3, stride=1, dilation=1):
        super(LeakyBatchNormConv2D, self).__init__()
        self.conv = nn.Conv2d(input_ch, output_ch, kernel_size=kernel_size,
                              stride=stride,
                              dilation=dilation)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm2d(output_ch)

    def forward(self, tensor):
        return self.leakyrelu(self.bn(self.conv(tensor)))


class LeakyBatchNormTraspConv2D(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size=3, stride=1, dilation=1,
                 apply_post_conv=True):
        super(LeakyBatchNormTraspConv2D, self).__init__()
        self.conv = nn.ConvTranspose2d(input_ch, output_ch, kernel_size=kernel_size,
                                       stride=stride,
                                       dilation=dilation)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm2d(output_ch)
        self.appy_post_conv = apply_post_conv

    def forward(self, tensor):
        if self.appy_post_conv:
            return self.leakyrelu(self.bn(self.conv(tensor)))
        else:
            return self.conv(tensor)


from neurofire.models.unet.unet_3d import CONV_TYPES, Decoder, DecoderResidual, BaseResidual, Base, Output, Encoder, \
    EncoderResidual

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
                 # patch_shape=(1,27,27),
                 latent_variable_size=24,
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

        # assert isinstance(patch_shape, tuple)
        # assert len(patch_shape) == 3
        self._min_patch_shape = None

        assert isinstance(latent_variable_size, int)
        self.latent_variable_size = latent_variable_size

    def encode(self, input_):
        x = input_
        encoder_out = []
        # apply encoders and remember their outputs
        for encoder in self.encoders:
            x = encoder(x)
            encoder_out.append(x)

        N = x.shape[0]
        self.set_min_patch_shape(x.shape[2:])
        x = x.view(N, -1, 1)

        # x = unfold_3d(x, kernel_size=self.unfold_size)

        encoded_variable = self.base[0](x)

        mu, log_var = encoded_variable[:,:self.latent_variable_size], encoded_variable[:,self.latent_variable_size:]
        return mu, log_var

    def set_min_patch_shape(self, shape):
        # TODO: assert tuple and not list
        if self._min_patch_shape is None:
            self._min_patch_shape = shape
        else:
            assert self._min_patch_shape == shape

    def decode(self, encoded_variable):
        assert self._min_patch_shape is not None

        x = self.base[1](encoded_variable)
        N = x.shape[0]
        assert x.shape[-1] == 1
        x = x.view(N, -1, *self._min_patch_shape)

        # x = fold_3d(x, output_size=(self.unfold_size, self.unfold_size), kernel_size=self.unfold_size)

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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate(self, shape):
        return torch.randn(shape)

    def forward(self, input_):
        # encoded_variable = self.encode(input_)
        mu, logvar = self.encode(input_)

        z = self.reparameterize(mu, logvar)
        # z2 = self.reparameterize(mu, logvar)

        return [self.decode(z), mu, logvar]


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
                 patch_size=(3,27,27),
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
        # TODO: can I get rid of this ugly shape?
        folded_shape = np.array(patch_size).prod()
        base_pre = nn.Conv1d(f0e * folded_shape, latent_variable_size * 2, 1, 1, 0)
        base_post = nn.Conv1d(latent_variable_size, f0e * folded_shape, 1, 1, 0)
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
                                          latent_variable_size=latent_variable_size)




class AE_loss(nn.Module):
    def __init__(self):
        super(AE_loss, self).__init__()
        # self.reconstruction_function = nn.BCELoss()
        # self.reconstruction_function.size_average = False
        # self.reconstruction_function = SorensenDiceLoss()
        self.reconstruction_function = nn.MSELoss()

    def forward(self, predictions, target):
        x = target[:, :, 0]
        recon_x, mu, logvar = predictions
        recon_x = recon_x[:, :, 0]
        BCE = self.reconstruction_function(recon_x, x)

        return BCE

        # # https://arxiv.org/abs/1312.6114 (Appendix B)
        # # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        # KLD = torch.sum(KLD_element).mul_(-0.5)
        #
        # return BCE + KLD



class VAE_loss(nn.Module):
    def __init__(self):
        super(VAE_loss, self).__init__()
        self.reconstruction_loss = nn.BCELoss(reduction="sum")
        # self.reconstruction_function.size_average = False
        # self.reconstruction_function = SorensenDiceLoss()
        # self.reconstruction_function = nn.MSELoss()

    def forward(self, predictions, target):
        # x = target[:, :, 0]
        recon_x, mu, logvar = predictions

        # Reconstruction loss:
        # BCE = 0
        BCE = self.reconstruction_loss(recon_x, target)


        # BCE = self.reconstruction_function(recon_x, target)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


        return BCE + KLD

        # # https://arxiv.org/abs/1312.6114 (Appendix B)
        # # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        # KLD = torch.sum(KLD_element).mul_(-0.5)
        #
        # return BCE + KLD

# def train(epoch):
#     model.train()
#     train_loss = 0
#     for batch_idx in train_loader:
#         data = load_batch(batch_idx, True)
#         data = Variable(data)
#         if args.cuda:
#             data = data.cuda()
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data)
#         loss = loss_function(recon_batch, data, mu, logvar)
#         loss.backward()
#         train_loss += loss.data[0]
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), (len(train_loader)*128),
#                 100. * batch_idx / len(train_loader),
#                 loss.data[0] / len(data)))
#
#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#           epoch, train_loss / (len(train_loader)*128)))
#     return train_loss / (len(train_loader)*128)
#
# def test(epoch):
#     model.eval()
#     test_loss = 0
#     for batch_idx in test_loader:
#         data = load_batch(batch_idx, False)
#         data = Variable(data, volatile=True)
#         if args.cuda:
#             data = data.cuda()
#         recon_batch, mu, logvar = model(data)
#         test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
#
#         torchvision.utils.save_image(data.data, '../imgs/Epoch_{}_data.jpg'.format(epoch), nrow=8, padding=2)
#         torchvision.utils.save_image(recon_batch.data, '../imgs/Epoch_{}_recon.jpg'.format(epoch), nrow=8, padding=2)
#
#     test_loss /= (len(test_loader)*128)
#     print('====> Test set loss: {:.4f}'.format(test_loss))
#     return test_loss
#
#
# def perform_latent_space_arithmatics(items): # input is list of tuples of 3 [(a1,b1,c1), (a2,b2,c2)]
#     load_last_model()
#     model.eval()
#     data = [im for item in items for im in item]
#     data = [totensor(i) for i in data]
#     data = torch.stack(data, dim=0)
#     data = Variable(data, volatile=True)
#     if args.cuda:
#         data = data.cuda()
#     z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
#     it = iter(z.split(1))
#     z = zip(it, it, it)
#     zs = []
#     numsample = 11
#     for i,j,k in z:
#         for factor in np.linspace(0,1,numsample):
#             zs.append((i-j)*factor+k)
#     z = torch.cat(zs, 0)
#     recon = model.decode(z)
#
#     it1 = iter(data.split(1))
#     it2 = [iter(recon.split(1))]*numsample
#     result = zip(it1, it1, it1, *it2)
#     result = [im for item in result for im in item]
#
#     result = torch.cat(result, 0)
#     torchvision.utils.save_image(result.data, '../imgs/vec_math.jpg', nrow=3+numsample, padding=2)
#
#
# def latent_space_transition(items): # input is list of tuples of  (a,b)
#     load_last_model()
#     model.eval()
#     data = [im for item in items for im in item[:-1]]
#     data = [totensor(i) for i in data]
#     data = torch.stack(data, dim=0)
#     data = Variable(data, volatile=True)
#     if args.cuda:
#         data = data.cuda()
#     z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
#     it = iter(z.split(1))
#     z = zip(it, it)
#     zs = []
#     numsample = 11
#     for i,j in z:
#         for factor in np.linspace(0,1,numsample):
#             zs.append(i+(j-i)*factor)
#     z = torch.cat(zs, 0)
#     recon = model.decode(z)
#
#     it1 = iter(data.split(1))
#     it2 = [iter(recon.split(1))]*numsample
#     result = zip(it1, it1, *it2)
#     result = [im for item in result for im in item]
#
#     result = torch.cat(result, 0)
#     torchvision.utils.save_image(result.data, '../imgs/trans.jpg', nrow=2+numsample, padding=2)
#
#
# def rand_faces(num=5):
#     load_last_model()
#     model.eval()
#     z = torch.randn(num*num, model.latent_variable_size)
#     z = Variable(z, volatile=True)
#     if args.cuda:
#         z = z.cuda()
#     recon = model.decode(z)
#     torchvision.utils.save_image(recon.data, '../imgs/rand_faces.jpg', nrow=num, padding=2)
#
# def load_last_model():
#     models = glob('../models/*.pth')
#     model_ids = [(int(f.split('_')[1]), f) for f in models]
#     start_epoch, last_cp = max(model_ids, key=lambda item:item[0])
#     print('Last checkpoint: ', last_cp)
#     model.load_state_dict(torch.load(last_cp))
#     return start_epoch, last_cp
#
# def resume_training():
#     start_epoch, _ = load_last_model()
#
#     for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
#         train_loss = train(epoch)
#         test_loss = test(epoch)
#         torch.save(model.state_dict(), '../models/Epoch_{}_Train_loss_{:.4f}_Test_loss_{:.4f}.pth'.format(epoch, train_loss, test_loss))
#
# def last_model_to_cpu():
#     _, last_cp = load_last_model()
#     model.cpu()
#     torch.save(model.state_dict(), '../models/cpu_'+last_cp.split('/')[-1])
#
# if __name__ == '__main__':
#     resume_training()
#     # last_model_to_cpu()
#     # load_last_model()
#     # rand_faces(10)
#     # da = load_pickle(test_loader[0])
#     # da = da[:120]
#     # it = iter(da)
#     # l = zip(it, it, it)
#     # # latent_space_transition(l)
#     # perform_latent_space_arithmatics(l)
