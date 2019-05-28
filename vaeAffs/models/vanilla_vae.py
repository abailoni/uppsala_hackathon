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

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*8)
        self.global_pool = GlobalMeanPooling()

        self.fc1 = nn.Conv2d(ndf * 8, latent_variable_size, 1, 1, 0)
        self.fc2 = nn.Conv2d(ndf * 8, latent_variable_size, 1, 1, 0)

        # decoder
        self.d1 = nn.Conv2d(latent_variable_size, ngf*8*2, 1, 1, 0)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
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
            x = x[:,:,0]
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

        self.e2 = nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=1, dilation=2)
        self.bn2 = nn.BatchNorm2d(ndf*2)

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




class VAE_bigger(VAE):
    def __init__(self, input_ch, encoder_fact, decoder_fact, latent_variable_size, unfold_size=6):
        super(VAE_bigger, self).__init__(input_ch, encoder_fact, decoder_fact, latent_variable_size)

        # encoder
        self.layer1 = LeakyBatchNormConv2D(input_ch, decoder_fact, kernel_size=5, stride=1, dilation=1)
        self.layer2 = LeakyBatchNormConv2D(decoder_fact, decoder_fact, kernel_size=3, stride=1, dilation=3)
        self.layer3 = LeakyBatchNormConv2D(decoder_fact, decoder_fact*2, kernel_size=3, stride=1, dilation=4)
        self.layer4 = LeakyBatchNormConv2D(decoder_fact*2, decoder_fact*4, kernel_size=3, stride=1, dilation=4)

        self.unfold = nn.Unfold(kernel_size=unfold_size)

        self.fc1 = nn.Conv1d(decoder_fact * 4 * unfold_size * unfold_size, latent_variable_size, 1, 1, 0)

        # decoder
        self.d1 = nn.Conv1d(latent_variable_size, encoder_fact * 4 * unfold_size * unfold_size, 1, 1, 0)

        self.fold = nn.Fold(output_size=(unfold_size, unfold_size), kernel_size=unfold_size)
        self.trans_layer1 = LeakyBatchNormTraspConv2D(encoder_fact * 4, encoder_fact * 2, kernel_size=3, stride=1, dilation=4)
        self.trans_layer2 = LeakyBatchNormTraspConv2D(encoder_fact * 2, encoder_fact, kernel_size=3, stride=1, dilation=4)
        self.trans_layer3 = LeakyBatchNormTraspConv2D(encoder_fact, encoder_fact, kernel_size=3, stride=1, dilation=3)
        self.trans_layer4 = LeakyBatchNormTraspConv2D(encoder_fact, input_ch, kernel_size=5, stride=1, dilation=1,
                                                      apply_post_conv=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)
        unfolded = self.unfold(h4)
        return self.fc1(unfolded)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        nb_blocks = h1.size()[2]
        assert nb_blocks == 1

        h1 = self.fold(h1)
        h2 = self.trans_layer1(h1)
        h3 = self.trans_layer2(h2)
        h4 = self.trans_layer3(h3)
        h5 = self.trans_layer4(h4)

        return self.sigmoid(h5)


    def forward(self, x):
        x_shape = x.size()
        x = self.check_dim(x)
        z = self.encode(x)
        # z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        res = res.view(*x_shape)
        return [res, z, z]

class VAE_loss(nn.Module):
    def __init__(self):
        super(VAE_loss, self).__init__()
        # self.reconstruction_function = nn.BCELoss()
        # self.reconstruction_function.size_average = False
        # self.reconstruction_function = SorensenDiceLoss()
        self.reconstruction_function = nn.MSELoss()

    def forward(self, predictions, target):
        x = target[:,:,0]
        recon_x, mu, logvar = predictions
        recon_x = recon_x[:,:,0]
        BCE = self.reconstruction_function(recon_x, x)

        return BCE

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
