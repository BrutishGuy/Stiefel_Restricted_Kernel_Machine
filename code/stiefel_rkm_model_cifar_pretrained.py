import stiefel_optimizer
from dataloader import *
from utils import Lin_View

import os
import json
import math
import numpy as np
import argparse

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()
sns.set()

## Progress bar
from tqdm import tqdm
#from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.models as models

from torchvision import transforms
# PyTorch Lightning
import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Tensorboard extension (for visualization purposes later)
from torch.utils.tensorboard import SummaryWriter


class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*16*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x
    
    
class RKM_Stiefel_Transfer(nn.Module):
    """ Defines the Stiefel RKM model and its loss functions """
    def __init__(self, 
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 num_input_channels: int = 3, 
                 recon_loss=nn.MSELoss(reduction='sum'), 
                 ngpus: int = 1,
                 width: int = 32,
                 height: int = 32):
        super(RKM_Stiefel_Transfer, self).__init__()
        self.ipVec_dim = num_input_channels * width * height
        self.ngpus = ngpus
        self.args = argparse.Namespace()
        self.args.capacity = base_channel_size
        self.args.x_fdim1 = latent_dim
        self.args.x_fdim2 = latent_dim//2
        self.args.loss= 'splitloss'
        self.args.h_dim= 64
        self.args.noise_level= 1e-3
        self.args.proc= 'cuda'
        self.args.c_accu= 1
        self.args.shuffle= False
        self.num_input_channels = num_input_channels
        self.nChannels = num_input_channels
        self.recon_loss = recon_loss
        self.width = width
        self.height = height
        self.latent_dim = latent_dim
        self.base_channel_size = base_channel_size
        
        
        #self.save_hyperparameters()
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        
        self.encoder_extra_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim//2),
        )
        self.decoder_extra_layer = nn.Sequential(
            nn.Linear(latent_dim//2, latent_dim),
            nn.GELU()
        )
        
    def _set_args(self, args):
        self.args = args
        self.manifold_param = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.args.h_dim, self.args.x_fdim2)))

    def _reset_manifold_param(self):
        # Initialize Manifold parameter (Initialized as transpose of U defined in paper)
        self.manifold_param = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.args.h_dim, self.args.x_fdim2)))

    def _freeze_encoder_weights(self):
        for c in self.encoder.children():
            for params in c.parameters():
                params.requires_grad = False
    
    def _freeze_decoder_weights(self):
        for c in self.decoder.children():
            for params in c.parameters():
                params.requires_grad = False
        return
    
    def forward(self, x):
        op1 = self.encoder(x)  # features
        op1 = self.encoder_extra_layer(op1)
        op1 = op1 - torch.mean(op1, dim=0)  # feature centering
        C = torch.mm(op1.t(), op1)  # Covariance matrix

        """ Various types of losses as described in paper """
        if self.args.loss == 'splitloss':
            x_tilde1 = self.decoder(self.decoder_extra_layer(torch.mm(torch.mm(op1, self.manifold_param.t())
                                            + self.args.noise_level * torch.randn((x.shape[0], self.args.h_dim)).to(self.args.proc),
                                            self.manifold_param)))
            x_tilde2 = self.decoder(self.decoder_extra_layer(torch.mm(torch.mm(op1, self.manifold_param.t()), self.manifold_param)))
            f2 = self.args.c_accu * 0.5 * (
                    self.recon_loss(x_tilde2.view(-1, self.ipVec_dim), x.view(-1, self.ipVec_dim))
                    + self.recon_loss(x_tilde2.view(-1, self.ipVec_dim),
                                      x_tilde1.view(-1, self.ipVec_dim))) / x.size(0)  # Recons_loss

        elif self.args.loss == 'noisyU':
            x_tilde = self.decoder(torch.mm(torch.mm(op1, self.manifold_param.t())
                                            + self.args.noise_level * torch.randn((x.shape[0], self.args.h_dim)).to(self.args.proc),
                                            self.manifold_param))
            f2 = self.args.c_accu * 0.5 * (
                self.recon_loss(x_tilde.view(-1, self.ipVec_dim), x.view(-1, self.ipVec_dim))) / x.size(0)  # Recons_loss
        elif self.args.loss == 'deterministic':
            x_tilde = self.decoder(torch.mm(op1, torch.mm(self.manifold_param.t(), self.manifold_param)))
            f2 = self.args.c_accu * 0.5 * (self.recon_loss(x_tilde.view(-1, self.ipVec_dim), x.view(-1, self.ipVec_dim)))/x.size(0)  # Recons_loss

        f1 = torch.trace(C - torch.mm(torch.mm(self.manifold_param.t(), self.manifold_param), C))/x.size(0)  # KPCA
        return f1 + f2, f1, f2

# Accumulate trainable parameters in 2 groups:
# 1. Manifold_params 2. Network param
def param_state(model):
    param_g, param_e1 = [], []
    for name, param in model.named_parameters():
        if param.requires_grad and name != 'manifold_param':
            param_e1.append(param)
        elif name == 'manifold_param':
            param_g.append(param)
    return param_g, param_e1

def stiefel_opti(stief_param, lrg=1e-4):
    dict_g = {'params': stief_param, 'lr': lrg, 'momentum': 0.9, 'weight_decay': 0.0005, 'stiefel': True}
    return stiefel_optimizer.AdamG([dict_g])  # CayleyAdam

def final_compute(model, args, ct, device=torch.device('cuda')):
    """ Utility to re-compute U. Since some datasets could exceed the GPU memory limits, some intermediate
    variables are saved  on HDD, and retrieved later"""
    if not os.path.exists('oti/'):
        os.makedirs('oti/')

    args.shuffle = False
    x, _, _ = get_dataloader(args)

    # Compute feature-vectors
    for i, sample_batch in enumerate(tqdm(x)):
        torch.save({'oti': model.encoder(sample_batch[0].to(device))},
                   'oti/oti{}_checkpoint.pth_{}.tar'.format(i, ct))

    # Load feature-vectors
    ot = torch.Tensor([]).to(device)
    for i in range(0, len(x)):
        ot = torch.cat((ot, torch.load('oti/oti{}_checkpoint.pth_{}.tar'.format(i, ct))['oti']), dim=0)
    os.system("rm -rf oti/")

    ot = (ot - torch.mean(ot, dim=0)).to(device)  # Centering
    u, _, _ = torch.svd(torch.mm(ot.t(), ot))
    u = u[:, :args.h_dim]
    with torch.no_grad():
        model.manifold_param.masked_scatter_(model.manifold_param != u.t(), u.t())
    return torch.mm(ot, u.to(device)), u
