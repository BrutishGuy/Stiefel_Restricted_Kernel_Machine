import torch
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
import stiefel_optimizer
from dataloader import *
from utils import Lin_View


class Net1(nn.Module):
    """ Encoder - network architecture """
    def __init__(self, nChannels, args, cnn_kwargs):
        super(Net1, self).__init__()  # inheritance used here.
        self.args = args

        if self.args.use_transfer_learning:
            self.main = models.inception_v3(pretrained=True)
            self.num_ftrs = self.main.fc.in_features
            self.main.fc = nn.Linear(self.num_ftrs, self.args.x_fdim1)
            self.main = nn.Sequential

        else:
            self.main = nn.Sequential(
                nn.Conv2d(nChannels, self.args.capacity, **cnn_kwargs[0]),
                nn.LeakyReLU(negative_slope=0.2),

                nn.Conv2d(self.args.capacity, self.args.capacity * 2, **cnn_kwargs[0]),
                nn.LeakyReLU(negative_slope=0.2),

                nn.Conv2d(self.args.capacity * 2, self.args.capacity * 4, **cnn_kwargs[1]),
                nn.LeakyReLU(negative_slope=0.2),

                nn.Flatten(),
                nn.Linear(self.args.capacity * 4 * cnn_kwargs[2] ** 2, self.args.x_fdim1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(self.args.x_fdim1, self.args.x_fdim2),
            )

    def forward(self, x):
        return self.main(x)

class Net2(nn.Module):
    """ Encoder - network architecture for second view of multi-view setting """
    def __init__(self, args):
        super(Net2, self).__init__()
        self.args = args
        self.fc1 = torch.nn.Linear(40, self.args.x_fdim3)
        self.fc2 = torch.nn.Linear(self.args.x_fdim3, self.args.x_fdim4)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        return x

class Net3(nn.Module):0
    """ Decoder - network architecture """
    def __init__(self, nChannels, args, cnn_kwargs):
        super(Net3, self).__init__()
        self.args = args
        self.main = nn.Sequential(
            nn.Linear(self.args.x_fdim2, self.args.x_fdim1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.args.x_fdim1, self.args.capacity * 4 * cnn_kwargs[2] ** 2),
            nn.LeakyReLU(negative_slope=0.2),
            Lin_View(self.args.capacity * 4, cnn_kwargs[2], cnn_kwargs[2]),  # Unflatten

            nn.ConvTranspose2d(self.args.capacity * 4, self.args.capacity * 2, **cnn_kwargs[1]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(self.args.capacity * 2, self.args.capacity, **cnn_kwargs[0]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(self.args.capacity, nChannels, **cnn_kwargs[0]),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

class Net4(nn.Module):
    """ Decoder network for second view of multi-view setting """
    def __init__(self, args):
        super(Net4, self).__init__()
        self.args = args
        self.fc1 = torch.nn.Linear(self.args.x_fdim4, self.args.x_fdim3)
        self.fc2 = torch.nn.Linear(self.args.x_fdim3, 40)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = torch.tanh(self.fc2(x))
        return x

class RKM_Stiefel(nn.Module):
    """ Defines the Stiefel RKM model and its loss functions """
    def __init__(self, ipVec_dim, args, nChannels=1, recon_loss=nn.MSELoss(reduction='sum'), ngpus=1):
        super(RKM_Stiefel, self).__init__()
        self.ipVec_dim = ipVec_dim
        self.ngpus = ngpus
        self.args = args
        self.nChannels = nChannels
        self.recon_loss = recon_loss

        # Initialize Manifold parameter (Initialized as transpose of U defined in paper)
        self.manifold_param = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.args.h_dim, self.args.x_fdim2 + self.args.x_fdim4)))

        # Settings for Conv layers
        self.cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)
        if self.ipVec_dim <= 28*28*3:
            self.cnn_kwargs = self.cnn_kwargs, dict(kernel_size=3, stride=1), 5
        else:
            self.cnn_kwargs = self.cnn_kwargs, self.cnn_kwargs, 8

        self.encoder_view1 = Net1(self.nChannels, self.args, self.cnn_kwargs)
        self.decoder_view1 = Net3(self.nChannels, self.args, self.cnn_kwargs)

        self.encoder_view2 = Net2(self.nChannels, self.args)
        self.decoder_view2 = Net4(self.nChannels, self.args)

    def forward(self, x):
        op1 = self.encoder_view1(x)  # features
        op1 = op1 - torch.mean(op1, dim=0)  # feature centering
        C = torch.mm(op1.t(), op1)  # Covariance matrix

        """ Various types of losses as described in paper """
        if self.args.loss == 'splitloss':
            x_tilde1 = self.decoder(torch.mm(torch.mm(op1, self.manifold_param.t())
                                            + self.args.noise_level * torch.randn((x.shape[0], self.args.h_dim)).to(self.args.proc),
                                            self.manifold_param))
            x_tilde2 = self.decoder(torch.mm(torch.mm(op1, self.manifold_param.t()), self.manifold_param))
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
