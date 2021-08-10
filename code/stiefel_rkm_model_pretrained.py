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

        self.main = nn.Sequential(nn.Linear(self.args.num_ftrs, self.args.x_fdim1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Linear(self.args.x_fdim1, self.args.x_fdim2)
                                  )

    def forward(self, x):
        return self.main(x)


class Net3(nn.Module):
    """ Decoder - network architecture """
    def __init__(self, nChannels, args, cnn_kwargs):
        super(Net3, self).__init__()
        self.args = args
        self.main = nn.Sequential(
            nn.Linear(self.args.x_fdim2, self.args.x_fdim1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(self.args.x_fdim1),
            nn.Linear(self.args.x_fdim1, self.args.capacity * 4 * cnn_kwargs[2] ** 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(self.args.capacity * 4 * cnn_kwargs[2] ** 2),
            Lin_View(self.args.capacity * 4, cnn_kwargs[2], cnn_kwargs[2]),  # Unflatten

            nn.ConvTranspose2d(self.args.capacity * 4, self.args.capacity * 2, **cnn_kwargs[1]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(self.args.capacity * 2),
            nn.ConvTranspose2d(self.args.capacity * 2, self.args.capacity, **cnn_kwargs[0]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(self.args.capacity),

            nn.ConvTranspose2d(self.args.capacity, nChannels, **cnn_kwargs[0]),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class RKM_Stiefel_Transfer(nn.Module):
    """ Defines the Stiefel RKM model and its loss functions """
    def __init__(self, ipVec_dim, args, nChannels=1, recon_loss=nn.MSELoss(reduction='sum'), ngpus=1):
        super(RKM_Stiefel_Transfer, self).__init__()
        self.ipVec_dim = ipVec_dim
        self.ngpus = ngpus
        self.args = args
        self.nChannels = nChannels
        self.recon_loss = recon_loss

        # Initialize Manifold parameter (Initialized as transpose of U defined in paper)
        self.manifold_param = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.args.h_dim, self.args.x_fdim2)))

        # Settings for Conv layers
        self.cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)
        if self.ipVec_dim <= 28*28*3:
            self.cnn_kwargs = self.cnn_kwargs, dict(kernel_size=3, stride=1), 5
        else:
            if self.args.dataset_name == 'cifar10' or self.args.dataset_name == 'cifar10subset':
                self.cnn_kwargs = self.cnn_kwargs, self.cnn_kwargs, 4 # change this bad boy to shift the dataset size, 28 for 224x224, 8 for 64x64 and 4 for 32x32
            elif self.args.dataset_name == 'cifar10' or self.args.dataset_name == 'imagenette':
                self.cnn_kwargs = self.cnn_kwargs, self.cnn_kwargs, 8 # change this bad boy to shift the dataset size, 28 for 224x224, 8 for 64x64 and 4 for 32x32
            else:
                self.cnn_kwargs = self.cnn_kwargs, self.cnn_kwargs, 28 # change this bad boy to shift the dataset size, 28 for 224x224, 8 for 64x64 and 4 for 32x32

        self.encoder = Net1(self.nChannels, self.args, self.cnn_kwargs)
        self.decoder = Net3(self.nChannels, self.args, self.cnn_kwargs)

    def forward(self, x, x_image_view):
        op1 = self.encoder(x)  # features
        op1 = op1 - torch.mean(op1, dim=0)  # feature centering
        C = torch.mm(op1.t(), op1)  # Covariance matrix

        """ Various types of losses as described in paper """
        if self.args.loss == 'splitloss':
            x_tilde1 = self.decoder(torch.mm(torch.mm(op1, self.manifold_param.t())
                                            + self.args.noise_level * torch.randn((x_image_view.shape[0], self.args.h_dim)).to(self.args.proc),
                                            self.manifold_param))
            x_tilde2 = self.decoder(torch.mm(torch.mm(op1, self.manifold_param.t()), self.manifold_param))
            temp_view2 = x_image_view.view(-1, self.ipVec_dim)
            temp_view1 = x_tilde1.view(-1, self.ipVec_dim)
            temp_loss1 = self.recon_loss(temp_view1,  temp_view2)
            temp_loss2 = self.recon_loss(x_tilde2.view(-1, self.ipVec_dim), x_tilde1.view(-1, self.ipVec_dim))
            f2 = self.args.c_accu * 0.5 * (temp_loss1 + temp_loss2) / x_image_view.size(0)  # Recons_loss

        elif self.args.loss == 'noisyU':
            x_tilde = self.decoder(torch.mm(torch.mm(op1, self.manifold_param.t())
                                            + self.args.noise_level * torch.randn((x_image_view.shape[0], self.args.h_dim)).to(self.args.proc),
                                            self.manifold_param))
            f2 = self.args.c_accu * 0.5 * (
                self.recon_loss(x_tilde.view(-1, self.ipVec_dim), x_image_view.view(-1, self.ipVec_dim))) / x_image_view.size(0)  # Recons_loss

        elif self.args.loss == 'deterministic':
            x_tilde = self.decoder(torch.mm(op1, torch.mm(self.manifold_param.t(), self.manifold_param)))
            f2 = self.args.c_accu * 0.5 * (self.recon_loss(x_tilde.view(-1, self.ipVec_dim), x_image_view.view(-1, self.ipVec_dim)))/x.size(0)  # Recons_loss

        f1 = torch.trace(C - torch.mm(torch.mm(self.manifold_param.t(), self.manifold_param), C))/x_image_view.size(0)  # KPCA
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

def calculate_conv_output(input_size, cnn_kwargs):
    assert type(cnn_kwargs) == dict
    for cnn_kwarg_set in cnn_kwargs:
        return


def fetch_pretrained_model(model_name):
    if model_name == 'resnet152':
        # Freeze all parameters manually
        model = models.resnet152(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        return model
    if model_name == 'resnet50':
        # Freeze all parameters manually
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        return model

def final_compute(model, args, ct, device=torch.device('cuda')):
    """ Utility to re-compute U. Since some datasets could exceed the GPU memory limits, some intermediate
    variables are saved  on HDD, and retrieved later"""
    if not os.path.exists('oti/'):
        os.makedirs('oti/')

    args.shuffle = False
    x, _, _ = get_dataloader(args)
    x_transfer_features = get_transfer_features(args)
    
    # Compute feature-vectors
    for i, sample_batch in enumerate(tqdm(x)):
        indices = sample_batched[2]

        x_transfer_feature_batch = x_transfer_features.iloc[indices, :]
        x_transfer_feature_batch = torch.tensor(x_transfer_feature_batch.drop(['Image'], axis = 1).values.astype(np.float32))
                
        torch.save({'oti': model.encoder(x_transfer_feature_batch.to(device), sample_batched[0].to(device))},
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
