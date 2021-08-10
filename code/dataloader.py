import os
import glob
import PIL
import scipy.io as sio
import h5py
import numpy as np
from sklearn.utils.extmath import cartesian
import torch
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler
from torchvision import datasets, transforms
import pandas as pd

from utils import Resize


    
def get_dataloader(args):
    print('Loading data...')
    if args.dataset_name == 'mnist':
        return get_mnist_dataloader(args=args)

    elif args.dataset_name == 'fashion-mnist':
        return get_fashion_mnist_dataloaders(args=args)

    elif args.dataset_name == 'svhn':
        return get_svhn_dataloader(args=args)

    elif args.dataset_name == 'cars3d':
        return get_cars3d_dataloader(args=args)

    elif args.dataset_name == '3dshapes':
        return get_3dshapes_dataloader(args=args)

    elif args.dataset_name == 'dsprites':
        return get_dsprites_dataloader(args=args)

    elif args.dataset_name == 'stl10':
        return get_stl10_dataloader(args=args)
    elif args.dataset_name == 'imagenette':
        return get_imagenette_dataloader(args=args)
    elif args.dataset_name == 'cifar10subset':
        return get_cifar10_subset_dataloader(args=args)
    elif args.dataset_name == 'galaxyzoo':
        return get_galaxyzoo_dataloader(args=args)
    
    

def get_transfer_features(args):
    print('Loading pre-trained features for transfer learning...')
    if args.dataset_name == 'imagenette':
        return get_imagenette_transfer_features(args=args)
    elif args.dataset_name == 'cifar10subset':
        return get_cifar10_subset_transfer_features(args=args)
    
def get_imagenette_transfer_features(args, path_to_data='./data/'):
    train_data = pd.read_csv(path_to_data + 'Transfer_Features__Resnet50_features_dataframe.csv').iloc[:, 1:]
    return train_data

def get_cifar10_subset_transfer_features(args, path_to_data='./data/'):
    train_data = pd.read_csv(path_to_data + 'Transfer_Features_CIFAR10_Resnet50_features_dataframe.csv').iloc[:, 1:]
    return train_data

def get_mnist_dataloader(args, path_to_data='mnist'):
    """MNIST dataloader with (28, 28) images."""

    all_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(path_to_data, train=True, download=True, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader, c*x*y, c


def get_fashion_mnist_dataloaders(args, path_to_data='fashion-mnist'):
    """FashionMNIST dataloader with (28, 28) images."""

    all_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.FashionMNIST(path_to_data, train=True, download=True, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader, c*x*y, c


def get_svhn_dataloader(args, path_to_data='svhn'):
    """SVHN dataloader with (28, 28) images."""

    all_transforms = transforms.Compose([transforms.Resize(28),
                                         transforms.ToTensor()])
    train_data = datasets.SVHN(path_to_data, split='train', download=True, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader, c*x*y, c


def get_cars3d_dataloader(args, path_to_data='cars3d'):
    """Cars3D dataloader with (64, 64, 3) images."""

    name = '{}/data/cars/'.format(path_to_data)
    if not os.path.exists(name):
        print('Data at the given path doesn\'t exist. Downloading now...')
        os.system(" mkdir cars3d/;"
                  " wget -O cars3d/nips2015-analogy-data.tar.gz http://www.scottreed.info/files/nips2015-analogy-data.tar.gz ;"
                  " cd cars3d/; tar xzf nips2015-analogy-data.tar.gz")

    all_transforms = transforms.Compose([transforms.ToTensor()])

    cars3d_data = cars3dDataset(path_to_data, transform=all_transforms)
    cars3d_loader = DataLoader(cars3d_data, batch_size=args.mb_size,
                                 shuffle=args.shuffle, pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(cars3d_loader))[0].size()
    return cars3d_loader, c*x*y, c

def get_stl10_dataloader(args, path_to_data='./data/stl10'):
    """STL10 dataloader with (64, 64, 3) images."""

    name = '{}/stl10_binary/'.format(path_to_data)
    if not os.path.exists(name):
        print('Data at the given path doesn\'t exist. Downloading now...')
        os.system(" mkdir stl10/;"
                  " wget -O stl10/stl10_binary.tar.gz http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz ;"
                  " cd stl10/; tar xzf stl10_binary.tar.gz")

    all_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    stl10_data = datasets.STL10('stl10', split='train', transform=all_transforms, download=True)
    if args.proc == 'cpu':
        stl10_loader = DataLoader(stl10_data, batch_size=args.mb_size,
                                     shuffle=args.shuffle, pin_memory=False, num_workers=0)
    else:
        stl10_loader = DataLoader(stl10_data, batch_size=args.mb_size,
                                     shuffle=args.shuffle, pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(stl10_loader))[0].size()
    return stl10_loader, c*x*y, c

def get_cifar10_subset_dataloader(args, path_to_data='./data/cifar10'):
    """ CIFAR10 loader with all the transforms needed for 64x64 sized images"""
    cifar10_dataset = CIFAR10Dataset(path_to_data)
    np.random.seed(42)
    
    dog_indices, cat_indices, bird_indices, horse_indices = [], [], [], []
    dog_idx, cat_idx, bird_idx, horse_idx = cifar10_dataset.cifar_images.class_to_idx['dog'], cifar10_dataset.cifar_images.class_to_idx['cat'], cifar10_dataset.cifar_images.class_to_idx['bird'], cifar10_dataset.cifar_images.class_to_idx['horse']

    for i in range(len(cifar10_dataset)):
        current_class = cifar10_dataset[i][1]
        if current_class == dog_idx:
          dog_indices.append(i)
        elif current_class == cat_idx:
          cat_indices.append(i)
        elif current_class == bird_idx:
          bird_indices.append(i)
        elif current_class == horse_idx:
          horse_indices.append(i)
          
    subset_indices = dog_indices + cat_indices + bird_indices + horse_indices
    subset_indices = torch.tensor(subset_indices)
    subsetting_choice = subset_indices[torch.randperm(len(subset_indices))[:5000]]


    cifar10_dataset_subset = SubsetRandomSampler(subsetting_choice)
    ### PyTorch data loaders ###
    if args.proc == 'cpu':
        cifar_loader = DataLoader(cifar10_dataset, args.mb_size, shuffle=False, num_workers=0, pin_memory=False, sampler=cifar10_dataset_subset)
    else:
        cifar_loader = DataLoader(cifar10_dataset, args.mb_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=cifar10_dataset_subset)
    _, c, x, y = next(iter(cifar_loader))[0].size()
    return cifar_loader, c*x*y, c
    
def get_imagenette_dataloader(args, use_sub_sample=True, path_to_data='./data/imagenette2'):
    """ Imagenette loader with all the transforms needed for 224 sized images"""
    name = '{}/imagenette2/train/'.format(path_to_data)
    if not os.path.exists(name):
        print('Data at the given path doesn\'t exist. Downloading now...')
        os.system(" mkdir imagenette2/;"
                  " wget -O imagenette2/imagenette2.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz ;"
                  " cd imagenette2/; tar xzf imagenette2.tgz")
    
    imagenette_dataset = ImagenetteDataset(path_to_data)
    np.random.seed(42)
    subsetting_choice = torch.randperm(len(imagenette_dataset))[:1000]
    imagenette_dataset_subset = SubsetRandomSampler(subsetting_choice)
    ### PyTorch data loaders ###
    if args.proc == 'cpu':
        imagenette_loader = DataLoader(imagenette_dataset, args.mb_size, shuffle=False, num_workers=0, pin_memory=False, sampler=imagenette_dataset_subset)
    else:
        imagenette_loader = DataLoader(imagenette_dataset, args.mb_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=imagenette_dataset_subset)
    _, c, x, y = next(iter(imagenette_loader))[0].size()
    return imagenette_loader, c*x*y, c

def get_galaxyzoo_dataloader(args, path_to_data='./data/'):
    """ Galaxy Zoo loader with all the transforms needed for 128 sized images"""

    galaxy_dataset = GalaxyZooDataset(path_to_data)
    np.random.seed(42)

    ### PyTorch data loaders ###
    if args.proc == 'cpu':
        galaxy_loader = DataLoader(galaxy_dataset, args.mb_size, shuffle=False, num_workers=0, pin_memory=False)
    else:
        galaxy_loader = DataLoader(galaxy_dataset, args.mb_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    _, c, x, y = next(iter(galaxy_loader))[0].size()
    return galaxy_loader, c*x*y, c


def get_dsprites_dataloader(args, path_to_data='dsprites'):
    """DSprites dataloader (64, 64) images"""

    name = '{}/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'.format(path_to_data)
    if not os.path.exists(name):
        print('Data at the given path doesn\'t exist. Downloading now...')
        os.system("  mkdir dsprites;"
                  "  wget -O dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

    transform = transforms.Compose([transforms.ToTensor()])

    dsprites_data = DSpritesDataset(name, transform=transform)
    dsprites_loader = DataLoader(dsprites_data, batch_size=args.mb_size,
                                 shuffle=args.shuffle, pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(dsprites_loader))[0].size()
    return dsprites_loader, c*x*y, c


def get_3dshapes_dataloader(args, path_to_data='3dshapes'):
    """3dshapes dataloader with images rescaled to (28,28,3)"""

    name = '{}/3dshapes.h5'.format(path_to_data)
    if not os.path.exists(name):
        print('Data at the given path doesn\'t exist. ')
        os.system("  mkdir 3dshapes;"
                  "  wget -O 3dshapes/3dshapes.h5 https://storage.googleapis.com/3d-shapes/3dshapes.h5")

    transform = transforms.Compose([Resize(28), transforms.ToTensor()])

    d3shapes_data = d3shapesDataset(name, transform=transform)
    d3shapes_loader = DataLoader(d3shapes_data, batch_size=args.mb_size,
                                 shuffle=args.shuffle, pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(d3shapes_loader))[0].size()
    return d3shapes_loader, c*x*y, c


class DSpritesDataset(Dataset):
    """DSprites dataloader class"""

    lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
    lat_sizes = np.array([3, 6, 40, 32, 32])

    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        dat = np.load(path_to_data)
        self.imgs = dat['imgs'][::subsample]
        self.lv = dat['latents_values'][::subsample]
        # self.lc = dat['latents_classes'][::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx] * 255
        sample = sample.reshape(sample.shape + (1,))

        if self.transform:
            sample = self.transform(sample)
        return sample, self.lv[idx]


class d3shapesDataset(Dataset):
    """3dshapes dataloader class"""

    lat_names = ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation')
    lat_sizes = np.array([10, 10, 10, 8, 4, 15])

    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        dataset = h5py.File(path_to_data, 'r')
        self.imgs = dataset['images'][::subsample]
        self.lat_val = dataset['labels'][::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx] / 255
        if self.transform:
            sample = self.transform(sample)
        return sample, self.lat_val[idx]


class cars3dDataset(Dataset):
    """Cars3D dataloader class

    The data set was first used in the paper "Deep Visual Analogy-Making"
    (https://papers.nips.cc/paper/5845-deep-visual-analogy-making) and can be
    downloaded from http://www.scottreed.info/. The images are rescaled to 64x64.

    The ground-truth factors of variation are:
    0 - elevation (4 different values)
    1 - azimuth (24 different values)
    2 - object type (183 different values)

    Reference: Code adapted from
    https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/cars3d.py
    """
    lat_names = ('elevation', 'azimuth', 'object_type')
    lat_sizes = np.array([4, 24, 183])

    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.imgs = self._load_data()[::subsample]
        self.lat_val = cartesian([np.array(list(range(i))) for i in self.lat_sizes])[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.transform:
            sample = self.transform(self.imgs[idx])
        return sample.float(), self.lat_val[idx]

    def _load_data(self):
        dataset = np.zeros((24 * 4 * 183, 64, 64, 3))
        all_files = glob.glob("cars3d/data/cars/*.mat")
        for i, filename in enumerate(all_files):
            data_mesh = self._load_mesh(filename)
            factor1 = np.array(list(range(4)))
            factor2 = np.array(list(range(24)))
            all_factors = np.transpose([
                np.tile(factor1, len(factor2)),
                np.repeat(factor2, len(factor1)),
                np.tile(i,
                        len(factor1) * len(factor2))
            ])
            dataset[np.arange(i, 24*4*183, 183)] = data_mesh
        return dataset

    def _load_mesh(self, filename):
        """Parses a single source file and rescales contained images."""
        mesh = np.einsum("abcde->deabc", sio.loadmat(filename)["im"])
        flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
        rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
        for i in range(flattened_mesh.shape[0]):
            pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
            pic.thumbnail((64, 64), PIL.Image.ANTIALIAS)
            rescaled_mesh[i, :, :, :] = np.array(pic)
        return rescaled_mesh * 1. / 255


class ImagenetteDataset(Dataset):
    def __init__(self, path_to_data):
        self.all_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )])
        self.imagenette_images = datasets.ImageFolder(path_to_data + '/imagenette2/train/', self.all_transforms)  
    def __getitem__(self, index):
        data, target = self.imagenette_images[index]        
        # Your transformations here (or set it in ImageFolder class instantiation) 
        return data, target, index
    def __len__(self):
        return len(self.imagenette_images)

class GalaxyZooDataset(Dataset):
    def __init__(self, path_to_data):
        self.all_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(128),
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])
        self.galaxy_images = datasets.ImageFolder(path_to_data + '/galaxy_zoo_images/', self.all_transforms)  
    def __getitem__(self, index):
        data, target = self.galaxy_images[index]        
        # Your transformations here (or set it in ImageFolder class instantiation) 
        return data, target, index
    def __len__(self):
        return len(self.galaxy_images)
    
class CIFAR10Dataset(Dataset):
    def __init__(self, path_to_data):
        self.all_transforms = transforms.Compose([
        transforms.CenterCrop(32),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
        )])
        self.cifar_images = datasets.CIFAR10(root=path_to_data + '/cifar10/train/', train=True , download=True, transform=self.all_transforms ) 
    def __getitem__(self, index):
        data, target = self.cifar_images[index]        
        # Your transformations here (or set it in ImageFolder class instantiation) 
        return data, target, index
    def __len__(self):
        return len(self.cifar_images)
   