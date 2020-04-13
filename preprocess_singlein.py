# we used the precomputed min_max values from the original implementation: 
# https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/1901612d595e23675fb75c4ebb563dd0ffebc21e/src/datasets/mnist.py

import torch
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from PIL import Image


def global_contrast_normalization(x):
    """Apply global contrast normalization to tensor. """
    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean
    x_scale = torch.mean(torch.abs(x))
    x /= x_scale
    return x


class MNIST_loader(data.Dataset):
    """This class is needed to processing batches for the dataloader."""
    def __init__(self, data, target1, target2, transform):
        self.data = data
        self.target1 = target1
        self.target2 = target2
        self.transform = transform

    def __getitem__(self, index):
        """return transformed items."""
        x = self.data[index]
        y1 = self.target1[index]
        y2 = self.target2[index]
        if self.transform:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y1, y2

    def __len__(self):
        """number of samples."""
        return len(self.data)


def get_mnist(args, data_dir='./data/mnist/'):
    """get dataloders"""
    # min, max values for each class after applying GCN (as the original implementation)
    min_max = [(-0.8826567065619495, 9.001545489292527),
                (-0.6661464580883915, 20.108062262467364),
                (-0.7820454743183202, 11.665100841080346),
                (-0.7645772083211267, 12.895051191467457),
                (-0.7253923114302238, 12.683235701611533),
                (-0.7698501867861425, 13.103278415430502),
                (-0.778418217980696, 10.457837397569108),
                (-0.7129780970522351, 12.057777597673047),
                (-0.8280402650205075, 10.581538445782988),
                (-0.7369959242164307, 10.697039838804978)]
    normal_class = args.anormal_class
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: global_contrast_normalization(x)),
                                    transforms.Normalize([min_max[normal_class][0]],
                                                         [min_max[normal_class][1] \
                                                         -min_max[normal_class][0]])])
    
    train = datasets.MNIST(root=data_dir, train=True, download=True)
    test = datasets.MNIST(root=data_dir, train=False, download=True)

    x_train = train.data
    y_train = train.targets

    x_train = x_train[np.where(y_train==normal_class)]
    y_train = y_train[np.where(y_train==normal_class)]
    
    N_train = int(x_train.shape[0]*0.8)

    x_val = x_train[N_train:]
    y1_val = y_train[N_train:]
    y2_val = np.where(y_train[N_train:]==normal_class, 0, 1)
    
    data_val = MNIST_loader(x_val, y1_val, y2_val, transform)
    dataloader_val = DataLoader(data_val, batch_size=args.batch_size, 
                                  shuffle=False, num_workers=0)
    x_train = x_train[:N_train]
    y1_train = y_train[:N_train]
    y2_train = np.where(y_train[:N_train]==normal_class, 0, 1)
    
    data_train = MNIST_loader(x_train, y1_train, y2_train, transform)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=0)
    
    x_test = test.data
    y1_test = test.targets
    y2_test = np.where(y1_test==normal_class, 0, 1)
    data_test = MNIST_loader(x_test, y1_test, y2_test, transform)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=0)
    return dataloader_train, dataloader_val, dataloader_test