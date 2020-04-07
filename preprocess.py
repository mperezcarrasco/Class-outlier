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
    # min, max values for the normal data, where the anormal class is the ith index of the list.
    min_max = [(-0.82804, 20.108057),
               (-0.8826562, 13.103283),
               (-0.8826562, 20.108057),
               (-0.8826562, 20.108057),
               (-0.8826562, 20.108057),
               (-0.8826562, 20.108057),
               (-0.8826562, 20.108057),
               (-0.8826562, 20.108057),
               (-0.8826562, 20.108057),
               (-0.8826562, 20.108057)]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: global_contrast_normalization(x)),
                                    transforms.Normalize([min_max[args.anormal_class][0]],
                                                         [min_max[args.anormal_class][1] \
                                                         -min_max[args.anormal_class][0]])])
    train = datasets.MNIST(root=data_dir, train=True, download=True)
    test = datasets.MNIST(root=data_dir, train=False, download=True)

    x_train = train.data
    y_train = train.targets

    x_train = x_train[np.where(y_train!=args.anormal_class)]
    y_train = y_train[np.where(y_train!=args.anormal_class)]
    
    N_train = int(x_train.shape[0]*0.8)
    
    x_val = x_train[N_train:]
    y1_val = y_train[N_train:]
    y2_val = np.where(y_train[N_train:]==args.anormal_class, 1, 0)
    
    data_val = MNIST_loader(x_val, y1_val, y2_val, transform)
    dataloader_val = DataLoader(data_val, batch_size=args.batch_size, 
                                  shuffle=False, num_workers=0)
    
    x_train = x_train[:N_train]
    y1_train = y_train[:N_train]
    y2_train = np.where(y_train[:N_train]==args.anormal_class, 1, 0)
                                    
    data_train = MNIST_loader(x_train, y1_train, y2_train, transform)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=0)
    
    x_test = test.data
    y1_test = test.targets
    y2_test = np.where(test.targets==args.anormal_class, 1, 0)
    data_test = MNIST_loader(x_test, y1_test, y2_test, transform)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=0)
    return dataloader_train, dataloader_val, dataloader_test