import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SetData():

    def __init__(self, dataroot='../../../hymenoptera_data'):
        self.transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        self.dataroot = dataroot

        self.datasets = {'train': datasets.CIFAR10(root='../data',
                                                   train=True,
                                                   transform=self.transform['train'],
                                                   download=True),
                         'val': datasets.CIFAR10(root='../data',
                                                   train=False,
                                                   transform=self.transform['val'],
                                                   download=True)}

        self.dataloaders = {'train': DataLoader(dataset=self.datasets['train'],
                                                batch_size=100,
                                                shuffle=True,
                                                num_workers=4),
                            'val': DataLoader(dataset=self.datasets['val'],
                                               batch_size=100,
                                               shuffle=False,
                                               num_workers=4)}


def get_ds_loaders():
    d = SetData()
    return d.datasets, d.dataloaders