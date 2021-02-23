import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import time
import learn2learn as l2l
from dataset import SiameseDataset, TripletDataset

# get train transforms:
def get_train_transforms():
    return T.Compose([T.Resize((224, 224)),
                      T.RandomHorizontalFlip(0.5),
                      T.RandomVerticalFlip(0.5),
                      T.RandomApply([T.RandomRotation(10)], 0.25),
                      T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
                      T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])


# get validation and test transforms
def get_val_transforms():
    return T.Compose([T.Resize((224, 224)),
                      T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])


# create SiameseNet dataloader given a root path
def get_siamese_dataloader(root=None, batch_size=1, transforms=None):
    dataset = SiameseDataset(root=root, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# create TripletNet dataloader given a root path
def get_triplet_dataloader(root=None, batch_size=1, transforms=None):
    dataset = TripletDataset(root=root, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


if __name__ == '__main__':
    siamese = get_siamese_dataloader(root='/home/lucas/Documents/pg/pg-testing/dataset/train/', transforms=get_train_transforms())
    triplet = get_triplet_dataloader(root='/home/lucas/Documents/pg/pg-testing/dataset/train/', transforms=get_val_transforms())

    batch = next(iter(siamese))
