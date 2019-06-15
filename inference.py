from __future__ import print_function, division

# pytorch imports
import torch
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms, utils

# image / graphics imports
from skimage import io, transform
from PIL import Image
from pylab import *
import seaborn as sns
from matplotlib.pyplot import show 

# data science
import numpy as np
import scipy as sp
import pandas as pd


# import other modules
from copy import deepcopy
import cxr_dataset as CXR
import eval_model as E
import pdb


path_images = '/home/ben/Desktop/MIBLab/'
path_model = '/home/ben/Desktop/MIBLab/hospital-cls/reproduce-chexnet/results/checkpoint10'

checkpoint = torch.load(path_model, map_location=lambda storage, loc: storage)
model = checkpoint['model']
del checkpoint
#model.cuda()


# build dataloader on test
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# define torchvision transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(256),

        # because scale doesn't always give 224 x 224, this ensures 224 x
        # 224
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

'''
dataset = CXR.CXRDataset(
    path_to_images=PATH_TO_IMAGES,
    fold='test',
    transform=data_transform)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=1)
'''

preds, aucs = E.make_pred_multilabel(
    data_transforms, model, path_images)
