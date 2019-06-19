from __future__ import print_function, division

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F

# image imports
from skimage import io, transform
from PIL import Image

# general imports
import os
import time
from shutil import copyfile
from shutil import rmtree

# data science imports
import pandas as pd
import numpy as np
import csv
import pdb

import cxr_dataset as CXR

## load model
PATH_TO_IMAGES = 'starter_images/'
path_model = 'results/checkpoint11'

checkpoint = torch.load(path_model, map_location=lambda storage, loc: storage)
model = checkpoint['model']
del checkpoint
model.cuda()

# build dataloader on test
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# define torchvision transforms
data_transforms = {
    'val': transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

BATCH_SIZE = 50
dataset = CXR.CXRDataset(
    path_to_images=PATH_TO_IMAGES,
    fold="test",
    transform=data_transforms['val'])
dataloader = torch.utils.data.DataLoader(
    dataset, BATCH_SIZE, shuffle=True, num_workers=8)


# first extract features: trained model before proj and after proj
feature_bank = []
with torch.no_grad():
	model.eval()
	for i, data in enumerate(dataloader):
		inputs, label_disease, label_dataset, _ = data
		label_dataset = label_dataset.to(dtype=torch.int64)
		label_dataset = label_dataset.reshape(-1)
		batch_size = inputs.shape[0]
		inputs = Variable(inputs.cuda())
		label_disease = Variable(label_disease.cuda()).float()
		label_dataset = Variable(label_dataset.cuda())

		common_feature = model.densenet_model(inputs)
		dataset_feature = (model.x1(common_feature)).cpu().data.numpy()
		disease_feature = (model.x2(common_feature)).cpu().data.numpy()

		








