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
import pickle

import cxr_dataset as CXR

## load model
PATH_TO_IMAGES = '/home/ben/Desktop/MIBLab/'
path_model = 'results/checkpoint19'

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

BATCH_SIZE = 128
dataset = CXR.CXRDataset(
    path_to_images=PATH_TO_IMAGES,
    fold="test",
    transform=data_transforms['val'])
dataloader = torch.utils.data.DataLoader(
    dataset, BATCH_SIZE, shuffle=True, num_workers=8)


# first extract features: trained model before proj and after proj
hex_flag = False


disease_feature_bank = []
dataset_feature_bank = []
label_disease_bank = []
label_dataset_bank = []
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
        model.class_classifier.c_fc3 = nn.Identity()
        model.domain_classifier.d_fc2 = nn.Identity()
        disease_feature = (model.class_classifier(common_feature)).cpu().data.numpy()
        dataset_feature = (model.domain_classifier(common_feature)).cpu().data.numpy()

        label_disease = label_disease.cpu().data.numpy()
        label_dataset = label_dataset.cpu().data.numpy()
        
        disease_feature_bank.append(disease_feature)
        dataset_feature_bank.append(dataset_feature)
        label_disease_bank.append(label_disease)
        label_dataset_bank.append(label_dataset)
disease_feature = np.concatenate(disease_feature_bank, axis=0)
dataset_feature = np.concatenate(dataset_feature_bank, axis=0)
label_disease = np.concatenate(label_disease_bank, axis=0)
label_dataset = np.concatenate(label_dataset_bank, axis=0)
plk_dict = {"x_disease": disease_feature, 'x_dataset': dataset_feature, 'y_disease': label_disease, 'y_dataset': label_dataset}


'''
disease_feature_bank = []
label_disease_bank = []
label_dataset_bank = []

with torch.no_grad():
    model.classifier = torch.nn.Identity()
    model.eval()
    for i, data in enumerate(dataloader):
        inputs, label_disease, label_dataset, _ = data
        label_dataset = label_dataset.to(dtype=torch.int64)
        label_dataset = label_dataset.reshape(-1)
        batch_size = inputs.shape[0]
        inputs = Variable(inputs.cuda())
        label_disease = Variable(label_disease.cuda()).float()
        label_dataset = Variable(label_dataset.cuda())

        #pdb.set_trace()
        common_feature = model(inputs)
        disease_feature = common_feature.cpu().data.numpy()
        label_disease = label_disease.cpu().data.numpy()
        label_dataset = label_dataset.cpu().data.numpy()
        
        disease_feature_bank.append(disease_feature)
        label_disease_bank.append(label_disease)
        label_dataset_bank.append(label_dataset)

disease_feature = np.concatenate(disease_feature_bank, axis=0)
label_disease = np.concatenate(label_disease_bank, axis=0)
label_dataset = np.concatenate(label_dataset_bank, axis=0)

plk_dict = {"x_disease": disease_feature, 'y_disease': label_disease, 'y_dataset': label_dataset}
'''
print ('feature extraction done!')

with open('extracted_feature_dann_ckpt19.pkl', 'wb') as f:
    pickle.dump(plk_dict, f)