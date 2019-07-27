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


class DualHex_model(torch.nn.Module):
    def __init__(self, model_core, dropout_ratio, num_dataset=2, num_disease=5):
        super(DualHex_model, self).__init__()
        
        self.common_feature = model_core
        self.num_dataset = num_dataset
        self.num_disease = num_disease
        
        self.x1 =  nn.Linear(9216, 32)
        nn.init.xavier_normal_(self.x1.weight)
        #self.bn1 = nn.BatchNorm1d(64,eps = 2e-1)
        
        self.x2 =  nn.Linear(9216, 4096)
        nn.init.xavier_normal_(self.x2.weight)
        #self.bn2 = nn.BatchNorm1d(4096, eps = 2e-1)

        self.x3 =  nn.Linear(4096, 4096)
        nn.init.xavier_normal_(self.x3.weight)

        #heads
        self.y1 = nn.Linear(4096 + 32, self.num_dataset)
        nn.init.xavier_normal_(self.y1.weight)

        self.y2 = nn.Linear(4096 + 32, self.num_disease)
        nn.init.xavier_normal_(self.y2.weight)
        
        self.d_out = nn.Dropout(dropout_ratio)

    def forward(self, x, phase):

        # prepare feature
        common_feature = self.common_feature(x)
        common_feature = self.d_out(common_feature)

        dataset_feature =  F.leaky_relu(self.x1(common_feature)) 
        dataset_feature = F.normalize(dataset_feature, p=2, dim=0)

        diseases_feature = F.leaky_relu(self.x3(F.leaky_relu(self.x2(common_feature)))) 
        diseases_feature = F.normalize(diseases_feature, p=2, dim=0)

        ## start hex projection 1st path
        # prepare logits following hex paper and github
        y_concat1 = self.y2(torch.cat([diseases_feature, dataset_feature], 1)) # N x 1056 -> N x 5
        y_pad_dataset = self.y2(torch.cat([torch.zeros_like(diseases_feature), dataset_feature], 1))
        y_disease_pad = self.y2(torch.cat([diseases_feature, torch.zeros_like(dataset_feature)], 1))
        y_hex1 = y_concat1 - torch.mm(torch.mm(torch.mm(y_pad_dataset, torch.inverse(torch.mm(y_pad_dataset.t(), y_pad_dataset))), y_pad_dataset.t()), y_concat1)

        ## start hex projection 2nd path
        # prepare logits following hex paper and github
        y_concat2 = self.y1(torch.cat([dataset_feature, diseases_feature], 1)) # N x 1056 -> N x 5
        y_pad_diseases = self.y1(torch.cat([torch.zeros_like(dataset_feature), diseases_feature], 1))
        y_dataset_pad = self.y1(torch.cat([dataset_feature, torch.zeros_like(diseases_feature)], 1))
        y_hex2 = y_concat2 - torch.mm(torch.mm(torch.mm(y_pad_diseases, torch.inverse(torch.mm(y_pad_diseases.t(), y_pad_diseases))), y_pad_diseases.t()), y_concat2)
        
        return y_hex1, y_hex2, y_disease_pad



class Hex_model(torch.nn.Module):
    def __init__(self, model_core, dropout_ratio, num_dataset=2, num_disease=5):
        super(multi_output_model, self).__init__()
        
        self.common_feature = model_core
        self.num_dataset = num_dataset
        self.num_disease = num_disease
        
        self.x1 =  nn.Linear(9216, 32)
        nn.init.xavier_normal_(self.x1.weight)
        #self.bn1 = nn.BatchNorm1d(64,eps = 2e-1)
        
        self.x2 =  nn.Linear(9216, 4096)
        nn.init.xavier_normal_(self.x2.weight)
        #self.bn2 = nn.BatchNorm1d(512, eps = 2e-1)

        self.x3 = nn.Linear(4096, 4096)
        nn.init.xavier_normal_(self.x3.weight)

        #heads
        self.y1 = nn.Linear(32, self.num_dataset)
        nn.init.xavier_normal_(self.y1.weight)

        self.y2 = nn.Linear(4096 + 32, 5)
        nn.init.xavier_normal_(self.y2.weight)
        
        self.d_out = nn.Dropout(dropout_ratio)

    def forward(self, x, phase):

        # prepare feature
        # l2-normalize as indicated in the paper
        common_feature = self.common_feature(x)
        common_feature = self.d_out(common_feature)
        # add dropout 

        dataset_feature =  F.relu(self.x1(common_feature)) # of 32
        dataset_feature = F.normalize(dataset_feature, p=2, dim=0)

        diseases_feature = F.relu(self.x3(F.relu(self.x2(common_feature)))) # of 1024
        diseases_feature = F.normalize(diseases_feature, p=2, dim=0)

        ## start hex projection
        # prepare logits following hex paper and github
        y_dataset = self.y1(dataset_feature) # this gonna be supervised   N x 32 -> N x 2
        y_disease = self.y2(torch.cat([diseases_feature, dataset_feature], 1)) # N x 1056 -> N x 5
        y_padded = self.y2(torch.cat([torch.zeros_like(diseases_feature), dataset_feature], 1))
        y_raw = self.y2(torch.cat([diseases_feature, torch.zeros_like(dataset_feature)], 1))

        # to project
        y_hex = y_disease -  torch.mm(torch.mm(torch.mm(y_padded, torch.inverse(torch.mm(y_padded.t(), y_padded))), y_padded.t()), y_disease)
        
        return y_hex, y_dataset, y_raw



## aslo tested: simpleGating, proj in a small space;

