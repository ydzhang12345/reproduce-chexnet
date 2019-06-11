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
import eval_model as E

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))


def checkpoint(model, best_loss, epoch, LR):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR
    }

    torch.save(state, 'results/checkpoint')


def train_model(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay):
    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_acc = 0.0
            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            for data in dataloaders[phase]:
                i += 1
                inputs, labels, _ = data
                labels = labels.to(dtype=torch.int64)
                labels = labels.reshape(-1)
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data * batch_size
                running_acc += torch.sum(outputs.argmax(dim=1) == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = running_acc.to(dtype=torch.float32) / dataset_sizes[phase]
            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f}, acc {:.4f} with data size {}'.format(
                epoch, epoch_loss, epoch_accuracy, dataset_sizes[phase]))

            # decay learning rate if no val loss improvement in this epoch

            if phase == 'val' and epoch_loss > best_loss:
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 10) + " as not seeing improvement in val loss")
                LR = LR / 10
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=weight_decay)
                print("created new optimizer with LR " + str(LR))

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open("results/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])
        
        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")
        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            print("no improvement in 3 epochs, break")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load('results/checkpoint')
    model = checkpoint_best['model']

    return model, best_epoch


class multi_output_model(torch.nn.Module):
    def __init__(self, model_core, dropout_ratio):
        super(multi_output_model, self).__init__()
        
        self.densenet_model = model_core

        #https://blog.csdn.net/Geek_of_CSDN/article/details/90179421
        
        self.x1 =  nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)
        
        self.bn1 = nn.BatchNorm1d(256, eps = 2e-1)
        self.x2 =  nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x2.weight)
        self.bn2 = nn.BatchNorm1d(256, eps = 2e-1)
        #self.x3 =  nn.Linear(64,32)
        #nn.init.xavier_normal_(self.x3.weight)
        #comp head 1
        
        
        #heads
        self.y1o = nn.Linear(256,gender_nodes)
        nn.init.xavier_normal_(self.y1o.weight)#
        self.y2o = nn.Linear(256,region_nodes)
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(256,fighting_nodes)
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(256,alignment_nodes)
        nn.init.xavier_normal_(self.y4o.weight)
        self.y5o = nn.Linear(256,color_nodes)
        nn.init.xavier_normal_(self.y5o.weight)
        
        
        self.d_out = nn.Dropout(dd)
    def forward(self, x):
       
        x1 = self.resnet_model(x)
        #x1 =  F.relu(self.x1(x1))
        #x1 =  F.relu(self.x2(x1))
        
        x1 =  self.bn1(F.relu(self.x1(x1)))
        x1 =  self.bn2(F.relu(self.x2(x1)))
        #x = F.relu(self.x2(x))
        #x1 = F.relu(self.x3(x))
        
        # heads
        y1o = F.softmax(self.y1o(x1),dim=1)
        y2o = F.softmax(self.y2o(x1),dim=1)
        y3o = F.softmax(self.y3o(x1),dim=1)
        y4o = F.softmax(self.y4o(x1),dim=1)
        y5o = torch.sigmoid(self.y5o(x1)) #should be sigmoid
        
        #y1o = self.y1o(x1)
        #y2o = self.y2o(x1)
        #y3o = self.y3o(x1)
        #y4o = self.y4o(x1)
        #y5o = self.y5o(x1) #should be sigmoid
        
        return y1o, y2o, y3o, y4o, y5o


def densenet121_hex(input_data):
    model_ft = models.densenet121(pretrained=True)


    #num_ftrs = model.classifier.in_features
    # add final layer with # outputs in same dimension of labels with sigmoid
    # activation
    #model.classifier = nn.Sequential(
    #    nn.Linear(num_ftrs, N_LABELS))#, nn.Sigmoid())


    #num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs, 512)

    dd = .1
    model_1 = multi_output_model(model_ft,dd)
    model_1 = model_1.to(device)
    print(model_1)
    print(model_1.parameters())    
    criterion = [nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.BCELoss()]



def train_cnn(PATH_TO_IMAGES, LR, WEIGHT_DECAY):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    NUM_EPOCHS = 100
    BATCH_SIZE = 16

    try:
        rmtree('results/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs("results/")

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    N_LABELS = 2  # we are predicting 14 labels

    # load labels
    df = pd.read_csv("labels.csv", index_col=0)

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

    # create train/val dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='train',
        transform=data_transforms['train'])
    transformed_datasets['val'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='val',
        transform=data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")

    model_ft = models.densenet121(pretrained=True)
    model_1 = multi_output_model(model_ft, dropout_ratio=0.2)

    #model_1 = model_1.to(device)

    #pdb.set_trace()

    ## the last layer of densenet121 is of feature dim 1024
    # our target prediction is of size 5
    # dropout 1024 -> 1024
    # fc1 1024 -> 512 -> l2 normalize -> dropout -> l2 normalize  # 1024 -> 5, w, b
    # fc2 1024 -> 128 -> l2 normalize -> dropout -> l2 normalize -> # 128 -> 2
    # (128 + pad) -> 512, hex out -> 512
    # classifier 

    # 512 + 128 -> training

    # 
    # y_conv_loss = fc_{1}(1024 -> 512)
    # y_conv_H = fc_{2}(1024 -> 128 -> 2) -> train 128 features
    # [512, 128] -> predict
    # [zeros, 128] -> hex out



    num_ftrs = model.classifier.in_features
    # add final layer with # outputs in same dimension of labels with sigmoid
    # activation
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, N_LABELS))#, nn.Sigmoid())

    # put model on GPU
    model = model.cuda()

    # define criterion, optimizer for training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    # train model
    model, best_epoch = train_model(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY)

    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES)

    return preds, aucs
