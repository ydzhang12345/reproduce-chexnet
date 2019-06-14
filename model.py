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
        criterion1,
        criterion2,
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
                continue
                model.train(True)
            else:
                model.train(False)
                model.eval()

            running_loss1 = 0.0
            running_loss2 = 0.0
            running_acc2 = 0.0
            i = 0
            total_done = 0

            # iterate over all data in train/val dataloader:
            for data in dataloaders[phase]:
                i += 1
                inputs, label_disease, label_dataset, _ = data
                #label_disease = label_disease.to(dtype=torch.int64)
                label_dataset = label_dataset.to(dtype=torch.int64)
                #label_disease = label_disease.reshape(-1)
                label_dataset = label_dataset.reshape(-1)
                #labels = labels.reshape(-1)
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.cuda())
                label_disease = Variable(label_disease.cuda()).float()
                label_dataset = Variable(label_dataset.cuda())
                #labels = Variable(labels.cuda())
                pred_disease, pred_dataset = model.forward(inputs, phase)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                loss1 = criterion1(pred_disease, label_disease)
                loss2 = criterion2(pred_dataset, label_dataset)
                pdb.set_trace()
                #print(loss1, "*****", loss2)
                loss = loss1 + 0.2 * loss2
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss1 += loss1.data * batch_size
                running_loss2 += loss2.data * batch_size
                running_acc2 += torch.sum(pred_dataset.argmax(dim=1) == label_dataset)
                #break

            epoch_loss1 = running_loss1 / dataset_sizes[phase]
            epoch_loss2 = running_loss2 / dataset_sizes[phase]
            epoch_accuracy = running_acc2.to(dtype=torch.float32) / dataset_sizes[phase]
            if phase == 'train':
                last_train_loss = epoch_loss1

            print(phase + ' epoch {}:loss1 {:.4f}, loss2 {:.4f}, acc {:.4f} with data size {}'.format(
                epoch, epoch_loss1, epoch_loss2, epoch_accuracy, dataset_sizes[phase]))

            # decay learning rate if no val loss improvement in this epoch
            if phase == 'val' and epoch_loss1 > best_loss:
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 2) + " as not seeing improvement in val loss")
                LR = LR / 2
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
            if phase == 'val' and epoch_loss1 < best_loss:
                best_loss = epoch_loss1
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open("results/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss1])
        
        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")
        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            print("no improvement in 3 epochs, break")
            break
        #break

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
        self.num_dataset = 2

        #https://blog.csdn.net/Geek_of_CSDN/article/details/90179421
        
        self.x1 =  nn.Linear(1024, 32)
        nn.init.xavier_normal_(self.x1.weight)
        #self.bn1 = nn.BatchNorm1d(64,eps = 2e-1)
        
        #self.x2 =  nn.Linear(1024, 992)
        #nn.init.xavier_normal_(self.x2.weight)
        #self.bn2 = nn.BatchNorm1d(512, eps = 2e-1)

        #heads
        self.y1 = nn.Linear(32, self.num_dataset)
        nn.init.xavier_normal_(self.y1.weight)

        self.y2 = nn.Linear(1024, 5)
        nn.init.xavier_normal_(self.y2.weight)
        
        self.d_out = nn.Dropout(dropout_ratio)

    def forward(self, x, phase):

        # prepare feature
        # l2-normalize as indicated in the paper
        common_feature = self.densenet_model(x)
        #common_feature = self.d_out(common_feature)
        # add dropout 

        #pdb.set_trace()
        dataset_feature =  F.relu(self.x1(common_feature)) # of 32
        #dataset_feature = F.normalize(dataset_feature, p=2, dim=0)

        #diseases_feature = F.relu(self.x2(common_feature)) # of 512
        #diseases_feature = F.normalize(diseases_feature, p=2, dim=0)

        ## start hex projection
        # prepare logits following hex paper and github

        y_dataset = self.y1(dataset_feature) # this gonna be supervised   N x 64 -> N x 2
        '''
        if phase=='train':
            # in training, we concat dataset_feature, in testing, we pad zero
            y_disease = self.y2(torch.cat([diseases_feature, dataset_feature], 1)) # N x 576 -> N x 5
        else:
            y_disease = self.y2(torch.cat([diseases_feature, torch.zeros_like(dataset_feature)], 1)) # N x 576 -> N x 5
        '''
        y_disease = self.y2(common_feature)
        y_padded = self.y2(torch.cat([(torch.zeros([dataset_feature.shape[0], 1024-32]).cuda()), dataset_feature], 1))

        # to project
        y_hex = y_disease -  torch.mm(torch.mm(torch.mm(y_padded, torch.inverse(torch.mm(y_padded.t(), y_padded))), y_padded.t()), y_disease)
        return y_hex, y_dataset


'''
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
'''



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

    '''
    try:
        rmtree('results/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs("results/")
    '''

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

    
    model = models.densenet121(pretrained=True)
    del model.classifier
    model.classifier = nn.Identity()
    model_new = multi_output_model(model, dropout_ratio=0.2)
    
    '''
    path_images = '/home/lovebb/Documents/MIBLab/chest-Xray-dataset'
    path_model = '/home/lovebb/Documents/MIBLab/undo_bias/reproduce-chexnet/results/checkpoint'

    checkpoint = torch.load(path_model, map_location=lambda storage, loc: storage)
    model_new = checkpoint['model']
    del checkpoint
    model_new.cuda()
    for name, param in model_new.named_parameters():
        if param.requires_grad:
            print(name, ' ')
    pdb.set_trace()
    '''

    # define criterion, optimizer for training
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    # train model
    model, best_epoch = train_model(model_new, criterion1, criterion2, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY)


    '''
    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES)
    '''
    return preds, aucs
