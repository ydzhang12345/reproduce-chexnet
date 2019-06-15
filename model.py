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
                model.train(True)
            else:
                model.train(False)
                model.eval()

            running_loss1 = 0.0
            running_loss2 = 0.0
            running_acc2 = 0.0
            i = 0
            total_acc = 0.0
            total_acc_raw = 0.0
            total_done = 0

            # iterate over all data in train/val dataloader:
            for data in dataloaders[phase]:
                i += 1
                inputs, label_disease, label_dataset, _ = data
                label_dataset = label_dataset.to(dtype=torch.int64)
                label_dataset = label_dataset.reshape(-1)
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.cuda())
                label_disease = Variable(label_disease.cuda()).float()
                label_dataset = Variable(label_dataset.cuda())

                if phase=='train':
                    pred_hex, pred_dataset, pred_disease = model.forward(inputs, phase)
                else:
                    with torch.no_grad():
                        pred_hex, pred_dataset, pred_disease = model.forward(inputs, phase)

                
                if phase=='train':

                    # calculate gradient and update parameters in train phase
                    optimizer.zero_grad()
                    loss1 = criterion1(pred_hex, label_disease)
                    loss2 = criterion2(pred_dataset, label_dataset)
                    loss = loss1 + (1 / (epoch + 1)) * loss2
                    #print(loss1, '***', loss2):
                    loss.backward()
                    optimizer.step()
                else:
                    optimizer.zero_grad()
                    loss1 = criterion1(pred_disease, label_disease)
                    loss2 = criterion2(pred_dataset, label_dataset)
                    loss = loss1 + (1 / (epoch + 1)) * loss2
                    probs = (torch.sigmoid(pred_disease)).cpu().data.numpy()
                    label_disease = label_disease.cpu().data.numpy()
                    total_acc += np.sum(np.uint8(probs>0.5)==label_disease)
                    probs_raw = (torch.sigmoid(pred_disease)).cpu().data.numpy()
                    total_acc_raw += np.sum(np.uint8(probs_raw>0.5)==label_disease)


                running_loss1 += loss1.data * batch_size
                running_loss2 += loss2.data * batch_size
                running_acc2 += torch.sum(pred_dataset.argmax(dim=1) == label_dataset)
            epoch_loss1 = running_loss1 / dataset_sizes[phase]
            epoch_loss2 = running_loss2 / dataset_sizes[phase]
            epoch_accuracy = running_acc2.to(dtype=torch.float32) / dataset_sizes[phase]
            if phase == 'train':
                last_train_loss = epoch_loss1
            else:
                print("total_acc: ", total_acc, "total_acc_raw: ", total_acc_raw)

            print(phase + ' epoch {}:loss1 {:.4f}, loss2 {:.4f}, acc {:.4f} with data size {}'.format(
                epoch, epoch_loss1, epoch_loss2, epoch_accuracy, dataset_sizes[phase]))

            # decay learning rate if no val loss improvement in this epoch
            if phase == 'val' and epoch_loss1 > best_loss:
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

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load('results/checkpoint')
    model = checkpoint_best['model']

    return model, best_epoch

class simpleCNN(torch.nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            ) # 112
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            ) # 56
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            ) # 28
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            ) # 14
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            ) # 7
        self.global_pool = nn.AvgPool2d(7, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x



class multi_output_model(torch.nn.Module):
    def __init__(self, model_core):
        super(multi_output_model, self).__init__()
        
        self.densenet_model = model_core
        self.simpleCNN = simpleCNN()
        self.num_dataset = 2

        #heads

        self.y1 = nn.Linear(32, self.num_dataset)
        nn.init.xavier_normal_(self.y1.weight)
        
        self.y2 = nn.Linear(1024 + 32, 5)
        nn.init.xavier_normal_(self.y2.weight)
        

    def forward(self, x, phase):

        # prepare feature
        diseases_feature = self.densenet_model(x)
        dataset_feature = self.simpleCNN(x)

        # l2-normalize as indicated in the paper
        dataset_feature = F.normalize(dataset_feature, p=2, dim=0)
        diseases_feature = F.normalize(diseases_feature, p=2, dim=0)

        ## start hex projection
        # prepare logits following hex paper and github
        y_dataset = self.y1(dataset_feature) # this gonna be supervised   N x 64 -> N x 2

        y_disease = self.y2(torch.cat([diseases_feature, (torch.zeros([dataset_feature.shape[0], 32]).cuda())], 1))
        y_padded = self.y2(torch.cat([(torch.zeros([dataset_feature.shape[0], 1024]).cuda()), dataset_feature], 1))
        y_all = self.y2(torch.cat([diseases_feature, dataset_feature], 1))

        # to project
        y_hex = y_all -  torch.mm(torch.mm(torch.mm(y_padded, torch.inverse(torch.mm(y_padded.t(), y_padded))), y_padded.t()), y_all)
        return y_hex, y_dataset, y_disease



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
    BATCH_SIZE = 32

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
    model_new = multi_output_model(model)
    model_new.cuda()    
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
            model_new.parameters()),
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
