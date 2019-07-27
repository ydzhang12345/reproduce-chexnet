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

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(2018)


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

    torch.save(state, 'results/checkpoint' + str(epoch))


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
                #continue
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
                    pred_disease, pred_dataset, pred_raw = model.forward(inputs, phase)
                else:
                    with torch.no_grad():
                        pred_disease, pred_dataset, pred_raw = model.forward(inputs, phase)


                if phase == 'train':
                    # calculate gradient and update parameters in train phase
                    optimizer.zero_grad()
                    loss1 = criterion1(pred_disease, label_disease)
                    loss2 = criterion2(pred_dataset, label_dataset)
                    print(loss1, "*****", loss2)
                    loss = loss1 + loss2 #epoch / num_epochs * loss2
                    loss.backward()
                    optimizer.step()
                else:
                    optimizer.zero_grad()
                    loss1 = criterion1(pred_disease, label_disease)
                    loss2 = criterion2(pred_dataset, label_dataset)
                    #print(loss1, "*****", loss2)
                    loss = loss1 + loss2

                    #probs = (torch.sigmoid(pred_disease)).cpu().data.numpy()
                    probs = (torch.sigmoid(pred_raw)).cpu().data.numpy()
                    label_disease = label_disease.cpu().data.numpy()
                    total_acc += np.sum(np.uint8(probs>0.5)==label_disease)

                    probs_raw = (torch.sigmoid(pred_raw)).cpu().data.numpy()
                    total_acc_raw += np.sum(np.uint8(probs_raw>0.5)==label_disease)


                running_loss1 += loss1.data * batch_size
                running_loss2 += loss2.data * batch_size
                temp = (torch.sigmoid(pred_dataset)).cpu().data.numpy()
                running_acc2 += torch.sum(pred_dataset.argmax(dim=1) == label_dataset)
                #running_acc2 += np.sum(np.uint8(temp>0.5) == label_dataset.cpu().data.numpy())
                #if i > 100:
                #    break

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
    checkpoint_best = torch.load('results/checkpoint' + str(best_epoch))
    model = checkpoint_best['model']

    return model, best_epoch


class dualHex_model(torch.nn.Module):
    def __init__(self, model_core, dropout_ratio):
        super(dualHex_model, self).__init__()
        
        self.common_feature = model_core
        self.num_dataset = 2
        
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

        self.y2 = nn.Linear(4096 + 32, 5)
        nn.init.xavier_normal_(self.y2.weight)
        
        self.d_out = nn.Dropout(dropout_ratio)

    def forward(self, x, phase):

        # prepare feature
        # l2-normalize as indicated in the paper
        common_feature = self.common_feature(x)
        common_feature = self.d_out(common_feature)
        # add dropout 
        #pdb.set_trace()

        dataset_feature =  F.relu(self.x1(common_feature)) # of 32
        dataset_feature = F.normalize(dataset_feature, p=2, dim=0)

        diseases_feature = F.relu(self.x3(F.relu(self.x2(common_feature)))) # of 1024
        diseases_feature = F.normalize(diseases_feature, p=2, dim=0)

        #y_dataset = self.y1(dataset_feature) # this gonna be supervised   N x 32 -> N x 2

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


'''
class multi_output_model(torch.nn.Module):
    def __init__(self, model_core, dropout_ratio):
        super(multi_output_model, self).__init__()
        
        self.densenet_model = model_core
        self.num_dataset = 2 - 1

        #https://blog.csdn.net/Geek_of_CSDN/article/details/90179421
        
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

        self.y2 = nn.Linear(4096 + 32, 5)
        nn.init.xavier_normal_(self.y2.weight)
        
        self.d_out = nn.Dropout(dropout_ratio)

    def forward(self, x, phase):

        # prepare feature
        # l2-normalize as indicated in the paper
        common_feature = self.densenet_model(x)
        common_feature = self.d_out(common_feature)
        # add dropout 
        #pdb.set_trace()

        dataset_feature =  F.leaky_relu(self.x1(common_feature)) # of 32
        dataset_feature = F.normalize(dataset_feature, p=2, dim=0)

        diseases_feature = F.leaky_relu(self.x3(F.leaky_relu(self.x2(common_feature)))) # of 1024
        diseases_feature = F.normalize(diseases_feature, p=2, dim=0)

        #y_dataset = self.y1(dataset_feature) # this gonna be supervised   N x 32 -> N x 2

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
        
        return y_hex1, y_hex2, y_disease_pad, diseases_feature, dataset_feature
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
    BATCH_SIZE = 256

    if not os.path.exists("results/"):
        os.makedirs("results/")

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # load labels
    df = pd.read_csv("cheX_mimic.csv", index_col=0)

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
    
    
    #model = models.densenet121(pretrained=True)
    '''
    model = models.alexnet(pretrained=True)
    del model.classifier
    model.classifier = nn.Identity()
    model_new = dualHex_model(model, dropout_ratio=0.2)
    model_new.cuda()
    '''
    
    
    path_images = '/home/ben/Desktop/MIBLab/'
    path_model = '/home/ben/Desktop/MIBLab/hospital-cls/reproduce-chexnet/results/checkpoint4'
    checkpoint = torch.load(path_model, map_location=lambda storage, loc: storage)
    model_new = checkpoint['model']
    model_new.cuda()
    del checkpoint  
    '''
    
    torch.backends.cudnn.enabled = False
    preds, aucs = E.make_pred_multilabel(
    data_transforms, model_new, PATH_TO_IMAGES)
    pdb.set_trace()
    

    path_model = '/home/ben/Desktop/MIBLab/hospital-cls/reproduce-chexnet/results/checkpoint4'
    checkpoint = torch.load(path_model, map_location=lambda storage, loc: storage)
    model_new = checkpoint['model']
    model_new.cuda()
    del checkpoint
    '''
    
    
    # define criterion, optimizer for training
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.CrossEntropyLoss()

    '''
    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model_new.parameters()),
        lr=LR)
        #momentum=0.9,
        #weight_decay=WEIGHT_DECAY)
    '''

    
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


    
    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES)
    
    return preds, aucs
