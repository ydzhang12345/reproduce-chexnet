from __future__ import print_function, division

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable, Function
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

setup_seed(2019)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


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
    num_epochs = 100

    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(0, num_epochs):
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
            target_loss = 0
            i = 0
            total_acc = 0.0
            total_acc_raw = 0.0
            total_done = 0
            total_batch = min(len(dataloaders[0][phase]), len(dataloaders[1][phase]))

            # iterate over all data in train/val dataloader:
            for data1, data2 in zip(dataloaders[0][phase], dataloaders[1][phase]):
                p = float(i + epoch * total_batch) / num_epochs / total_batch
                alpha =  2. / (1. + np.exp(-10 * p)) - 1

                i += 1
                # source domain
                s_inputs, s_label_disease, s_label_dataset, _ = data1
                s_label_dataset = s_label_dataset.to(dtype=torch.int64)
                s_label_dataset = s_label_dataset.reshape(-1)
                batch_size = s_inputs.shape[0]
                s_inputs = Variable(s_inputs.cuda())
                s_label_disease = Variable(s_label_disease.cuda()).float()
                s_label_dataset = Variable(s_label_dataset.cuda())

                # target domain
                t_inputs, t_label_disease, t_label_dataset, _ = data2
                t_label_dataset = t_label_dataset.to(dtype=torch.int64)
                t_label_dataset = t_label_dataset.reshape(-1)
                t_inputs = Variable(t_inputs.cuda())
                t_label_disease = Variable(t_label_disease.cuda()).float()
                t_label_dataset = Variable(t_label_dataset.cuda())


                if phase=='train':
                    s_class_out, s_domain_out = model.forward(s_inputs, alpha)
                    _, t_domain_out = model.forward(t_inputs, alpha)
                else:
                    with torch.no_grad():
                        s_class_out, s_domain_out = model.forward(s_inputs, alpha)
                        t_class_out, t_domain_out = model.forward(t_inputs, alpha)
                        #pdb.set_trace()

                if phase=='train':
                    # calculate gradient and update parameters in train phase
                    optimizer.zero_grad()
                    s_loss1 = criterion1(s_class_out, s_label_disease)
                    s_loss2 = criterion2(s_domain_out, s_label_dataset)
                    t_loss2 = criterion2(t_domain_out, t_label_dataset)
                    loss = s_loss1 + s_loss2 + t_loss2
                    print(s_loss1, ' ', s_loss2, ' ',t_loss2)
                    #print(loss1, '***', loss2)
                    loss.backward()
                    optimizer.step()
                else:
                    optimizer.zero_grad()
                    s_loss1 = criterion1(s_class_out, s_label_disease)
                    s_loss2 = criterion2(s_domain_out, s_label_dataset)
                    t_loss1 = criterion1(t_class_out, t_label_disease)
                    t_loss2 = criterion2(t_domain_out, t_label_dataset)
                    loss = s_loss1 + s_loss2 + t_loss2

                    '''
                    probs = (torch.sigmoid(t_class_out)).cpu().data.numpy()
                    label_disease = label_disease.cpu().data.numpy()
                    total_acc += np.sum(np.uint8(probs>0.5)==label_disease)
                    '''
                    target_loss += t_loss1.data * batch_size

                running_loss1 += s_loss1.data * batch_size
                running_loss2 += s_loss2.data + t_loss2.data
                #break
                #running_acc2 += torch.sum(domain_out.argmax(dim=1) == label_dataset)

            #pdb.set_trace()
            epoch_loss1 = running_loss1 / dataset_sizes[0][phase]
            epoch_loss2 = running_loss2 / total_batch 
            #epoch_accuracy = running_acc2.to(dtype=torch.float32) / dataset_sizes[phase]
            if phase == 'train':
                last_train_loss = epoch_loss1
            else:
                print("target domain loss: ", target_loss / dataset_sizes[1][phase])

            print(phase + ' epoch {}:loss1 {:.4f}, loss2 {:.4f} with data size {}'.format(
                epoch, epoch_loss1, epoch_loss2, dataset_sizes[0][phase]))

            
            # decay learning rate if no val loss improvement in this epoch
            '''
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
            '''
            
            
            
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
    checkpoint_best = torch.load('results/checkpoint' + str(best_epoch))
    model = checkpoint_best['model']

    return model, best_epoch



class multi_output_model(torch.nn.Module):
    def __init__(self, model_core, dropout_ratio):
        super(multi_output_model, self).__init__()
        
        self.densenet_model = model_core

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(1024, 1024))
        #self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(1024))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        #self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        #self.class_classifier.add_module('c_fc2', nn.Linear(1024, 1024))
        #self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(1024))
        #self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(1024, 5))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(1024, 32))
        #self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(32))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(32, 2))
        self.d_out = nn.Dropout(0.2)

    def forward(self, x, alpha):
        # prepare feature
        common_feature = self.d_out(self.densenet_model(x))
        reverse_feature = ReverseLayerF.apply(common_feature, alpha)
        class_output = self.class_classifier(common_feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output



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
    BATCH_SIZE = 25

    if not os.path.exists("results/"):
        os.makedirs("results/")

    # use imagenet mean,std for normalization
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

    # create train/val dataloaders
    transformed_datasets1 = {}
    transformed_datasets1['train'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='train',
        label_path='sampled_nih.csv',
        transform=data_transforms['train'])
    transformed_datasets1['val'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='val',
        label_path='sampled_nih.csv',
        transform=data_transforms['val'])
    dataloaders1 = {}
    dataloaders1['train'] = torch.utils.data.DataLoader(
        transformed_datasets1['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataloaders1['val'] = torch.utils.data.DataLoader(
        transformed_datasets1['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)

    transformed_datasets2 = {}
    transformed_datasets2['train'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='train',
        label_path='sampled_chex.csv',
        transform=data_transforms['train'])
    transformed_datasets2['val'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='val',
        label_path='sampled_chex.csv',
        transform=data_transforms['val'])
    dataloaders2 = {}
    dataloaders2['train'] = torch.utils.data.DataLoader(
        transformed_datasets2['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataloaders2['val'] = torch.utils.data.DataLoader(
        transformed_datasets2['val'],
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
    model_new.cuda()
    

    '''  
    path_images = '/home/ben/Desktop/MIBLab/'
    path_model = '/home/ben/Desktop/MIBLab/hospital-cls/reproduce-chexnet/results/checkpoint9'
    checkpoint = torch.load(path_model, map_location=lambda storage, loc: storage)
    model_new = checkpoint['model']
    model_new.cuda()
    del checkpoint
    '''

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
    
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model_new.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    
    dataset_sizes1 = {x: len(transformed_datasets1[x]) for x in ['train', 'val']}
    dataset_sizes2 = {x: len(transformed_datasets2[x]) for x in ['train', 'val']}
    dataloaders = [dataloaders1, dataloaders2]
    dataset_sizes = [dataset_sizes1, dataset_sizes2]

    # train model
    model, best_epoch = train_model(model_new, criterion1, criterion2, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY)


    
    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES)
    
    return preds, aucs
