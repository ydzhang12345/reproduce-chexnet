import torch
import pandas as pd
import cxr_dataset as CXR
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn
import sklearn.metrics as sklm
from torch.autograd import Variable
import numpy as np
import pdb


def make_pred_multilabel(data_transforms, model, PATH_TO_IMAGES):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model

    Args:
        data_transforms: torchvision transforms to preprocess raw images; same as validation transforms
        model: densenet-121 from torchvision previously fine tuned to training data
        PATH_TO_IMAGES: path at which NIH images can be found
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    # calc preds in batches of 16, can reduce if your GPU has less RAM
    BATCH_SIZE = 16

    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    # create dataloader
    dataset = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold="test",
        transform=data_transforms['val'])
    dataloader = torch.utils.data.DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=8)
    size = len(dataset)

    # create empty dfs
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    # iterate over dataloader
    acc = 0
    for i, data in enumerate(dataloader):

        inputs, label_disease, label_dataset, _ = data
        label_dataset = label_dataset.to(dtype=torch.int64)
        label_dataset = label_dataset.reshape(-1)

        batch_size = inputs.shape[0]
        inputs = Variable(inputs.cuda())
        label_disease = Variable(label_disease.cuda()).float()
        label_dataset = Variable(label_dataset.cuda())


        #inputs, labels, _ = data
        #labels = labels.to(dtype=torch.int64)
        #labels = labels.reshape(-1)
        #inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        #true_labels = labels.cpu().data.numpy()
        #batch_size = true_labels.shape
        label_disease = label_disease.cpu().data.numpy()
        label_dataset = label_dataset.cpu().data.numpy()

        disease_pred, dataset_pred = model(inputs)
        #outputs = torch.nn.functional.softmax(outputs, dim=1)
        probs = disease_pred.cpu().data.numpy()
        acc += torch.sum(dataset_pred.argmax(dim=1) == labels)
        # get predictions and true values for each item in batch
        for j in range(0, batch_size[0]):
            thisrow = {}
            truerow = {}
            thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]
            truerow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]

            # iterate over each entry in prediction vector; each corresponds to
            # individual label
            if len(dataset.PRED_LABEL_DATASET)==1:
                thisrow["prob_" + "Dataset ID"] = dataset_pred[j, 0]
                truerow["Dataset ID"] = int(label_dataset[j]==0)
            for k in range(len(dataset.PRED_LABEL_DISEASE)):
                thisrow["prob_" + dataset.PRED_LABEL_DISEASE[k]] = probs[j, k]
                truerow[dataset.PRED_LABEL_DISEASE[k]] = label_disease[j, k]

            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)
        
        if(i % 10 == 0):
            print(str(i * BATCH_SIZE))
    print (acc.to(dtype=torch.float32) / ((i+1)*BATCH_SIZE)) 
    auc_df = pd.DataFrame(columns=["label", "auc"])
    # calc AUCs
    for column in true_df:
        if column not in [
            'Dataset ID'
            'Atelectasis',
            'Cardiomegaly',
            'Consolidation',
            'Edema',
            'Effusion']:
                    continue
        
        actual = true_df[column]
        pred = pred_df["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        thisrow['auc'] = np.nan
        try:
            thisrow['auc'] = sklm.roc_auc_score(
                actual.as_matrix().astype(int), pred.as_matrix())
        except BaseException:
            print("can't calculate auc for " + str(column))
        auc_df = auc_df.append(thisrow, ignore_index=True)

    pred_df.to_csv("results/preds.csv", index=False)
    auc_df.to_csv("results/aucs.csv", index=False)
    return pred_df, auc_df
