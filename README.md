# reproduce-chexnet

## This forked repo will mainly focus on the hospital systyem prediction

## NIH Dataset
To explore the full dataset, [download images from NIH (large, ~40gb compressed)](https://nihcc.app.box.com/v/ChestXray-NIHCC),
extract all `tar.gz` files to a single folder, and provide path as needed in code.

## CheXpert Dataset
https://stanfordmlgroup.github.io/competitions/chexpert/

## Results:
A DenseNet-121 model correctly classifies NIH (25595 out of 25596) images and CheXpert (30807 out of 30809) images with 99.99% accuracy and AUC score of 0.99 (train with binary-classification setting).

A ResNet-101 model correctly classifies NIH (25594 out of 25596) images and CheXpert (30802 out of 30809) images with 99.98% accuracy and AUC score of 0.99 (train with binary-classification setting).






