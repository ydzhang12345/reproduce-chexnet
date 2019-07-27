from PIL import Image
import numpy as np
import csv
import os
import pdb
import cv2

chest_bank = []
mimic_bank = []
label_path = 'sampled_cheX_mimic.csv'
with open(label_path) as file:
	reader = csv.reader(file, quotechar='|')
	count = 0
	for row in reader:
		if count==0:
			count += 1
			continue
		if 'CheXpert' in row[0]:
			chest_bank.append(row[0])
		else:
			mimic_bank.append(row[0])

chest_imgs = []
mimic_imgs = []

path_to_images = '/home/ben/Desktop/MIBLab/'
for i in chest_bank:
	image = Image.open(
            os.path.join(
                path_to_images,
                i))
	img = np.array(image)
	img = cv2.resize(img, (224, 224))
	chest_imgs.append(img)
	image.close()

pdb.set_trace()

for i in mimic_bank:
	image = Image.open(
            os.path.join(
                path_to_images,
                i))
	img = np.array(image)
	img = cv2.resize(img, (224, 224))
	mimic_imgs.append(img)
	image.close()
chest_imgs = np.array(chest_imgs)
mimic_imgs = np.array(mimic_imgs)

pdb.set_trace()


def compute_mean_and_std(image_list):
	mean = np.mean(image_list)
	std = np.std(image_list)
	return mean, std


print ("chest: ", compute_mean_and_std(chest_imgs))
print ("mimic: ", compute_mean_and_std(mimic_imgs))



