import csv
import pdb
import numpy as np

nih_imgs = {}
nih_list = []
chex_imgs = {}
chex_list = []
mimic_imgs = {}
mimic_list = []

with open('labels.csv') as file:
	reader = csv.reader(file, quotechar='|')
	count = 0
	for row in reader:
		if count==0: 
			count+=1
			start = row
			continue
		if "NIH" in row[0]:
			nih_imgs[row[0]] = row
			nih_list.append(row[0])
		else:
			chex_imgs[row[0]] = row
			chex_list.append(row[0])

with open('mimic_labels.csv') as file:
	reader = csv.reader(file, quotechar='|')
	count = 0
	pre = "MIMIC/"
	for row in reader:
		if count==0: 
			count+=1
			continue
		mimic_imgs[pre + row[0]] = row
		mimic_list.append(pre + row[0])

# write new csv file
np.random.seed(2019)
nih = {"train":0, "val":0, "test":0}
chex = {"train":0, "val":0, "test":0}
mimic = {"train":0, "val":0, "test":0}
with open('sampled_cheX_mimic.csv', 'w', newline='') as file:
	writer = csv.writer(file, quotechar='|', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(start)
	for img in chex_list:
		p = np.random.uniform()
		if p < 0.1:
			writer.writerow([img] + chex_imgs[img][1:])
			chex[chex_imgs[img][-1]] += 1
	for img in mimic_list:
		p = np.random.uniform()
		if p < 0.1:
			mimic_imgs[img][1] = 0 # modify dataset ID to be 2-class
			writer.writerow([img] + mimic_imgs[img][1:])
			mimic[mimic_imgs[img][-1]] += 1
	'''
	for img in nih_list:
		p = np.random.uniform()
		if p < 0.1:
			writer.writerow([img, nih_imgs[img]["dataset"], nih_imgs[img]["fold"]])
			nih[nih_imgs[img]["fold"]] += 1
	'''

#print(nih)
print(chex)
print(mimic)
