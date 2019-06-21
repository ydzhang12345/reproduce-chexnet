import pickle
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import pdb



# read feature
with open('extracted_feature_sigmoid.pkl', 'rb') as f:
	extract_dict = pickle.load(f)
	x_disease, y_disease_raw = extract_dict['x_disease'], extract_dict['y_disease']
	#x_dataset, y_dataset = extract_dict['x_dataset'], extract_dict['y_dataset']
	y_dataset = extract_dict['y_dataset']


y_disease = []
for i in range(y_disease_raw.shape[0]):
	y_disease.append(int(y_disease_raw[i][0]))

#pdb.set_trace()


# t-sne
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(x_disease)

print("Org data dimension is {}. Embedded data dimension is {}".format(x_disease.shape[-1], X_tsne.shape[-1]))


# visualization
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # normalization


plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y_disease[i]), color=plt.cm.Set1(y_disease[i]), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()
plt.close()


plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y_dataset[i]), color=plt.cm.Set1(y_dataset[i]), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()
plt.close()