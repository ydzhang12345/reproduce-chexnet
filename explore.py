import visualize_prediction as V

import pandas as pd

#suppress pytorch warnings about source code changes
import warnings
warnings.filterwarnings('ignore')

#STARTER_IMAGES=True
#PATH_TO_IMAGES = "starter_images/"

STARTER_IMAGES=False
PATH_TO_IMAGES = "/home/ben/Desktop/MIBLab/"

PATH_TO_MODEL = "results/checkpoint"

LABEL="Hospital"

POSITIVE_FINDINGS_ONLY=False

dataloader,model= V.load_data(PATH_TO_IMAGES,LABEL,PATH_TO_MODEL,POSITIVE_FINDINGS_ONLY,STARTER_IMAGES)
print("Cases for review:")
print(len(dataloader))


s = 100

for i in range(s):
	preds=V.show_next(dataloader,model, LABEL)
	preds