import cxr_dataset as CXR
import eval_model as E
import model as M


# you will need to customize PATH_TO_IMAGES to where you have uncompressed
# NIH images
PATH_TO_IMAGES = "/home/lovebb/Documents/MIBLab/chest-Xray-dataset"
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-2
preds, aucs = M.train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY)

