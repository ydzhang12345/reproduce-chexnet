import os
import io
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
from guided_backprop import GuidedBackprop
import pandas as pd


def get_cam(image_path, net):
	finalconv_name = 'layer4'
	normalize = transforms.Normalize(
	   mean=[0.485, 0.456, 0.406],
	   std=[0.229, 0.224, 0.225]
	)
	preprocess = transforms.Compose([
	   transforms.Resize((224,224)),
	   transforms.ToTensor(),
	   normalize
	])

	# hook the feature extractor
	features_blobs = []
	def hook_feature(module, input, output):
	    features_blobs.append(output.data.cpu().numpy())

	net._modules.get(finalconv_name).register_forward_hook(hook_feature)

	# get the softmax weight
	params = list(net.parameters())
	weight_softmax = np.squeeze(params[-2].data.numpy())

	def returnCAM(feature_conv, weight_softmax, class_idx):
	    # generate the class activation maps upsample to 256x256
	    size_upsample = (224, 224)
	    bz, nc, h, w = feature_conv.shape
	    output_cam = []
	    for idx in class_idx:
	        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
	        cam = cam.reshape(h, w)
	        cam = cam - np.min(cam)
	        cam_img = cam / np.max(cam)
	        cam_img = np.uint8(255 * cam_img)
	        output_cam.append(cv2.resize(cam_img, size_upsample))
	    return output_cam


	img_pil = Image.open(img_path).convert('RGB')
	img_pil.save(out_path + 'test.jpg')

	img_tensor = preprocess(img_pil)
	img_variable = Variable(img_tensor.unsqueeze(0))
	logit = net(img_variable)

	# download the imagenet category list
	'''
	classes = {int(key):value for (key, value)
	          in requests.get(LABELS_URL).json().items()}

	'''
	classes = {0: "NIH", 1: "CheXpert"}

	h_x = F.softmax(logit, dim=1).data.squeeze()
	probs, idx = h_x.sort(0, True)
	probs = probs.numpy()
	idx = idx.numpy()

	# output the prediction
	for i in range(0, 2):
	    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

	# generate class activation mapping for the top1 prediction
	CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

	'''
	# render the CAM and output
	print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
	img = cv2.imread('test.jpg')
	height, width, _ = img.shape
	heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
	result = heatmap * 0.3 + img * 0.5
	cv2.imwrite('CAM.jpg', result
	'''
	return CAMs[0], idx[0]


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
    	pil_im = pil_im.resize((224, 224))
        #pil_im.thumbnail((224, 224))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


#path_model = '/home/lwv/Downloads/reproduce-chexnet/ResNet101-results/checkpoint'
path_model = '/home/ben/Desktop/MIBLab/hospital-cls/reproduce-chexnet/ResNet101-results/checkpoint'
checkpoint = torch.load(path_model, map_location=lambda storage, loc: storage)
net = checkpoint['model']
del checkpoint
net.eval()


#img_path = '/home/lwv/Downloads/reproduce-chexnet/starter_images/00000284_000.png'
df = pd.read_csv("/home/ben/Desktop/MIBLab/hospital-cls/reproduce-chexnet/hospital_labels.csv")
df = df[df['fold'] == 'test']
df = df.set_index("Image Index")
out = 'guided_grad_resnet/'

test_size = 56405
np.random.seed(2019)
class_name = ['NIH', 'CheXpert']
for ii in range(100):
	out_path = out + str(ii) + '/'
	os.mkdir(out_path)
	ind = int(np.random.sample() * test_size)
	img_path = '/home/ben/Desktop/MIBLab/' + df.index[ind]

	# get top1-prediction cam 
	cam, class_idx = get_cam(img_path, net)
	img = cv2.imread(out_path + 'test.jpg')
	height, width, _ = img.shape
	heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
	result = heatmap * 0.3 + img * 0.5
	cv2.imwrite(out_path + class_name[class_idx] + '-CAM.jpg', result)

	# Guided backprop
	GBP = GuidedBackprop(net)
	# Get gradients
	# Read image
	original_image = Image.open(img_path).convert('RGB')
	# Process image
	prep_img = preprocess_image(original_image)
	target_class = class_idx
	guided_grads = convert_to_grayscale(GBP.generate_gradients(prep_img, target_class))

	final_img = cam * guided_grads[0]
	final_img = (final_img - final_img.min()) / (final_img.max() - final_img.min())
	final_img = np.uint8(255 * final_img)


	# render the CAM and output
	#print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
	img = cv2.imread(out_path + 'test.jpg')
	height, width, _ = img.shape
	heatmap = cv2.applyColorMap(cv2.resize(final_img,(width, height)), cv2.COLORMAP_JET)
	result = heatmap * 0.3 + img * 0.5
	cv2.imwrite(out_path + 'Guided-CAM-' + class_name[class_idx] + '.jpg', result)
