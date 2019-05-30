"""
Created on Thu Oct 26 14:19:44 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np
import torch
import pdb

from torch.optim import SGD
from torchvision import models

from misc_functions import preprocess_image, recreate_image, save_image


class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target_class):
        self.model = model
        self.model.eval()
        self.target_class = target_class
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
        # Create the folder to export images if not exists
        if not os.path.exists('./generate_class_specific'):
            os.makedirs('./generate_class_specific')

    def generate(self):
        initial_learning_rate = 25
        for i in range(1, 1000):
            # Process image and return variable
            self.processed_image = preprocess_image(self.created_image, False)
            #pdb.set_trace()
            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=initial_learning_rate)
            # Forward
            output = self.model(self.processed_image)
            # Target specific class
            #pdb.set_trace()
            class_loss = -output[0, self.target_class] + 1e-4 * torch.sum(self.processed_image**2) 
            print('Iteration:', str(i), 'Loss', "{0:.2f}".format(class_loss.data.numpy()))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            if i % 100 == 0:
                # Save image
                #initial_learning_rate /= 2
                im_path = './generate_class_specific/c_specific_l2_iteration_'+str(i)+'.jpg'
                save_image(self.created_image, im_path)
        return self.processed_image


if __name__ == '__main__':
    target_class = 0 # NIH

    path_model = '/home/lwv/Downloads/reproduce-chexnet/DenseNet101-results/checkpoint'

    checkpoint = torch.load(path_model, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    del checkpoint
    #model.cuda()

    #pretrained_model = models.alexnet(pretrained=True)
    csig = ClassSpecificImageGeneration(model, target_class)
    csig.generate()
