from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import clean_images
import torchvision 
import os
from PIL import Image



   


class ImageDataset (Dataset):
        
    def __init__(self):
        clean_images.clean_image_data("./images")
        self.encoded_labels = {}
        self.label_count = 0
        self.data = self.images_to_tensors('./cleaned_images')
        self.train_loader = DataLoader (self.data, batch_size=10, shuffle=True) #customise batch size


    def images_to_tensors (self, parent_dir):
        tensor_list = []
        for image_label in os.listdir(parent_dir):
            PIL_image = Image.open(f"{parent_dir}/{image_label}")     
            transform = torchvision.transforms.PILToTensor() 
            image_tensor = transform(PIL_image)
            tensor_list.append (image_tensor)
            self.encode_label (image_label)
        return torch.cat(tensor_list)    


    def encode_label (self, image_label):
        self.encoded_labels[self.label_count] = image_label
        self.label_count += 1


    def decode_label(self, label_index):
        return self.encoded_labels[label_index]

            #iterate through images and:
                #clean them
                #transform them into tensors
                #save their labels
            

    def __getitem__ (self,index):
        """
        returns a tuple of features and labels; c
        """
        example = self.data.iloc[index]
        features = example[:a] #which features /columns do I want to return?
        label = example[b] #play around with what these return
        return (features, label)

    def __len__(self):
        return len (self.data)

    def encode(self):
        pass

    def decode(self):
        pass


test = ImageDataset()

print (test.encoded_labels)
# print(next(iter(test.train_loader)))



# There are a few things to remember when you create the Dataset:

# The dataset should contain the images and the labels. 
# You have two datasets, one for the images and one for the labels, so you should know what category correspond to what image.
# 
# You have to assign a label to each category, for example "Home & Garden" is category 0, "Appliances" is category 1, etc.
#  This will be your encoder, which can be a dictionary. Thus, the output you obtain from the model
#  will contain a list of numbers, which correspond to the categories, but you need to translate them. 
# That way, in addition to creating the encoder, you should also create a decoder.

# While not necessary, you can add image transformations to your dataset. For example, rotate the images or flip them horizontally.
# This is called data augmentation.

# A useful step here is to test that your DataLoader is working by running something like this:


