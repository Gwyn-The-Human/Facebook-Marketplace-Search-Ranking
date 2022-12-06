from torch.utils.data import Dataset, DataLoader
import torch 
import numpy as np
import clean_images
import clean_tabular_data
import torchvision 
import os
from PIL import Image
import pandas as pd

#makesure biases and weights tensors have requires_grad = true! 


class ImageDataset (Dataset):
    def __init__(self):
        self.prod_data = clean_tabular_data.clean_table_data("Products.csv")
        self.image_ids = pd.read_csv ("Images.csv")
        self.merged_data = self.image_ids.merge(self.prod_data[['category','product_id']], on='product_id') #not all data is kept in the data!
        self.encoded_labels = {}
        self.encode_labels (self.merged_data)
        clean_images.clean_image_data("./images")
        self.image_tensors = self.images_to_tensors('./cleaned_images')
        print ("ENCODED LABELS ARE")
        print (self.encoded_labels)
        print ("MERGED DATA:")
        print (self.merged_data)
        print ("IMAGE TENSORS")
        print (self.image_tensors)


    def encode_labels(self, merged_data):
        full_catagories = merged_data['category'].unique()
        for cat in enumerate (full_catagories):
            self.encoded_labels[cat[0]] = cat [1]


    def images_to_tensors (self, parent_dir):
        tensor_list = []
        for image_label in os.listdir(parent_dir):
            PIL_image = Image.open(f"{parent_dir}/{image_label}")     
            transform = torchvision.transforms.PILToTensor() 
            image_tensor = transform(PIL_image)
            tensor_list.append (image_tensor)
            # self.encode_label (image_label)
        return torch.cat(tensor_list) # should this be a tuple? or whats the structure here, and where does the tuple go?    


    def decode_label(self, label_index):
        return self.encoded_labels[label_index]
            

    def __getitem__ (self,index): # this is what should be returned by each batch; the label also, but I don't call this in the init? 
        """
        returns a tuple of features and labels; c
        """
        example = self.data.iloc[index]
        features = example[:a] #which features /columns do I want to return?
        label = example[b] #play around with what these return
        return (features, label)

    def __len__(self):
        return len (self.data)


def train(model, epochs):
    for epoch in range (epochs):
        for batch in train_loader:
            features = batch #in teh data set harry uses, this can be features, labels = batch; 
#for mine, my dataset just has features i think. Update it so it does later, and just run like this now to see how it goes
            prediction = model(features)
            loss = torch.nn.functional.mse_loss (prediction, labels)
            loss.backward()
            print (loss)
            #then optimise! 




class Cnn(torch.nn.Module): 
    def __init__(self):
        #initialise parameters
        super().__init__() # GIVE THIS A QUICK GOOGLE SO YOU MAKE SURE YOU UNDERSTAND IT!! 
        self.layer1 = torch.nn.Linear(10, 1) #features, outputs; how do i check how many I need? Can i just run it? 

    def forward (self,features): #replaces __call__  (this is inherited from the nn.module!)
        return self.layer1(features)













dataset = ImageDataset()
train_loader = DataLoader(dataset.merged_data, batch_size=10, shuffle=True) #customise batch size; will be number of groups of outputs returned in one run
model=Cnn()






# bcount = 0
# for batch in train_loader:
#     bcount += 1
#     for x in batch:
#         print ("Printing things inside each batch")
#         print (x)
#     print ("ASSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSD")

# print (bcount)

# example = next(iter(train_loader))
# a, b =example
# print (a)
# print (b)

# print (train (model, 1))


