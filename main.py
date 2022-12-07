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
       #clean and merge tabular data
        self.prod_data = clean_tabular_data.clean_table_data("Products.csv")
        self.image_ids = pd.read_csv ("Images.csv")
        self.merged_data = self.image_ids.merge(self.prod_data[['category','product_id']], on='product_id') #not all data is kept in the data!
        
        #encode labels
        self.encoded_labels = {}
        self.encode_labels (self.merged_data)

        #clean images and convert to tensor
        clean_images.clean_image_data("./images")
        self.image_tensors = self.images_to_tensors('./cleaned_images')

        self.features =   #just gotta work out how to do this! 
        self.labels = 

        #I guess ask yourself what happens when this dataset is called by the dataloader? 
        # Does it search through batches using get item?; if not, at what point are these labels, features returned?
        #and how are they linked/ related? 


    def encode_labels(self, merged_data):
        full_catagories = merged_data['category'].unique()
        for cat in enumerate (full_catagories):
            self.encoded_labels[cat[1]] = cat [0]
        # for label in merged_data['category']:
        #     label = self.encoded_labels[label]
        #     print (label)

        # print (merged_data['category'])



    def images_to_tensors (self, parent_dir):
        tensor_list = []
        for image_id in os.listdir(parent_dir):
            PIL_image = Image.open(f"{parent_dir}/{image_id}")     
            transform = torchvision.transforms.PILToTensor() 
            image_tensor = transform(PIL_image)
            tensor_list.append(image_tensor)
        return torch.cat(tensor_list) # should this be a tuple? or whats the structure here, and where does the tuple go?    


    def decode_label(self, label_index):
        return self.encoded_labels[label_index]
            

    def __getitem__ (self,index): # this is what should be returned by each batch; the label also, but I don't call this in the init? 
        """
        returns a tuple of features and labels; c
        """
        print ("RUNNING GET ITEM!!!")
        example = self.merged_data.iloc[index]
        features = example['product_id'] #which features /columns do I want to return?
        label = example['category'] #play around with what these return
        return (features, label)

    def __len__(self):
        return len (self.merged_data)





class Cnn(torch.nn.Module): 
    def __init__(self):
        #initialise parameters
        super().__init__() # GIVE THIS A QUICK GOOGLE SO YOU MAKE SURE YOU UNDERSTAND IT!! 
        self.layer1 = torch.nn.Linear(12604, 435) #features, outputs; how do i check how many I need? Can i just run it? 

    def forward (self,features): #replaces __call__  (this is inherited from the nn.module!)
        return self.layer1(features)




def train(model, epochs):
    for epoch in range (epochs):
        for batch in train_loader:
            features, labels = batch  
            prediction = model(features)
            loss = torch.nn.functional.mse_loss (prediction, labels)
            loss.backward()
            print (loss)
            #then optimise! 





dataset = ImageDataset()
batch_size = 10
train_loader = DataLoader(dataset.image_tensors, batch_size=batch_size, shuffle=True) #customise batch size; will be number of groups of outputs returned in one run
it = iter(train_loader)
first = next(it) #error is here; for some reason My loader can't iterate through the data at all!
# --> theres probably something wrong with DATASET


example = next(iter(train_loader))
print (example)

# model=Cnn()
# model(features)
# train (model, 2)

   #add image label to row
            # real_id = image_id[:-12] #indexes out "_cleaned.jpeg"
            # row = self.merged_data.loc[self.merged_data['id'] == real_id] #finds row related to image
            # print ("ROW IS")
            # print (row)
            # category = row.iloc[0,3] #finds category of that row
            # print (f"CAT IS {category}")
            # label = self.encoded_labels[category] #encodes category to label
            # print (f"LABEL IS {label}")
            # element_list.append(torch.tensor([[[label]]]))
            # print (len(element_list))
            # tensor_list.append (torch.cat(element_list))



