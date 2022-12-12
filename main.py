from torch.utils.data import Dataset, DataLoader
import torch 
import numpy as np
import clean_images
import clean_tabular_data
import torchvision 
import os
from PIL import Image
import pandas as pd

#TODO Make sure everything is tensors! 
#TODO add rbg / greyscale in getitem w/inj dataset when relevant (see ivan's tips)
#makesure biases and weights tensors have requires_grad = true! 


class ImageDataset (Dataset):
    def __init__(self):
        # clean_images.clean_image_data("./images")
        #self.image_tensors = self.images_to_tensors('./cleaned_images')

       #clean and merge tabular data
        self.prod_data = clean_tabular_data.clean_table_data("Products.csv")
        self.image_ids = pd.read_csv ("Images.csv")
        self.merged_data = self.image_ids.merge(self.prod_data[['category','product_id']], on='product_id') #not all data is kept in the data!
        
        #seets labels and image WHAT? 
        self.labels = self.merged_data['category'].to_list()
        self.image_files = self.merged_data['id'].to_list()

        #encode labels
        self.encoded_labels = {}
        self.encode_labels (self.merged_data)


    def encode_labels(self, merged_data):
        full_catagories = merged_data['category'].unique()
        for cat in enumerate (full_catagories):
            self.encoded_labels[cat[1]] = cat [0]

    

    # def decode_label(self, label_index):
    #     return self.encoded_labels[label_index]
            

    def __getitem__ (self,index): # this is what should be returned by each batch; the label also, but I don't call this in the init? 
        """
        returns a tuple of features and labels; c
        """
        label = self.labels[index]
        encoded_label = torch.tensor(self.encoded_labels[label])
        image = self.image_files[index]

        PIL_image = Image.open(f"cleaned_images/{image}_resized.jpg")     
        transform = torchvision.transforms.PILToTensor() 
        feature = transform(PIL_image)
        #features = # transform the image to a tensor
        print (feature.shape)
        return feature, encoded_label


    def __len__(self):
        return len (self.merged_data)





class Cnn(torch.nn.Module): 
    def __init__(self):
        #initialise parameters
        super().__init__() # GIVE THIS A QUICK GOOGLE SO YOU MAKE SURE YOU UNDERSTAND IT!! 
        self.layer1 = torch.nn.Linear(1536, 1) #features, outputs; how do i check how many I need? Can i just run it? 
        # self.layer2 = torch.nn.Linear(2000, 1000)
        # self.layer3 = torch.nn.Linear(1000, 512)

    def forward (self,features): #replaces __call__  (this is inherited from the nn.module!)
        print ("FEATURES are")
        print (features.shape)
        return self.layer1(features)




def train(model, epochs=10):
    for epoch in range (epochs):
        for batch in train_loader:
            features, labels = batch  #unpacks features and labels from the batch
            prediction = model(features) #make a prediction with our model
            loss = torch.nn.functional.mse_loss (prediction, labels) #calculates the loss of the model
            loss.backward()
            print (loss)
            #then optimise! 





dataset = ImageDataset()
# print ("GOGOG GASDHET")
# print(dataset[50])
batch_size = 1
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #customise batch size; will be number of groups of outputs returned in one run

 #error is here; for some reason My loader can't iterate through the data at all!
# --> theres probably something wrong with DATASET


example = next(iter(train_loader))
print (example)

model=Cnn()
train (model)

   #add image label to row

# def images_to_tensors (self, parent_dir):
#         tensor_list = []
#         for image_id in os.listdir(parent_dir):
#             PIL_image = Image.open(f"{parent_dir}/{image_id}")     
#             transform = torchvision.transforms.PILToTensor() 
#             image_tensor = transform(PIL_image)
#             tensor_list.append(image_tensor)
#         return torch.cat(tensor_list) # should this be a tuple? or whats the structure here, and where does the tuple go?    

            # real_id = image_id[:-12] #indexes out "_cleaned.jpeg"
            # row = self.merged_data.loc[self.merged_data['id'] == real_id] #finds row related to image
            # print ("ROW IS")
            # print (row)
            # category = row.iloc[0,3] #finds category of that row
            # print (f"CAT IS {category}")
            # label = self.encoded_labels[category] #encodes category to label
            # print (f"LABEL IS {label}")
            # element_lisFee for changes made 60 days or less from departure: 53.00 € per passenger per flight


# def images_to_tensors (self, parent_dir):
#         tensor_list = []
#         for image_id in os.listdir(parent_dir):
#             PIL_image = Image.open(f"{parent_dir}/{image_id}")     
#             transform = torchvision.transforms.PILToTensor() 
#             image_tensor = transform(PIL_image)
#             tensor_list.append(image_tensor)
#         return torch.cat(tensor_list) # should this be a tuple? or whats the structure here, and where does the tuple go?    
#https://discuss.pytorch.org/t/runtimeerror-mat1-and-mat2-shapes-cannot-be-multiplied-64x13056-and-153600x2048/101315/6 