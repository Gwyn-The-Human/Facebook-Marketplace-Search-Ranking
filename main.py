from torch.utils.data import Dataset, DataLoader
import torch 
import numpy as np
import clean_images
import clean_tabular_data
import torchvision 
import os
from PIL import Image
import pandas as pd
import random
import NN
from torch.utils.tensorboard import SummaryWriter

SEED = (123)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#TODO normalise data?
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
        
        #sets labels and image 
        self.labels = self.merged_data['category'].to_list()
        self.image_files = self.merged_data['id'].to_list()

        #encode labels
        self.encoded_labels = {}
        self.encode_labels (self.merged_data)


    def encode_labels(self, merged_data): #435 labels total
        full_catagories = merged_data['category'].unique()
        for cat in enumerate (full_catagories):
            self.encoded_labels[cat[1]] = cat [0]

    

    # def decode_label(self, label_index):
    #     return self.encoded_labels[label_index]
            

    def __getitem__ (self,index): # this is what should be returned by each batch; the label also. 
        """
        returns a tuple of features and labels; c
        """
        label = self.labels[index]
        encoded_label = torch.tensor(self.encoded_labels[label])
        image = self.image_files[index]

        PIL_image = Image.open(f"cleaned_images/{image}_resized.jpg")     
        transform = torchvision.transforms.PILToTensor() 
        feature = transform(PIL_image)
        feature = torch.flatten(feature).to(torch.float32)
        # print (feature.shape)
        # print ("LABEL IS")
        # print (encoded_label)
        return feature, encoded_label


    def __len__(self):
        return len (self.merged_data)

#input size should be just the size of the image; 
#flatten the image (3d)
#Go through my db and make sure I understand how I get the features of the shape that I have; then I need to make them 2 D


def train(model, epochs=4):
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    batch_idx = 0
    writer = SummaryWriter()
    for epoch in range (epochs):
        for batch in train_loader:
            features, labels = batch  #unpacks features and labels from the batch
            prediction = model(features) #make a prediction with our model; currently is returning a tuple
            labels = labels.unsqueeze(1).float()
            loss = torch.nn.functional.mse_loss(prediction, labels) #calculates the loss of the model
            loss.backward()
            print (loss.item())
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1
            #then optimise! 
            optimiser.step()
            optimiser.zero_grad() #this reverts the grad value to zero so .backwards will overwright (otherwise it would just add to the grad val)

dataset = ImageDataset()
batch_size = 100
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #customise batch size; will be number of groups of outputs returned in one run

# example = next(iter(train_loader))
# print (example)

model = NN.Cnn()
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