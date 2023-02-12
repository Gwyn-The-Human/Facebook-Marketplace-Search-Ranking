from datetime import datetime
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import fb_dataset
import image_processor
import json
import os
import torch 
import torch.nn as nn


##Have to load the final model weights from main.py, because we need the output neurons to be 435 so it can be trained to the catafories.
##This is awesomE!!!! It means once it can analyse the pictures accuretly, we can re-write it to tell us totalyl new things about them!
##Even things that are incomprehensible to humans! WowoWOWoWOWOwo!


class FeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = nn.Linear(2048,1000) 


    def forward(self, X):
        return self.resnet50(X)

def save_embeddings(image_dir):  #takes the directory within which you want to 
    image_embeddings = {}
    extract = FeatureExtraction()
    with open("image_embeddings.json", "w") as fp:
        for id in os.listdir(image_dir):
            image = Image.open(f"{image_dir}/{id}")
            image = image_processor.proc_img(image)
            embedding = extract(image.float())
            image_embeddings[id] = embedding
        json.dump(image_embeddings, fp)

save_embeddings("images")

# Use the feature extraction model to extract feature or image embeddings for 
# every image in the training dataset.

# Create a dictionary where every key is the image id and value is the image embedding. 
# Save this dictionary as a JSON file named image_embeddings.json.
