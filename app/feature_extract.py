#from datetime import datetime
from PIL import Image
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import image_processor
import json
import os
import torch 
import torch.nn as nn
import tuned_model



##Have to load the final model weights from main.py, because we need the output neurons to be 435 so it can be trained to the catafories.
##This is awesomE!!!! It means once it can analyse the pictures accuretly, we can re-write it to tell us totalyl new things about them!
##Even things that are incomprehensible to humans! WowoWOWoWOWOwo!



# class FeatureExtraction(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
#         self.resnet50.fc = nn.Linear(2048,1000) 


#     def forward(self, X):
#         return self.resnet50(X)


def load_extraction_model():        
        model = tuned_model.TunedResnet()
        state_dict = torch.load("final_model/image_model.pt")
        model.load_state_dict(state_dict)
        model.resnet50.fc = nn.Linear(2048,1000)
        return model


def save_embeddings(image_dir):  #takes the directory within which you want to 
    extract = load_extraction_model()
    image_embeddings = {}
    with open("image_embeddings.json", "w") as fp:
        for id in os.listdir(image_dir):
            image = Image.open(f"{image_dir}/{id}")
            image = image_processor.proc_img(image)
            embedding = extract(image.float())
            image_embeddings[id] = embedding.tolist()
        json.dump(image_embeddings, fp)
        #print ("dumped")


        
if __name__ == '__main__':
    save_embeddings("images")

