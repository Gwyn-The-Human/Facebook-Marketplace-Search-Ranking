from datetime import datetime
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import image_processor
import json
import os
import torch 
import torch.nn as nn
import resnet_training


def load_extraction_model():        
    """
    Loads the model, loads weights from findal_model directory, and changes the last layer 
    of the model to have 1000, so as to extract 1000 features. 

    Returns: 
        The model with loaded weights, modified for feature extraction. 
    """
    model = resnet_training.TunedResnet()
    state_dict = torch.load("final_model/image_model.pt")
    model.load_state_dict(state_dict)
    model.resnet50.fc = nn.Linear(2048,1000)
    return model


def save_embeddings(image_dir):  #takes the directory within which you want to 
    """
    Takes a directory of images. Proccesses each image, then extracts its embeddings using the above
    extraction model. Saves each image uuid/images features in a dictionary as a json in current 
    working directory. 
    
    Args:
        image_dir: directory containing the images whos embeddings are to be extracted. 
    """
    extract = load_extraction_model()
    image_embeddings = {}
    with open("image_embeddings.json", "w") as fp:
        for id in os.listdir(image_dir):
            image = Image.open(f"{image_dir}/{id}")
            image = image_processor.proc_img(image)
            embedding = extract(image.float())
            image_embeddings[id] = embedding.tolist()
        json.dump(image_embeddings, fp)
        print ("dumped")


        
if __name__ == '__main__':
    save_embeddings("images")

