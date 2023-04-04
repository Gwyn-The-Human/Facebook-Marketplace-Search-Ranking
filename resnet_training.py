from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import fb_dataset

import os
import torch 
import torch.nn as nn
import torch.utils.data 
import numpy as np


np.random.seed(2)


class TunedResnet(nn.Module):
    def __init__(self):
        """
        Loads the pretrained resnet50 model from torchhub and replaces the last layer with a linear layer
        with 13 outputs.
        """
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = nn.Linear(2048,13) 
        

    def forward(self, X):
        """
        Takes in an image, and returns the activations of the last layer of the modified ResNet50
        model
        
        Args:
            X: input tensor of shape (N, C, H, W)

        Returns: 
            The output of the resnet50 model.
        """
        return self.resnet50(X)
    

    def predict(self, image):
        """
        Takes an image as input, and returns the predicted class of that image
        
        Args:
            image: the image to be classified
        Returns:
            The output of the forward pass of the model.
        """
        with torch.no_grad():
            x = self.forward(image)
            return x


#train-test split function
def split_datasets(dataset, val_split=0.3):
    """
    Takes a dataset and a proportion by which to split the dataset, and returns two datasets.
    Used for splitting datasets into train/test/validation datasets
    
    Args:
        dataset: The dataset to split
        val_split: The proportion of the dataset to be used for validationor testing
    
    :return:
        A tuple of two datasets, one for training and one for validation or testing.
    """
    main_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    main_dataset =  Subset(dataset, main_idx)
    split_dataset = Subset(dataset, val_idx)
    return main_dataset, split_dataset


#hyperparams 
learning_rate = 0.0004
batch_size = 128
num_epochs = 16

#variables 
dataset = fb_dataset.ImageDataset()
model = TunedResnet()

train_ds, test_ds = split_datasets(dataset)
test_ds, validation_ds = split_datasets(test_ds, val_split=0.5)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size) 
validation_loader = DataLoader(validation_ds, batch_size=batch_size)

optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
writer = SummaryWriter()


def train(model, model_tag):
    """
    Iterates over the training data for the given number of epochs, and for each batch, calculates the loss,
    backpropagate the loss, and update the weights. Writes this information to TensorBoard for visualisations. 
    Saves weights every 2 epochs.
    
    Args:
        model: the model we want to train
        model_tag: the tag generate from create_evaluation_dir. Used to save model weights. 
    """
    batch_id = 0
    for epoch in range(num_epochs):
        print (f"Epoch {epoch}")
        for batch in train_loader:
            images, labels = batch
            pred = model(images)
            loss = nn.CrossEntropyLoss()
            loss = loss (pred, labels)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('train loss / batch id', loss.item(), batch_id) #Writes to TensorBoard
            batch_id = batch_id + 1
            if batch_id % 10 == 0:
                 print (f"batch {batch_id}") 
        #save weights and evaulate accuracy every 2 epochs         
        evaluation(epoch)
        torch.save(model.state_dict(), f"model_evaluation/{model_tag}/{epoch}.pt")


#evaluation
def evaluation(epoch):
    """
    With gradient calculation disabledIterates through the validation set, and for each batch, 
    gets predictions and compares them to labels. 
    Prints percentage of correct predictions, and writes total percentage to tensorboard
    
    Args:
        epoch: the current epoch number
    """
    with torch.no_grad():
        batch_id = 0
        num_correct = 0
        num_samples = 0
        for batch in validation_loader:
            images, labels = batch
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            num_samples += labels.shape[0]
            num_correct += (predictions == labels).sum().item()
            batch_id += 1
            accuracy = 100 * num_correct / num_samples
            writer.add_scalar('Validation Accuracy / Epoch', accuracy, epoch) #writes to tensorboard
            print (f"Correct: {num_correct} out of {num_samples}")
            print(f"Accuracy: {accuracy}")


def create_evaluation_dir():
    """
    Creates a directory and associated model tag to save model weights during training.
    
    Returns:
        model_tag: a string that id's a model by its hyperparameters
    """
    now = datetime.now()
    model_tag = f"{now}:BS={batch_size}:LR={learning_rate}" 
    if not os.path.exists (f"model_evaluation/{model_tag}"):
        os.makedirs (f"model_evaluation/{model_tag}")
    return model_tag


if __name__ == '__main__':

    # model_tag = create_evaluation_dir()
    # train(model, model_tag)
    state_dict = torch.load("final_model/image_model.pt")
    model.load_state_dict(state_dict)
    evaluation(0)

    pass