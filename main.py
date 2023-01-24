from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import fb_dataset
import os
import torch 
import torch.nn as nn

print (torch.cuda.is_available())


#TODO add rbg / greyscale in getitem w/inj dataset when relevant (see ivan's tips)
#makesure biases and weights tensors have requires_grad = true! 
#TODO test label decoder


class TunedResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = nn.Linear(2048,435) 
        

    def forward(self, X):
        return self.resnet50(X)


#hyperparams 
learning_rate = 0.01
batch_size = 3
num_epochs = 10

#variables 
dataset = fb_dataset.ImageDataset()
model = TunedResnet()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #customise batch size; will be number of groups of outputs returned in one run
tt = iter(train_loader)
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
writer = SummaryWriter()

#train/test split
training_samples = 3360#3360 is 80 % total 4202 elements (len(train_model))
testing_samples = 842 #842 is remaining 20%; this is where I split test and training data. 

#training loop

def train(model):
    #name model file for evaluation
    model_name = f"BS={batch_size}:E={num_epochs}:LR={learning_rate}.pt" 
    if not os.path.exists (f"model_evaluation/{model_name}"):
        os.makedirs (f"model_evaluation/{model_name}")
    batch_id = 0

    #train
    for epoch in range(num_epochs):
        print ("training")
        batch = next(tt) #gives first iteration
        while batch_id <= training_samples: # 80% of the total 4202 elements 
            images, labels = batch
            print ("SHAPE IS ")
            print(images.shape)
            pred = model(images)
            loss = nn.CrossEntropyLoss()
            loss = loss (pred, labels)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('loss', loss.item(), batch_id)
            batch_id = batch_id + 1
            batch = next(tt)
            if batch_id % 10 == 0:
                 print (batch_id) 
        if epoch % 10 == 0:
            print (f"LOSS IS: {loss.item()}")
        torch.save(model.state_dict(), f"model_evaluation/{model_name}/{epoch}.pt")


def evaluation():
    with torch.no_grad():
        batch_id = 0
        num_correct = 0
        num_samples = 0
        batch = next(tt)
        while batch_id <= testing_samples:
            images, labels = batch
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            num_samples += labels.shape[0]
            num_correct += (predictions == labels).sum().item()
            batch_id += 1
            batch = next(tt)
            accuracy = 100 * num_correct / num_samples
            print (f"Correct: {num_correct} out of {num_samples}")
            print(f"Accuracy: {accuracy}")



if __name__ == '__main__':
    train(model)
    evaluation()


