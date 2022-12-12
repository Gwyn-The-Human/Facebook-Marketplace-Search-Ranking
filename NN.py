import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np



class Cnn(torch.nn.Module): 
    def __init__(self):
        #initialise parameters
        super().__init__()
        self.layer1 = torch.nn.Linear(786432, 250) #features, outputs; how do i check how many I need? Can i just run it? 
        #activation function
        self.layer2 = torch.nn.Linear(250, 200)
        self.layer3 = torch.nn.Linear(200, 100)
        self.layer4 = torch.nn.Linear(100,1)

    def forward (self,features): #replaces __call__  (this is inherited from the nn.module!)
        
        x = self.layer1 (features)
        x = F.relu(x)

        x = self.layer2(x)
        x = F.relu(x)

        x = self.layer3(x)
        x = F.relu(x)

        x = self.layer4(x)       
        return x








# class linear_NN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.input_fc = nn.Linear(input_dim, 250)
#         self.hidden_fc = nn.Linear(250, 100)
#         self.output_fc = nn.Linear(100, output_dim)

#     def forward(self, features):

#         #x = [batch size, height, width]
#         batch_size = features.shape[0]
#         features = features.view(batch_size, -1) #basically reshapes tensor to shape(batch_size, -1) = [3, 786756871623 or wtvr]

#         # x = [batch size, height * width]
#         h_1 = F.relu(self.input_fc(features)) #in other words its taking a batch of 3 full image tensors all in one tensor

#         # h_1 = [batch size, 250]
#         h_2 = F.relu(self.hidden_fc(h_1))

#         # h_2 = [batch size, 100]
#         y_pred = self.output_fc(h_2)

#         # y_pred = [batch size, output dim]
#         return y_pred


# #compare the forward methods here
# #understand hashes
