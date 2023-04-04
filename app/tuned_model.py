
#from sklearn.model_selection import train_test_split
#from torch.utils.data import Subset
#from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
#import fb_dataset
#import os
import torch 
import torch.nn as nn


#building the docker file we only need the model, hence everything else is hashed out

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


# #train-test split function
# def train_test_datasets(dataset, val_split=0.2):
#     train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
#     train_dataset =  Subset(dataset, train_idx)
#     test_dataset = Subset(dataset, val_idx)
#     return train_dataset, test_dataset



# #hyperparams 
# learning_rate = 0.01
# batch_size = 128
# num_epochs = 10

# #variables 
# dataset=None #
# dataset = fb_dataset.ImageDataset()
# model = TunedResnet()
# train_ds, test_ds = train_test_datasets(dataset)
# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True) #customise batch size; will be number of groups of outputs returned in one run
# test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True) #customise batch size; will be number of groups of outputs returned in one run
# #test_tt = iter(test_loader) #I Dont need these anymore
# #tt = iter(train_loader)
# optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
# writer = SummaryWriter()


# #training loop
# def train(model):
#     #name model file for evaluation
#     model_name = f"BS={batch_size}:LR={learning_rate}" 
#     if not os.path.exists (f"model_evaluation/{model_name}"):
#         os.makedirs (f"model_evaluation/{model_name}")


#     #train
#     for epoch in range(num_epochs):
#         batch_id = 0
#         print ("training")
#         for batch in train_loader:
#             images, labels = batch
#             #print(images.shape)
#             pred = model(images)
#             loss = nn.CrossEntropyLoss()
#             loss = loss (pred, labels)
#             print(loss.item())
#             loss.backward()
#             optimiser.step()
#             optimiser.zero_grad()
#             writer.add_scalar('loss', loss.item(), batch_id)
#             batch_id = batch_id + 1
#             if batch_id % 10 == 0:
#                  print (batch_id) 
#         if epoch % 10 == 0:
#             print (f"LOSS IS: {loss.item()}")
#         torch.save(model.state_dict(), f"model_evaluation/{model_name}/{epoch}.pt")




# def evaluation():
#     with torch.no_grad():
#         batch_id = 0
#         num_correct = 0
#         num_samples = 0
#         for batch in test_loader:
#             images, labels = batch
#             outputs = model(images)
#             _, predictions = torch.max(outputs, 1)
#             num_samples += labels.shape[0]
#             num_correct += (predictions == labels).sum().item()
#             batch_id += 1
#             accuracy = 100 * num_correct / num_samples
#             print (f"Correct: {num_correct} out of {num_samples}")
#             print(f"Accuracy: {accuracy}")



# if __name__ == '__main__':


#     evaluation()
#     state_dict = torch.load("model_evaluation/BS=128:E=10:LR=0.01.pt/5.pt")
#     model.load_state_dict(state_dict)
#     evaluation()






    
