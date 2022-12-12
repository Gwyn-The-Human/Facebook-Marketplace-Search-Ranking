from torch.utils.data import Dataset, DataLoader
import torch 
import numpy as np
import clean_images
import torchvision 
import os
from PIL import Image
import pandas as pd


x = torch.tensor ([[1,2,3],[1,2,3],[1,2,3], [1,2,3]])
print (x.shape)
print(x.view (12,-1).shape)
print(x.view (12,1).shape)

# image_df = pd.read_csv ("Images.csv")
# prod_df = pd.read_csv ("Products.csv", lineterminator='\n')

# prod_df.rename(columns = {'id':'product_id'}, inplace = True)
# df = image_df.merge(prod_df[['category','product_id']], on= 'product_id')
# print (df.columns)
# full_catagories = df['category'].unique()
# encoded_labels = {}

# for cat in enumerate (full_catagories):
#     encoded_labels[cat[0]] = cat [1]




#FOR LATER: 
# for x in full_catagories:
#     print (x.split('/'))
#OR
# df['col'].str.split('/',expand=True)


# merge them by id; and then for id

# --> get a an id

# used enumerate - gives a numeric value for each index