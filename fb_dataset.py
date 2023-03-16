from PIL import Image
from torch.utils.data import Dataset
import clean_data
import pandas as pd
import torch 
import torchvision.transforms as transforms



class ImageDataset (Dataset):
    def __init__(self):
       #clean and merge tabular data
        self.prod_data = clean_data.clean_table_data("Products.csv")
        self.image_ids = pd.read_csv("Images.csv")
        self.merged_data = self.image_ids.merge(self.prod_data[['category','product_id']], on='product_id') 
        self.merged_data = clean_data.drop_missing_ids(self.merged_data)
        
        #sets labels and image 
        self.labels = self.merged_data['category'].to_list()
        self.image_files = self.merged_data['id'].to_list()

        #encode labels
        self.encoded_labels = {}
        self.encode_labels(self.merged_data)


    def encode_labels(self, merged_data): #435 labels total previously, changed to 13 to speed up training. 
        full_catagories = merged_data['category'].unique()
        # print (full_catagories)
        # print (len(full_catagories))
        for cat in enumerate (full_catagories):
            self.encoded_labels[cat[1]] = cat [0]

    
    def decode_label(self, label_index):
        return self.encoded_labels[label_index]
            

    def __getitem__(self,index): 
        """
        returns a tuple of features and labels.
        """
        label = self.labels[index]
        encoded_label = torch.tensor(self.encoded_labels[label])
        image = self.image_files[index]
        #PIL_image = Image.open(f"cleaned_images/{image}_resized.jpg")
        PIL_image = Image.open(f"images/{image}.jpg")
        PIL_image = clean_data.resize_image(clean_data.final_size, PIL_image)
        transform = transforms.PILToTensor() 
        feature = transform(PIL_image).float()
        return feature, encoded_label


    def __len__(self):
        return len (self.merged_data)


    