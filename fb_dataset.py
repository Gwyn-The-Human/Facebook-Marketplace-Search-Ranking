from PIL import Image
from torch.utils.data import Dataset
import clean_data
import pandas as pd
import torch 
import torchvision.transforms as transforms



class ImageDataset (Dataset):
    def __init__(self):
        """
        Cleans and extracts relevant tabular data, encodes image catagories into integers. 
        
        Upon initialising the class, this method merges the useful tabular data from 'catagory' and 'product_id'
        columns, then cleans the data using drop_missing_ids method from the clean_data module.
        Then generates lists of the image uuids and of the total catagories that they will be classified into. 
        Lastly calls the method encode_labels on the merged tabular data to encode each catagory as a number. 

        """
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


    def encode_labels(self, merged_data): 
        """
        Encodes the labels of the data set and saves them to the self.encoded variable.
        
        Args:
            merged_data: the dataframe that contains the data that we want to encode
        """
        full_catagories = merged_data['category'].unique()
        # print (full_catagories)
        # print (len(full_catagories))
        for cat in enumerate (full_catagories):
            self.encoded_labels[cat[1]] = cat [0]

    
    def decode_label(self, label_index):
        return self.encoded_labels[label_index]
            

    def __getitem__(self,index): 
        """
        Takes in an index, and returns a tuple of the image and the label at that index of the dataset.
        
        Args:
            index: the index of the image in the dataset
        Returns:
            The feature and encoded label at that index of the dataset. 
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


    