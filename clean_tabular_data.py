
import pandas as pd
import torch
import numpy as np


def clean_table_data(csv):
    prod_df = pd.read_csv(csv, lineterminator='\n')
    prod_df = torch.from_numpy
    prod_df.rename(columns = {'id':'product_id'}, inplace = True)
    prod_df['price'] = prod_df['price'].str.strip('£')
    prod_df['price'] = prod_df['price'].str.strip(',')
    return prod_df


