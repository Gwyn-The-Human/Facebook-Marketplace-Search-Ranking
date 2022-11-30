
import pandas as pd

prod_df = pd.read_csv("Products.csv", lineterminator='\n')

prod_df['price'] = prod_df['price'].str.strip('£')
prod_df['price'] = prod_df['price'].str.strip(',')
print (prod_df['price'])