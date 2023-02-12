from PIL import Image
import os
import pandas as pd


final_size = 128


def clean_image_data (images_path):   #used if you want to clean all the images in one go
    if not os.path.exists ("cleaned_images"):
            os.makedirs ("cleaned_images")
    dirs = os.listdir(images_path)
    for n, item in enumerate(dirs[:], 1): 
        # print ("ID" + str (item))
        im = Image.open('images/' + item)
        new_im = resize_image(final_size, im)
        new_im.save(f'cleaned_images/{str(item)[:-4]}_resized.jpg')
        

def resize_image(final_size, im): #is currently called in main to resize images in the tunedresnet.get_item tp resize images on the go
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im


def clean_table_data(csv):
    prod_df = pd.read_csv(csv, lineterminator='\n')
    prod_df.rename(columns = {'id':'product_id'}, inplace = True)
    prod_df['price'] = prod_df['price'].str.strip('£')
    prod_df['price'] = prod_df['price'].str.strip(',')
    prod_df['category'] = prod_df['category'].apply(simplify_category)
    #print (prod_df['category'])
    return prod_df


def simplify_category(cat):
    #print (cat)
    try:
        cat = cat.split("/")
        return cat[0]
    except AttributeError:
        return None
 

def get_missing_image_ids(merged_data):
  missing_ids = []
  images = os.listdir("images")
  prod_ids = merged_data #pd.read_csv("/content/drive/My Drive/Coding/Images.csv")
  for ids in prod_ids['id']:
    if f"{ids}.jpg" not in images:
      missing_ids.append(f"{ids}")
  return missing_ids


def drop_missing_ids(merged_data):
    missing_ids = get_missing_image_ids(merged_data)
    for ids in missing_ids:
        row = merged_data.loc[merged_data['id'] == ids]
        index_to_drop = row.iloc[0,0]
        merged_data = merged_data.drop(index_to_drop)
    return merged_data


if __name__ == '__main__':
    # clean_image_data ("images/")
    clean_table_data("Products.csv")
