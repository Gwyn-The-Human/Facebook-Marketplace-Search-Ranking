from PIL import Image
import os
import pandas as pd

final_size = 128


def clean_image_data (images_path):   #used if you want to clean all the images in one go
    """
    For each image in the directory of images at images_path, resizes the image to given size and saves it in a new
    folder called cleaned_images.
    
    Args:
        images_path: the path to the folder containing the images you want to clea
    """
    if not os.path.exists ("cleaned_images"):
            os.makedirs ("cleaned_images")
    dirs = os.listdir(images_path)
    for n, item in enumerate(dirs[:], 1): 
        # print ("ID" + str (item))
        im = Image.open('images/' + item)
        new_im = resize_image(final_size, im)
        new_im.save(f'cleaned_images/{str(item)[:-4]}_resized.jpg')
        

def resize_image(final_size, im): #is currently called in main to resize images to 128*128 in the tunedresnet.get_item tp resize images on the go
    """
    Takes an image and resizes it to a given size, while maintaining the aspect ratio of the image
    
    Args:
        final_size: the size of the image you want to resize to
        im: the image to be resized
        
    Returns:
        the image in the form of a tensor.
    """
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im


def clean_table_data(csv):
    """
    Given a csv, renames the column 'id' to 'product_id', removes the £ and , from the price
    column, and simplifies the category column name. 
    
    Args:
        csv: the path of the csv file you want to clean

    Returns: 
        A dataframe with the columns: product_id, name, price, category
    """
    prod_df = pd.read_csv(csv, lineterminator='\n')
    prod_df.rename(columns = {'id':'product_id'}, inplace = True)
    prod_df['price'] = prod_df['price'].str.strip('£')
    prod_df['price'] = prod_df['price'].str.strip(',')
    prod_df['category'] = prod_df['category'].apply(simplify_category)
    #print (prod_df['category'])
    return prod_df


def simplify_category(cat):
    """
    Splits a given string on the "/" character, and returns the first element of the resulting
    list
    
    Args:
        cat: the all the nested categories of the product

    Returns:
        The first category of the list of  nested categories.
    """
    try:
        cat = cat.split("/")
        return cat[0]
    except AttributeError:
        return None
 

def get_missing_image_ids(merged_data):
    """
    Takes a dataframe as an argument, and returns a list of product ids that are missing from the
    images folder.
    
    Args:
    merged_data: This is the dataframe that contains the product ids and the image urls
    
    Returns:
        A list of missing image ids
    """
    missing_ids = []
    images = os.listdir("images")
    prod_ids = merged_data #pd.read_csv("/content/drive/My Drive/Coding/Images.csv")
    for ids in prod_ids['id']:
        if f"{ids}.jpg" not in images:
            missing_ids.append(f"{ids}")
    return missing_ids


def drop_missing_ids(merged_data):
    """
    Drops the rows that in given dataframe that contain images that are missing from the images directory.
    
    Args:
        merged_data: the dataframe that contains the merged data

    Returns:
        A dataframe with the rows that have missing images dropped.
    """
    missing_ids = get_missing_image_ids(merged_data)
    for ids in missing_ids:
        row = merged_data.loc[merged_data['id'] == ids]
        index_to_drop = row.iloc[0,0]
        merged_data = merged_data.drop(index_to_drop)
    return merged_data






if __name__ == '__main__':
    # clean_image_data ("images/")
    clean_table_data("Products.csv")
