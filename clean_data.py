from PIL import Image
import os
import pandas as pd


final_size = 512


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
    return prod_df


if __name__ == '__main__':
    clean_image_data ("images/")
