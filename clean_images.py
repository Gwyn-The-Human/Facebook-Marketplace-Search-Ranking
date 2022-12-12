from PIL import Image
import os
import pandas as pd
import urllib.request


def clean_image_data (images_path):
    if not os.path.exists ("cleaned_images"):
            os.makedirs ("cleaned_images")
    dirs = os.listdir(images_path)
    final_size = 512
    for n, item in enumerate(dirs[:], 1): 
        # print ("ID" + str (item))
        im = Image.open('images/' + item)
        new_im = resize_image(final_size, im)
        new_im.save(f'cleaned_images/{str(item)[:-4]}_resized.jpg')
        

def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im


if __name__ == '__main__':
    clean_image_data ("images/")


# The image dataset that you have has contains multiple images of the products.
# Take a look at the images, do they have the same size? The same number of channels? 
# If not, then you need to change that so that they are all consistent.

# Create a file named clean_images.py in your repository.

# In this file, you will write code to clean the image dataset.

# Create a pipeline that will apply the necessary cleaning to the image dataset by defining a function called clean_image_data.

# It should take in a filepath to the folder which contains the images, then clean them and save them into a new folder called "cleaned_images".

# You can use this file  to get started. make sure you make it work for your dataset and you give the desired image size.

