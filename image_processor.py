import clean_data
import torch
import torchvision.transforms as transforms


def proc_img(image): #takes an image of shape (n_channels, hight, width)
    clean_image = clean_data.resize_image(clean_data.final_size, image)#apply transformations from clean_images; expects image = Image.open(<file>) but maybe shoudl be a tensor??  
    transform = transforms.PILToTensor() 
    image_tesnor = transform(clean_image)
    processed_image = torch.reshape(image_tesnor, [1, 3, 128, 128])
    return processed_image

