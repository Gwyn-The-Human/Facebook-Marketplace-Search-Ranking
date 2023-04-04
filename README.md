# Facebook-Marketplace-Search-Ranking

## Project overview
Welcome to my image search ranking using FAISS! Please take a look at my [ethics](https://github.com/Gwyn-The-Human/Data-Collection-Pipeline/blob/main/README.md) guide before using any of my code <3. 

This project builds a torch dataset from images, their uuids and associated catagories that they can be classified into - in this case the 13 parent catagories in facebook marketplace. 

From those images, the `resnet_training.py` trains a tuned resnet model on that dataset with those labels. Then the model is modified in `feature_extract.py` to return 1000 abstract features from the given image, rather than a prediction of its category. (SO cool that we can train the model on patterns (e.g images sorted into catagories), then change its output to access incomprehensible dimensions of those similarities wowo)

Using those features we can build an FAISS index. Then the `faiss_search.py`  module takes a given image, extracts its 1000 abstract features, runs the faiss search to find a given number of most similar images (in the context of the 13 catagories it was trained in!). The search is run in `faiss_api.py`, and it's all been containerised in a docker container. 


 ### Challenges
Learning these things quickly, the most challenging moments where getting a strong conceptual understanding of what my code was actually doing. For example, building the torch dataset was easy, but I it took me a bit longer to be satisfied that I understood what I had actually built. Playing around with it and researching helped build that understanding, and with it came confidence to be creative in how I thought about building the dataset into the rest of my code. Same is true of feature extraction; My biggest challenge was to not dissociate the code on my screen from the mechanisms I had read about that are happening underneath; but it didn't take me too long to get it, and overall a really rewarding experience! 


 ### Applications 
 As well as being used in the actual facebook marketplace search, these techniques have a pretty broad range of applications. It would be really easy to build this app from your own dataset, with your own categories. Coudl be used for any sort of image recognition, like facial recognition (see ethics guide!), finding landmarks/placespotting, searching for art, etc. 


## File Guide

- **app** - directory for building the docker image; almost all files here have counterparts in the repo outside of this file, with the same names. Some small modifications have been made to the files in app, so I've kept both them and their counterparts so a viewer (or future me!) can move back through the development proccess and understand it better.  

    - **final_model** - directory containing the weights used in the final model (see tuned_model)
        - `image_model.pt` - the model weights

    - `clean_data.py` - functions used for proccessing and cleaning data across the developement proccess. Identical to file of the same name in the reposiotory. 

    - `dockerfile` - used for the docker build. 

    - `faiss_api.py` - api for running the FAISS search on a given image. 

    - `faiss_search.py` - the class that actually runs the FAISS search. Called in `faiss_api.py`. 

    - `feature_extract.py` - this module was origionally used to extract features to generate the`image_embeddings.json`, a dictionary of uuid:[1000 abstract features] pairs for each image in the dataset. The features  are used for assesing similarity between queried images and the images in the dataset during the faiss search. 
    However, here the features have already been abstracted, and we only need the `load_extraction_model` method, caled in the `faiss_api.py`. 

    - `image_processor` - processes images so that they are the right shape for the FAISS search in `faiss_api.py`, and for feature extraction in `feature_extract.py`. 

    - `requirements.txt` - requirements for the docker build. 

    - `tuned_model.py` - defines the modified model; counterpart of `resnet_training.py` although a lot is hashed out; since we're loading the weights from **final_model** we don't need the training parts, just the model itself. 


- **final_model** - (same as a above) directory containing the weights used in the final model (see `tuned_model.py`)
    - `image_model.pt` - the model weights

- `clean_data.py` - functions used for proccessing and cleaning data across the developement proccess. Identical to file of the same name in the **app** directory. 

- `fb_dataset.py` - a custom dataset built using the torch Dataset class, made up of over 11000 images and their from facebook marketplace, their respective UUIDs and marketplace catagoires. 

- `feature_extract.py` - (same as above of the same name, but nothing hashed out.) Used to extract features to generate the`image_embeddings.json`. 

- `image_processor.py` -  (same as above) proccesses images for the FAISS search, and for feature extraction. 

- `README.md` - That's me!

- `resnet_training` - Where I modify and tune a resnet_50 model. Includes data splitting, model modification, training loop and evaluation. 


## Building and Cleaning the dataset

I built the dataset using the torch.data module. 

Initially, cleaning invlovled converting prices in Products.csv by removing £ and , signs from the df. 
Images are resized to all be the same size. Also renames id to product_id in the Products.csv for easier merginging with Images.csv. Later I realised that it is much more efficient and flexible to resize the images as they are passed to the model, rather than resizing all at once, so resize function is now called directly in the fb_dataset method __get_item_ as an image is 'got' from the dataset. 

As i progressed, i built up the clean_data module with some more functionality that I began to use pretty consistantly in other modules. I also realised there were a few discrepancies across my data sets, so I built the functions here to clean that up as well (get_missing_ids & drop_missing_ids). 


## Desiging the model:

I first built my own convolutional neural network from scratch, and made sure that was running ok; then I heard about tuning and resent, and decided to try implimenting that instead since I had already built an understanding of what's going on under the hood. So the final version was a tuned resnet model. 

Initially my training loop was waAyY too slow; and it turned out my graphics card was malfunctioning, so I experimented with google collab's free GPU to speed things up. When it didn't help I took another look at my code, reduced the number of labels in the origional training of the modified resnet from 435 (the number of unique catacagories in the dataset, (e.g Home & Garden / Dining, Living Room Furniture / Carpets & Flooring)) to 13 (the numer of main parent catagories (e.g Home & Gardening)). A couple of tweaks to the code and everything was running at a workable speed! 


### Choosing Hyperparams: 

Initially I only split Data in to training and testing, and for the model trained there I acheived a 70% accuracy with a relatively low learning rate (0.01). As I learned more about this proccess though I was a little suspicious of my number. Also, when I split my data properly, I was unable to replicate 70 % or close to it with the same hyperparameters, so I suspect some data leakage from my old train_test_split method. Currently my best model hits 53% accuracy accross 13 catagories which I'm happy with because I can replicate it and learned good practice throughout. I also switched optimisers from SDG to AdamW, which noticably improved accuracy. 

[loss and accuracy of three learning rates; 0.0004(light orange), 0.0005(pink), 0.001(dark orange)](Tensorboard_screenshot.png)


 Some techniques I'd like to play with to improve this number are:
1. [LR stepping](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)  
2. [Cosine Annealing](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)  
3. [L2 Normalisation](https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_preparing_data.htm)
4. more playing with batch sizes. 


## The Feature Extraction Model

Once the tuned resnet model is futher trained on the marketplace images, now i've got a model that can catagorise the images. 
BUT the end goal is to have a model that can give me vectors describing the images abstract features, rather than just telling me what catagory they fit into. 

For that reason I now need to edit the model from a classification model (between 13 catagories) into a a feature extraction model that will give a vector of 1000 numbers describing the given image. 

This is AWESOME; that we can train the model with one set of catagories, and having learned from those, it is now able to tell us things about the images that are literally inarticulable to humans. WoWoWO!


## FAISS

Having extracted those features, I wrote a script to run the facebook AI similarity search (FAISS) on these features to find similar images. This was a really cool tool to learn about, and writing my own class to integrate the search into my own database and models I've made was a really great way to learn about what's happening under the hood. 
The final result is the FaissSearch class, which takes a numpy array of extracted features of an image, or of several sets of features for several images. For each set of features (e.g for each image) it returns a list of UUIDS of nearest similar images in the dataset. And these lists returned in a main list, so the end result is [[UUID 1][UUID 2]....[UUID n]].


## The API

I built an API using Fastapi, and it was really cool capability to add to my toolkit. The api runs on Uvicorn, and runs the FAISS search using the models I designed and trained, on an indexed set of images' UUIDs and extracted features to find a number of the most similar images to the queried image. Cool stuff! 
Once the api is running, I went to http://localhost:8080/docs#/default/predict_image_predict_image_post to use the search. 


## The docker build (app)

I moved the faiss_api and all its dependancies into the app directory, and built the docker file so that on being run it will run 

```uvicorn.run("faiss_api:api", host="0.0.0.0", port=8080)```


