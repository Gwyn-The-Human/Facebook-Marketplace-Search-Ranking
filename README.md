# Facebook-Marketplace-Search-Ranking

## Rational

Really excited to start this project! 


## Cleaning the data

Initially, cleaning invlovled converting prices in Products.csv by removing £ and , signs from the df. 
Images are resized to all be the same size. Also renames id to product_id in the Products.csv for easier merginging with Images.csv. 

As i progressed, i built up this module with some more functionality that I began to use pretty consistantly in other modules. I also realised there were a few discrepancies across my data sets, so I built the functions here to clean that up as well. 


## Desiging the model:
    
talk about the GPU and trying the google gpu, then changing number of labels, and cleaning up some code --=> va bene!! 



## The Feature Extraction Model

Once the tuned resnet model is futher trained the marketplace images, now i've got a model that can catagorise the images. 
BUT the end goal is to have a model that can give me vectors describing the images abstract features, rather than just telling me what catagory they fit into. 

For that reason I now need to edit the model from a classification model (between 435 catagories) into a a feature extraction model that will give a vector of 1000 numbers describing the given image. 

as a side note, this is AWESOME; that we can train the model with one set of catagories, and having learned from those, it is now able to tell us things about the images that are literally inarticulable to humans. WoWoWO!


## FAISS

Having extracted those features, I wrote a script to run the facebook AI similarity search on these features to find similar images. This was a really cool tool to learn about, and writing my own class to integrate the search into my own database and models I've made was a really great way to learn about what's happening under the hood. 
The final result is the FaissSearch class, which takes a numpy array of extracted features of an image, or of several sets of features for several images. For each set of features (e.g for each image) it returns a list of UUIDS of nearest similar images in the dataset. And these lists returned in a main list, so the end result is [[UUID 1][UUID 2]....[UUID n]].


## The API



## The docker build (app)

I moved the faiss_api and all its dependancies into the app directory, and built the docker file so that on being run it will run 

```uvicorn.run("faiss_api:api", host="0.0.0.0", port=8080)```