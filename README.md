# Facebook-Marketplace-Search-Ranking

## Rational

Really excited to start this project! 


## Cleaning the data

So far cleaning has invlovled converting prices in Products.csv by removing £ and , signs from the df. 
Images are resized to all be the same size. Also renames id to product_id in the Products.csv for easier merginging with Images.csv. 


## Desiging the model:
    

## The Feature Extraction Model

Once the tuned resnet model is futher trained the marketplace images, now i've got a model that can catagorise the images. 
BUT the end goal is to have a model that can give me vectors describing the images abstract features, rather than just telling me what catagory they fit into. 

For that reason I now need to edit the model from a classification model (between 435 catagories) into a a feature extraction model that will give a vector of 1000 numbers describing the given image. 

as a side note, this is AWESOME; that we can train the model with one set of catagories, and having learned from those, it is now able to tell us things about the images that are literally inarticulable to humans. WoWoWO!

