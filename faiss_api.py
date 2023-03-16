import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
from faiss_search import FaissSearch
import image_processor
import feature_extract            
  
# try:
#     class FeatureExtractor(nn.Module): 
#         def __init__(self,
#                     decoder: dict = None):
#             super(FeatureExtractor, self).__init__()
#             self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
#             self.resnet50.fc = nn.Linear(2048,1000)
#             #self.decoder = decoder # which decoder? 

#         def forward(self, image):
#             x = self.resnet50(image)
#             return x

#         def predict(self, image):
#             with torch.no_grad():
#                 x = self.forward(image)
#                 return x
# except:
#     raise OSError("No feature extraction model found. Check that you have the the model in the correct location")


try:
    class FAISSQuery(BaseModel):
        num_results: int 
        image : bytes
except:
    raise OSError("No pydantic class found.")

                                   
try:
    print ("loading model")    
    model = feature_extract.load_extraction_model() 
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")


api = FastAPI()
print("Starting server")

@api.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  return {"message": msg}


@api.post('/predict/image')
def predict_image(image: UploadFile, num_results : int): # FAISSQuery = Depends()): # work out depends, then try sending the image in testo to the API and see what happens!! 
    #also, so the above UploadFile is (like) a pydantic class, removed  = File(...), after uploadFIle
    
    #return JSONResponse(content={"Nearest similar image UUIDS" : " bababababa <3"}) #testing
    pil_image = Image.open(image.file)
    print ("imaged opened")
    proccessed_image = image_processor.proc_img(pil_image)
    features = model.predict(proccessed_image.float()).numpy()
    print ("Features extracted")
    faiss = FaissSearch() 
    print (" FAISS initialised")
    result_index = faiss.search(features, num_results)
    result_uuids = faiss.decode_search_result(result_index)
    return JSONResponse(content={"Nearest similar image UUIDS" : result_uuids})
        



#so my pseudo code will be

#we take an image in binary (worry about that later I guess)
#convert to a pil
#proccess to tensor
#use feature extration model on it
#then run the FAISS search



#BE VERBOSE hence those try statements 
#try a local build first for docker and then try the 

#might want a post that just takes an image, returns features. 
  
#image needs to binary with open file, write binary as f, teh contents you get when you do binary 


#going to use the model to extract teh features 
#model predict 
#   app.post:#read binary as f
# uploads the File
# reads it
# write that file to a local image File FILEIFILE
# open it as a pil image

# --> convert to tensor, pass to feature extractor
# then use output features 
# then delete that FILEFILEFILE

    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)




    # TODO                                                       
    # Process the input and use it as input for the image model  
    # image.file is the image that the user sent to your API     
    # Apply the corresponding methods to compute the category    
    # and the probabilities    


    #     content={
    # "Category": "", # Return the category here
    # "Probabilities": "" # Return a list or dict of probabilities here
    #     })
  
    #lets try starting with running the FAISS search and returning similar images, and then I can 