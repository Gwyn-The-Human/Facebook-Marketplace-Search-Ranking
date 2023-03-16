import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import UploadFile
import torch
import torch.nn as nn
from pydantic import BaseModel
from faiss_search import FaissSearch
import image_processor
import feature_extract            
  
                                    
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
def predict_image(image: UploadFile, num_results : int):# UploadFile is (like) a pydantic class, removed  = File(...), after uploadFIle
    pil_image = Image.open(image.file)
    print ("imaged opened")
    proccessed_image = image_processor.proc_img(pil_image)
    features = model.predict(proccessed_image.float()).numpy() #faiss needs wants numpy, NOT tensor
    print ("Features extracted")
    faiss = FaissSearch() 
    print ("FAISS initialised")
    result_index = faiss.search(features, num_results)
    result_uuids = faiss.decode_search_result(result_index)
    return JSONResponse(content={"Nearest similar image UUIDS" : result_uuids})

    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)



