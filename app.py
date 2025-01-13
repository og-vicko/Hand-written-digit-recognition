import io
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from preprocess import data_pipeline
from cnn_model import CNN 
from PIL import Image


app = FastAPI()
model = CNN()

# class Image(BaseModel):
#     image: str

@app.post("/")
async def pred_digits(image: UploadFile = File(...)): # Use UploadFile to handle file uploads and ensure files are uploaded
    # Preprocess and make prediction on the image
    try:
        # Read the uploaded image file
        image_bytes = await image.read()  # Read image bytes
        # Convert image bytes to PIL image
        pil_image = Image.open(io.BytesIO(image_bytes)) 
        # print("debug::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        # print(pil_image)

        np_image = np.array(pil_image) # Convert PIL image to numpy array for processing (we are using cv2 packages)
        # Preprocess and predict
        predicted_class = data_pipeline(np_image)  # Preprocess the image and make prediction
        
        # Return prediction
        return {"prediction": predicted_class}
    except Exception as e:
        return {"error": str(e)}    