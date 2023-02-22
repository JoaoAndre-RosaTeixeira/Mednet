import torch
import torchvision as tv
from PIL import Image, ImageOps
import numpy as np
import os

# Define the model
model = torch.jit.load("saved_model.pt", map_location=torch.device('cpu'))
model.eval()  # Set the model to inference mode
    

def scaleImage(y):          # Pass a PIL image, return a tensor

    if(y.min() < y.max()):  # Assuming the image isn't empty, rescale so its values run from 0 to 1
        y = (y - y.min())/(y.max() - y.min()) 
    z = y - y.mean()        # Subtract the mean value of the image
    return z


def predict_img(img):
    img = Image.open(img)
    #Transform to gray (1 channel)
    img = ImageOps.grayscale(img)
    #Resize the img
    size = 64, 64
    img = img.resize(size)
    #Convert to a matrix 
    toTensor = tv.transforms.ToTensor()
    img = toTensor(img)
    img.shape
    #Reshape for the model
    img = img.reshape([1,1,64,64])
    img=scaleImage(img) 
    classname = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']
    yOut = model(img)
    max_value, max_index = yOut.max(1)
    pred = classname[max_index.item()]
    return pred, max_index.item()
