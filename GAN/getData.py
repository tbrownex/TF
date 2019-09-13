import os
import pandas as pd
import numpy as np
from formatFile import formatFile

__author__ = "Tom Browne"

def getImageFiles(config):
    # Return a list of all the .jpg filenames of faces
    imageFiles = []
    
    rootFolder = config["dataLoc"]+"images/"
    for root, dirs, files in os.walk(rootFolder):
        for file in files:
            if file.endswith(".jpg"):
                filename = os.path.join(root, file)
                imageFiles.append(filename)
    return imageFiles

def getData(config):
    '''
    There are .jpg files scattered in thousands of folders. Images are returned in a numpy array
    There are attributes that correspond to each of the images in a separate file, separate folder (Week4Faces) but we don't need them
    for this assignent
    '''
    imageFiles = getImageFiles(config)
    
    images = []
    for file in imageFiles:
        image = formatFile(file, config)
        images.append(image)
            
    images = np.stack(images).astype('uint8')    
    images = images.astype('float32') / 255.0
    return images