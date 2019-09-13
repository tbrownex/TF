from PIL import Image
import numpy as np
from preProcess    import prepareImage

def formatFile(file, config):
    image = Image.open(file)
    image = np.asarray(image)
    return prepareImage(image, config)