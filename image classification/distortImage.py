# Crop the 2-d part of a 3-d image (don't crop the RGB layer)
# Input is a "bytes" object, as from a .bin file (CIFAR10)
# "dims" would be [32,32] and cropSize might be [24,24]
import numpy as np

def distort(image, dims, cropSize):
    assert type(image) is bytes, "image type must be bytes"
    image = np.frombuffer(image, dtype=np.uint8)  # convert bytes to int
    image = np.reshape(image, dims)
    image = crop(image, dims, cropSize)
    image = flip(image, dims)
    return image.tobytes("C")
            
def crop(image, dims, cropSize):
    diffH = dims[0] - cropSize[0]
    diffW = dims[1] - cropSize[1]
    offsetH = np.random.randint(1,diffH)
    offsetW = np.random.randint(1,diffW)
    
    cropped = image[offsetH:offsetH+cropSize[0],
                    offsetW:offsetW+cropSize[1],
                    :]
    return cropped
# randomly flip the image Left-Right 50% of the time
def flip(image, dims):
    if np.random.randint(0,2) == 1:
        image = np.fliplr(image)
    return image