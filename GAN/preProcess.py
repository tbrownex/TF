import numpy as np
import cv2

'''def decodeRawBytes(rawBytes):
    img = cv2.imdecode(np.asarray(bytearray(rawBytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img '''

def prepareImage(img, config):
    ''' For each image:
    - crop
    - resize/shrink
    '''
    cropY = config["cropY"]
    cropX = config["cropX"]
    img = img[cropY:-cropY, cropX:-cropX]
    img = cv2.resize(img, (config["dimX"], config["dimY"]))
    return img