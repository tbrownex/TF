''' Produce 10 .png files which you can view. Also print the label associated with each
image so you can confirm it's the right label '''

import numpy as np
import tensorflow as tf
from PIL import Image

DATA_DIR = '/home/tom/Dropbox/data/ML/cifar10/'
FILENM   = 'preProcessed.tfrecords'
category = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

def main():
    print("{:<15}{}".format("Picture #", "Label"))
    count = 0
    
    record_iterator = tf.python_io.tf_record_iterator(path=DATA_DIR+FILENM)

    for string_record in record_iterator:
        count += 1
        example = tf.train.Example()
        example.ParseFromString(string_record)
        height = int(example.features.feature["height"].int64_list.value[0])
        width  = int(example.features.feature["width"].int64_list.value[0])
        depth  = int(example.features.feature["depth"].int64_list.value[0])
        label = (example.features.feature["label"].int64_list.value[0])
        image = (example.features.feature["image"].bytes_list.value[0])
    
        image   = np.fromstring(image, dtype=np.uint8)
        image   = image.reshape((height, width, depth))
                
        img = Image.fromarray(image, 'RGB')
        img.save("/home/tom/Dropbox/" + str(count) +'.png')
        print("{:<15}{}".format(count, category[label]))
        if count ==10: break
    
if __name__ == "__main__":
    main()