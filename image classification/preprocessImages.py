''' CIFAR10 data consists of 6 binaries. The binaries hold 10,000 images each. Convert them to tfrecords for the tensorflow data pipeline

images are distorted via cropping and flipping, both with a random factor
10 distorted images are produced for every source image
Images are shuffled as well (I didn't think the TF batch shuffle was very thorough)'''

import numpy as np
import os
import tensorflow as tf
from random import shuffle
import distortImage as distort

DATA_DIR = '/home/tom/Dropbox/data/ML/cifar10/'
OUTPUT   = "preProcessed.tfrecords"

ROWS = 32
COLS = 32
CHANNELS = 3               # RGB
NUM_IMAGES = 10000
CROP_ROWS = 24
CROP_COLS = 24

image_size = ROWS*COLS*CHANNELS+1      # 1 byte for the label

def getFiles(dir):
    ''' Return a list of all the ".bin" files in the data folder. These hold all the images and labels '''
    files = os.listdir(dir)
    binaries = []
    for file in files:
        if file.endswith(".bin"):
            binaries.append(file)
    
    print("Files to be processed are:")
    for x in binaries: print(x)
        
    return binaries

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def loadData(dir, fileList):
    ''' Return a list of all the images and labels.
    Each row in the file is an image. The first byte of the record is the identifier
    (label)
    Each element will be distorted 10 times, so 10x as many images will be produced
    Images are 32x32x3'''
    dataList = []
    
    for f in fileList:
        input_file = np.fromfile(dir+f, dtype='uint8')
        for x in range(NUM_IMAGES):
            rec = input_file[x*image_size:(x+1)*image_size]
            label = int(rec[0])
            data = rec[1:]
            image = np.transpose(np.reshape(data,(CHANNELS,ROWS,COLS)), (1,2,0)).tostring()
            for _ in range(10):
                distorted = distort.distort(image, [ROWS,COLS,CHANNELS], [CROP_ROWS,CROP_COLS])
                data = (distorted, label)
                dataList.append(data)
    return dataList

def writeTFrecords(data):
    ''' each list element is a tuple: (image, label) '''
    writer = tf.python_io.TFRecordWriter(DATA_DIR+OUTPUT)
    
    for x in data:
        image = x[0]
        label = x[1]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'height': _int64_feature(CROP_ROWS),
                    'width':  _int64_feature(CROP_COLS),
                    'depth':  _int64_feature(CHANNELS),
                    'label':  _int64_feature(label),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))}))
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()

def main(argv=None):
    fileList = getFiles(DATA_DIR)
    dataList = loadData(DATA_DIR, fileList)
    shuffle(dataList)    # Extra shuffling prior to TF batch shuffle
    writeTFrecords(dataList)
    print("\n", "tfrecords file created")

if __name__ == '__main__':
    main()