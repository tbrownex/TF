import tensorflow as tf
from sklearn.utils import shuffle

EPOCHS = 1
BATCHSIZE = 128
BUFFERSIZE = 60000     # The number of records to include in shuffle
FILENM = '/home/tom/Dropbox/data/ML/cifar10/preProcessed.tfrecords'

# Return the image and label from a record in tfrecords file
def parse(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    features = {'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)
    image_raw = parsed_example['image']
    image = tf.decode_raw(image_raw, tf.uint8)
    image = tf.cast(image, tf.float32)    # Need the image to be float
    
    image = tf.reshape(image, [24, 24, 3])
    label = parsed_example['label']
    return image, label

def loadData():
    dataset = tf.data.TFRecordDataset(filenames=FILENM)
    dataset = dataset.map(parse)
    dataset = dataset.repeat(EPOCHS)
    dataset = dataset.batch(BATCHSIZE)
    dataset = dataset.shuffle(buffer_size=BUFFERSIZE)
    return dataset
    #iterator = dataset.make_one_shot_iterator()
    #images_batch, labels_batch = iterator.get_next()
    
def getBatch():
    return images_batch, labels_batch
    #i, l = shuffle(batch_i, batch_l)