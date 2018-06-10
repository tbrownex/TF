import tensorflow as tf
import numpy      as np
from timeit import default_timer as timer
from tensorflow.python.client import timeline

#files  = tf.train.match_filenames_once("/home/tom/data/ball bearings/1st_test/")
with tf.device('/gpu:0'):
    files  = tf.train.match_filenames_once("/home/tom/data/rectangles/rectangles-images_val.txt")
    file_q = tf.train.string_input_producer(files, num_epochs=1)

    reader     = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_q)

# For the "rectangles" data set, 8x28 columns of pixel values + the last
# column is the label: 0 or 1
    defaults = [[0.] for x in range(785)]
    T = tf.decode_csv(value, record_defaults=defaults)

gpu_times = []
cpu_times = []

print('\n', '\n')

# "k" is the filename; "v" the value; "r" the record
def view_data(k,v,t):
    print('\n', '\n')
    print('Key:        {}'.format(k))
    print('\n', '\n')
    #print('Value:      {}'.format(v))
    #print('Parsed rec: {}'.format(t))
    #print('\n')

x = 0

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    start = timer()
    threads = tf.train.start_queue_runners(coord=coord)
    print('\n', '\n')
    while reader.num_records_produced().eval() < 3:
        k,v,t = sess.run([key,value,T])
        x += 1
        view_data(k,v,t)
    coord.request_stop()
    '''try:
        while not coord.should_stop():
            #sess.run(train_op)
            x += 1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
        print('load_gpu: {:.4f}'.format(timer() - start))        
    finally:
        coord.request_stop()
        print('*********************')
        print(x)
        #coord.join(threads)    '''
print('Finished after {} records'.format(x))