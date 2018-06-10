import tensorflow as tf
from timeit import default_timer as timer

dim_size = 200

with tf.device('/gpu:0'):
    arr1 = tf.random_uniform([dim_size,dim_size,dim_size],
                            minval=1, maxval=99,
                            dtype=tf.float32,
                            seed=None,
                            name='GPUarray1')
    arr2 = tf.random_uniform([dim_size,dim_size,dim_size],
                            minval=1, maxval=99,
                            dtype=tf.float32,
                            seed=None,
                            name='GPUarray2')
    g = tf.matmul(arr1, arr2, name='GPUmatmul')
    
with tf.device('/cpu:0'):
    arr3 = tf.random_uniform([dim_size,dim_size,dim_size],
                            minval=1, maxval=99,
                            dtype=tf.float32,
                            seed=None,
                            name='CPUarray1')
    arr4 = tf.random_uniform([dim_size,dim_size,dim_size],
                            minval=1, maxval=99,
                            dtype=tf.float32,
                            seed=None,
                            name='CPUarray2')
    c = tf.matmul(arr3, arr4, name='CPUmatmul')

gpu_times = []
cpu_times = []

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    for x in range(100):
        start = timer()
        tom = sess.run(g)
        gpu_times.append(timer() - start)
        
        start = timer()
        tom = sess.run(c)
        cpu_times.append(timer() - start)
        
print(sum(gpu_times))
print(sum(cpu_times))
t = input('')

# #### Get list of GPU devices
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()

t = input('')

# ******  for Jupyter Notebook  ******
'''# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b, name='Tom')
# Creates a session with log_device_placement set to True.
sess = tf.Session()

# Runs the op.
options = tf.RunOptions(output_partition_graphs=True)
metadata = tf.RunMetadata()
c_val = sess.run(c, options=options, run_metadata=metadata)

print(metadata)'''
