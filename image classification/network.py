import tensorflow as tf

L3size = 384            # This comes from TF tutorial; seems arbitrary
L4size = 192            # This comes from TF tutorial; seems arbitrary
NUM_CLASSES = 10

def conv1(img):
    size    = 5          # This is the filter width/height
    depth   = 3         # This is the filter depth (matches input depth)
    filters = 64       # This is the number of filters (64 in the prod version)
    strides = 1
    padding = "SAME"
    
    weights = tf.Variable(tf.truncated_normal(shape=[size, size, depth, filters],
                                             stddev=0.3))
    bias   = tf.Variable(tf.truncated_normal([filters]))
    conv   = tf.nn.conv2d(input=img,
                          filter=weights,
                          strides=[1,strides,strides,1],
                          padding=padding)
    conv = tf.nn.bias_add(conv, bias)
    conv = tf.nn.relu(conv)
    return conv

# Input is the output of convL1
def pool1(L1):
    size    = 3
    strides = 2
    padding = "SAME"
    
    return tf.nn.max_pool(L1, ksize=[1, size, size, 1],
                          strides=[1, strides, strides, 1],
                          padding=padding)
# Input is the output of pool1 and has shape [batch, height, width, depth]. Depth = # of Filters
# Normalize by Feature aka Filter (along the depth dimension), across all samples in the batch
def bn1(L1):
    offset = 0
    scale  = 1
    variance_epsilon = 0.0001
    mean, variance = tf.nn.moments(L1, axes=[0, 1, 2])
    return tf.nn.batch_normalization(L1,
                                     mean,
                                     variance,
                                     offset,
                                     scale,
                                     variance_epsilon)

def conv2(L1out):
    size    = 5          # This is the filter width/height
    depth   = 64         # This is the filter depth (matches input depth)
    filters = 64       # This is the number of filters (64 in the prod version)
    strides = 1
    padding = "SAME"
    
    weights = tf.Variable(tf.truncated_normal(shape=[size, size, depth, filters],
                                             stddev=0.3))
    bias   = tf.Variable(tf.truncated_normal([filters]))
    conv   = tf.nn.conv2d(input=L1out,
                          filter=weights,
                          strides=[1,strides,strides,1],
                          padding=padding)
    conv = tf.nn.bias_add(conv, bias)
    conv = tf.nn.relu(conv)
    return conv
# Input has shape [batch, height, width, depth]. Depth = # of Filters
# Normalize by Feature aka Filter (along the depth dimension), across all samples in the batch
def bn2(L2):
    offset = 0
    scale  = 1
    variance_epsilon = 0.0001
    mean, variance = tf.nn.moments(L2, axes=[0, 1, 2])
    return tf.nn.batch_normalization(L2,
                                     mean,
                                     variance,
                                     offset,
                                     scale,
                                     variance_epsilon)
# Input is the output of bn2
def pool2(L2):
    size    = 3
    strides = 2
    padding = "SAME"
    
    return tf.nn.max_pool(L2, ksize=[1, size, size, 1],
                          strides=[1, strides, strides, 1],
                          padding=padding)
# Input is the output of L2. Need to flatten L2 first
# L2 looks like [batch, height, width, filters]
def fc1(L2out):    
    size = L2out.get_shape().as_list()
    size = size[1] * size[2] * size[3]
    L2resized = tf.reshape(L2out, [-1, size])
    
    weights = tf.Variable(tf.truncated_normal(shape=[size, L3size],
                                              stddev=0.3))
    bias   = tf.Variable(tf.truncated_normal([L3size]))
    return tf.nn.relu(tf.matmul(L2resized, weights) + bias)

# L3 looks like [batch, L3size]
def fc2(L3out):    
    weights = tf.Variable(tf.truncated_normal(shape=[L3size, L4size],
                                              stddev=0.3))
    bias   = tf.Variable(tf.truncated_normal([L4size]))
    return tf.nn.relu(tf.matmul(L3out, weights) + bias)

# L4 looks like [batch, L4size]
def final(L4out):
    weights = tf.Variable(tf.truncated_normal(shape=[L4size, NUM_CLASSES],
                                              stddev=0.3))
    bias   = tf.Variable(tf.truncated_normal([NUM_CLASSES]))
    return tf.nn.relu(tf.matmul(L4out, weights) + bias)