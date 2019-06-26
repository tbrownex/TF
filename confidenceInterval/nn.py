import numpy      as np
import pandas     as pd
import tensorflow as tf
from sklearn.utils import shuffle
import sys
import time

''' Trying to duplicate the code found here:
        https://github.com/yaringal/DropoutUncertaintyExps/blob/master/net/net.py
'''

EPOCHS = 30000

def runNN(dataDict, parmDict, config):
    '''data: dictionary holding Train, Validation and Test sets'''
    featureCount = dataDict['trainX'].shape[1]
    
    # Load hyper-parameters
    L1         = parmDict['l1Size']
    LR         = parmDict['lr']
    LAMBDA     = parmDict["lambda"]
    BATCH      = parmDict['batchSize']
    STD        = parmDict["std"]
    DROP       = parmDict["dropout"]
    DROPTEST   = parmDict["dropoutTest"]
    
    # Set up the network
    tf.reset_default_graph()
    x  = tf.placeholder("float", shape=[None, featureCount], name="input")
    y_ = tf.placeholder("float", shape=[None,1])
    dropoutRate = tf.placeholder(tf.float32)

    l1_w = tf.Variable(tf.truncated_normal([featureCount, L1], stddev=STD, dtype=tf.float32))
    l1_b = tf.Variable(tf.truncated_normal([1,L1], dtype=tf.float32))
    l2_w = tf.Variable(tf.truncated_normal([L1,1], stddev=STD, dtype=tf.float32))
    l2_b = tf.Variable(tf.truncated_normal([1,1]))
    
    l1_out   = tf.nn.tanh(tf.matmul(x,l1_w) + l1_b)
    l1_drop  = tf.nn.dropout(l1_out, rate=dropoutRate)
    l2_out   = tf.add(tf.matmul(l1_drop, l2_w), l2_b, name="L2")
    
    l1_reg = tf.nn.l2_loss(l1_w)
    l2_reg = tf.nn.l2_loss(l2_w)
    
    # Cost function
    err = tf.sqrt(tf.reduce_mean(tf.squared_difference(y_, l2_out)))
    cost = err + LAMBDA * (l1_reg + l2_reg)
    
    # Optimizer
    optimize = tf.train.AdagradOptimizer(learning_rate=LR).minimize(cost)

    # Run
    num_training_batches = int(len(dataDict['trainX']) / BATCH)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    lastRMSE = np.inf
    threshhold = 1e-3
    for i in range(EPOCHS):
        a,b = shuffle(dataDict['trainX'], dataDict['trainY'])
        for j in range(num_training_batches):
            x_mini = a[j*BATCH:j*BATCH+BATCH]
            y_mini = b[j*BATCH:j*BATCH+BATCH]
            _ = sess.run(optimize, feed_dict = {x: x_mini,
                                                y_: y_mini,
                                                dropoutRate: DROP})
        if i % 1000 == 0:
            e = sess.run(err, feed_dict = {x: dataDict['testX'],
                                           y_: dataDict['testY'],
                                           dropoutRate: 0.0})
            if np.abs(e - lastRMSE) < threshhold:
                break
            else:
                lastRMSE = e
    
    loop = 200
    predList = []
    for n in range(loop):
        preds = sess.run(l2_out, feed_dict = {x: dataDict['testX'],
                                              y_: dataDict['testY'],
                                              dropoutRate: DROPTEST})
        predList.append(preds)
    preds = np.asarray(predList)
    preds = np.squeeze(preds.T)
    return preds