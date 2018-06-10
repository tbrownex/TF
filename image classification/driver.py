import numpy as np
import tensorflow as tf
import loadData
import network
import time


  # norm1
    #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
def main():
    ds       = loadData.loadData()
    iterator = ds.make_one_shot_iterator()
    batch_images, batch_labels = iterator.get_next()
    
    L1    = network.conv1(batch_images)
    L1    = network.pool1(L1)
    L1out = network.bn1(L1)
    L2    = network.conv2(L1out)
    L2    = network.bn2(L2)
    L2out = network.pool2(L2)
    L3out = network.fc1(L2out)
    L4out = network.fc2(L3out)
    L5out = network.final(L4out)
    
    count = 0
    start_time = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            count += 1
            tom = sess.run(L5out)
            print(tom.shape)
            if count == 1: break
    print("Duration: {:,.0f} (minutes)".format((time.time() - start_time)/60))

if __name__ == "__main__":
    main()