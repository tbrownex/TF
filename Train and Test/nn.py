import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import sys
import time

TB_DIR   = '/home/tbrownex/TF/TensorBoard'         # where to store Tensorboard data
SAVE_DIR = "/home/tbrownex/TF/checkpoints/"

def run(data, parms, job_name):
    '''data: dictionary holding Train and Validation sets'''
    # get the true response rate to compare to my predictions
    true_rr = np.sum(data['val_labels'][:,-1]) / data['val_labels'].shape[0]
    
    feature_count = data['train_x'].shape[1]
    num_classes   = np.unique(data['train_labels']).shape[0]
    # Load hyper-parameters
    L1         = parms['l1_size']
    LR         = parms['learning_rate']
    LAMBDA     = parms['lambda']
    WEIGHT     = parms['weight']
    BATCH      = parms['batch_size']
    EPOCHS     = parms['epochs']
    ACTIVATION = parms['activation']
    STD        = parms["std"]
    # Set up the network
    tf.reset_default_graph()
    x  = tf.placeholder("float", shape=[None, feature_count], name="input")
    y_ = tf.placeholder("float", shape=[None, num_classes])

    l1_w     = tf.Variable(tf.truncated_normal([feature_count, L1], stddev=STD, dtype=tf.float32, seed=1814))
    l1_b     = tf.Variable(tf.truncated_normal([1,L1], dtype=tf.float32))
    
    if   ACTIVATION == 'tanh':
        l1_act = tf.nn.tanh(tf.matmul(x,l1_w) + l1_b)
    elif ACTIVATION == 'leakyReLU':
        l1_act   = leakyReLU(x, l1_w, l1_b)
    elif ACTIVATION == 'ReLU':
        l1_act   = tf.nn.relu(tf.matmul(x,l1_w) + l1_b)
    elif ACTIVATION == 'ReLU6':
        l1_act   = tf.nn.relu6(tf.matmul(x,l1_w) + l1_b)
        
    l2_w   = tf.Variable(tf.truncated_normal([L1,num_classes], stddev=STD, dtype=tf.float32, seed=1814))
    l2_b   = tf.Variable(tf.truncated_normal([1,num_classes]))

    l2_out = tf.add(tf.matmul(l1_act, l2_w), l2_b, name="L2")
    
    # Cost function
    entropy   = tf.losses.softmax_cross_entropy(y_, l2_out)
    '''entropy   = tf.nn.weighted_cross_entropy_with_logits(targets=y_, logits=l2_out,
                                                        pos_weight=WEIGHT)'''
    L1_layer1 = LAMBDA**tf.reduce_sum(tf.abs(l1_w))
    L2_layer1 = LAMBDA*tf.nn.l2_loss(l1_w)
    L2_layer2 = LAMBDA*tf.nn.l2_loss(l2_w)
    
    cost = tf.reduce_mean(entropy + L1_layer1 + L2_layer1 + L2_layer2)
    
    # Optimizer
    optimize = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)
    
    '''# Evaluate
    # count the number of True Positives in the top 20%
    K      = tf.cast(tf.shape(y_), tf.float32)
    pct    = tf.constant(0.2)              # Top 20%
    K      = tf.cast(tf.scalar_mul(pct,K[0]), tf.int32)
    val, idx = tf.nn.top_k(l2_out[:,-1],
                           k=K,
                           sorted=False)   # faster not to sort
    topK  = tf.reduce_sum(tf.gather(y_[:,1], idx))
    lift  = topK / tf.cast(K, tf.float32) / true_rr -1 '''
        
    training_cost = tf.summary.scalar('Training cost', cost)
    val_cost      = tf.summary.scalar('Validation cost', cost)
    '''val_lift      = tf.summary.scalar('lift', lift)'''
    merged = tf.summary.merge_all()

    # Run
    TB_counter = 1                    # For TensorBoard
    num_training_batches = int(len(data['train_x']) / BATCH)
    '''print('{} epochs of {} iterations with batch size {}'.format(EPOCHS,num_training_batches,BATCH))'''
    
    saver = tf.train.Saver()
    
    #CP = tf.ConfigProto( device_count = {'GPU': 1} )
    #sess = tf.Session(config=CP)
    
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(TB_DIR + '/' + job_name, sess.graph)
    sess.run(tf.global_variables_initializer())
    
    val_score_best = early_stop_counter = 0
    
    for i in range(EPOCHS):
        if early_stop_counter > 5000:
            print("early stop")
            break
        print("Epoch ", i)
        a,b = shuffle(data['train_x'],data['train_labels'])
        costList = []
        for j in range(num_training_batches):
            x_mini = a[j*BATCH:j*BATCH+BATCH]
            y_mini = b[j*BATCH:j*BATCH+BATCH]
            _, tom = sess.run([optimize, cost], feed_dict = {x: x_mini, y_: y_mini})
            if j% 80 == 0:
                tc = sess.run(training_cost,
                              feed_dict = {x: x_mini,
                                           y_: y_mini})
                train_writer.add_summary(tc, TB_counter)
                costList.append(tom)
                TB_counter += 1
                #vc, vl = sess.run([val_cost, val_lift],
                '''vc = sess.run(val_cost,
                                  feed_dict = {x: data['val_x'],
                                               y_: data['val_labels']})'''
                #train_writer.add_summary(vc , TB_counter)
                #train_writer.add_summary(vl , TB_counter)
            
                
            '''if j % 10 == 0:
                s, val_score = sess.run([merged, lift],
                                        feed_dict = {x:  data['val_x'],
                                                     y_: data['val_labels']})
                train_writer.add_summary(s, TB_counter)
                if val_score > val_score_best * 1.1:
                    val_score_best = val_score
                    early_stop_counter = 0
                else:
                    if val_score_best > 2.0:
                        early_stop_counter += 1'''
                
    '''_lift = sess.run(lift, feed_dict = {x:  data['val_x'],
                                        y_: data['val_labels']})
    print("Validation lift: ", _lift)'''
    #Assuming res is a flat list
    with open("/tmp/costs.csv", "w") as output:
        import csv
        writer = csv.writer(output, lineterminator='\n')
        for val in costList:
            writer.writerow([val])
    
    saver.save(sess, SAVE_DIR+'SS_'+job_name )
    train_writer.close()
    #return _lift
    return
