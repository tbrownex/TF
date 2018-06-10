# #### https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html

import numpy      as np
import tensorflow as tf

def viewTrainable(sess):
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print ("Variable: ", k)
        print ("Shape: ", v.shape)
        print (v)
    _=input('')

def shapeData(data, rnn):
    '''data comes in as one long sequence: put it into batches'''
    X, Y = data
    X = np.reshape(X, (rnn.num_batches, -1))
    Y = np.reshape(Y, (rnn.num_batches, -1))
    return X,Y

class RNN(object):
    num_classes  = 2

    def __init__(self, parms):        
        self.num_batches = parms['num_batches']
        self.minibatch   = parms['minibatch']
        self.state_size  = parms['state_size']
        self.LR          = parms['LR']
        self.activation  = parms['activation']

    def weights(self):
        self.W  = tf.Variable(tf.truncated_normal([self.state_size, self.minibatch],  stddev=0.6, dtype=tf.float32), name="W")
        self.U  = tf.Variable(tf.truncated_normal([self.state_size, self.state_size], stddev=0.6, dtype=tf.float32), name="U")
        self.b1 = tf.Variable(tf.truncated_normal([self.state_size, 1], stddev=0.1, dtype=tf.float32), name="b1")
    
        self.V  = tf.Variable(tf.truncated_normal([self.minibatch, self.state_size], stddev=0.6, dtype=tf.float32), name="V")
        self.b2 = tf.Variable(tf.truncated_normal([self.num_classes], stddev=0.1, dtype=tf.float32), name="b2")
    
    def updateState(self, data, state):
        return tf.matmul(self.W, data)+tf.matmul(self.U,state) + self.b1

    def logits(self, states):
        self.logits      = [tf.matmul(self.V, state) + self.b2 for state in states]
        self.predictions = [tf.nn.softmax(logit) for logit in self.logits]
    
    def losses(self, y_minis):
        self.losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit)
           for logit, label in zip(self.logits, y_minis)]
        self.total_loss = tf.reduce_mean(self.losses)
        #self.optimize   = tf.train.AdagradOptimizer(self.LR).minimize(self.total_loss)
        self.optimize   = tf.train.AdamOptimizer(self.LR).minimize(self.total_loss)

def run(data, parms, job_name):
    rnn    = RNN(parms)
    epochs = parms['epochs']
    X,Y    = shapeData(data, rnn)
    # This is how many groups we break the batch into
    num_steps  = int(len(X[0]) / rnn.minibatch)
    print("{} batches, {} steps of width {}".format(rnn.num_batches, num_steps, rnn.minibatch))
    
    rnn.weights()
    
    x = tf.placeholder(tf.int32, [None], name='input_placeholder')
    y = tf.placeholder(tf.int32, [None], name='labels_placeholder')
       
    # Turn our x placeholder into a list of one-hot tensors
    x_1hot = tf.one_hot(x, rnn.num_classes)

    # rnn_inputs holds a single batch where each element is a tensor of shape [minibatch, num_classes]
    x_minis = []
    y_minis = []
    
    for i in range(num_steps):
        x_minis.append(x_1hot[i*rnn.minibatch:(i+1) * rnn.minibatch])
        y_minis.append(y     [i*rnn.minibatch:(i+1) * rnn.minibatch])
        
    init_state = tf.zeros([rnn.state_size, rnn.num_classes])
    state      = init_state
    
    states = []
    for x_mini in x_minis:
        state = rnn.updateState(x_mini, state)
        states.append(state)
    #final_state = states[-1]
    
    rnn.logits(states)
    rnn.losses(y_minis)
    
    loss_counter = 100
    
    #training_state = np.zeros((rnn.state_size, rnn.num_classes))
    
    viewLoss     =True
    viewVariables=False
    viewPreds    =False
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if viewVariables:
            viewTrainable(sess)
        for e in range(epochs):
            loss_subtotal = 0
            state = init_state
            for b in range(rnn.num_batches):
                op1, _ = sess.run([rnn.total_loss,
                                   rnn.optimize],
                                   feed_dict={x:X[b], y:Y[b]})
                if viewLoss:
                    loss_subtotal += op1
                    if b % loss_counter ==0 and b > 0:
                        print("batch ", b, " loss:",loss_subtotal/loss_counter)
                        loss_subtotal = 0
                if viewPreds:
                    for p in c:
                        sigma = sum(np.argmax(p, axis=1))
                        ones_count += sigma
            if viewVariables:
                viewTrainable(sess)
            if viewPreds:
                print("Epoch: {}  Number of ones: {}".format(e,ones_count))
    return op1