import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from keras import Input
from keras.layers import Dropout
from keras.layers import Dense
from keras import Model

def buildLayers(parmDict, shape):
    inputs = tf.keras.Input(shape=(shape,))
    inter = layers.Dropout(parmDict["dropout"])(inputs, training=True)
    inter = layers.Dense(parmDict["l1Size"], activation='relu', kernel_regularizer=l2(parmDict["lambda"]))(inter)
    inter = layers.Dropout(parmDict["dropout"])(inter, training=True)
    inter = layers.Dense(parmDict["l1Size"], activation='relu', kernel_regularizer=l2(parmDict["lambda"]))(inter)
    inter = layers.Dropout(parmDict["dropout"])(inter, training=True)
    predictions = layers.Dense(1, kernel_regularizer=l2(parmDict["lambda"]))(inter)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    return model

def buildNetwork(parmDict, shape):
    nn = buildLayers(parmDict, shape)
    if parmDict["optimizer"] == "SGD":
        opt  = tf.keras.optimizers.SGD(lr=parmDict["lr"],\
                                       decay=1e-6,\
                                       momentum=0.0,
                                       nesterov=True)
    elif parmDict["optimizer"] == "Adam":
        opt = tf.keras.optimizers.Adam(lr=parmDict["lr"],\
                                       beta_1=0.9,\
                                       beta_2=0.999,\
                                       epsilon=None,\
                                       decay=1e-6,\
                                       amsgrad=False)
    nn.compile(optimizer=opt, loss="mse")
    return nn

def fitNetwork(dataDict, parmDict, nn, config):
    TB        = keras.callbacks.TensorBoard(log_dir=config["TBdir"])
    '''modelSave = keras.callbacks.ModelCheckpoint(config["modelDir"],
                                                save_weights_only=True,
                                                verbose=1)'''
    
    X = np.array(dataDict["trainX"])
    Y = np.array(dataDict["trainY"])
    nn.fit(X, Y,\
           batch_size=parmDict["batchSize"],\
           epochs=40,\
           validation_split=None,\
           verbose=0,\
           shuffle=False,
           callbacks=[TB])

def tutorial(dataDict):
    reg = 1e-3
    
    inputs = Input(shape=(dataDict["trainX"].shape[1],))
    inter = Dropout(0.05)(inputs, training=True)
    inter = Dense(50, activation='relu', W_regularizer=l2(reg))(inter)
    inter = Dropout(0.05)(inter, training=True)
    inter = Dense(50, activation='relu', W_regularizer=l2(reg))(inter)
    inter = Dropout(0.05)(inter, training=True)
    outputs = Dense(1, W_regularizer=l2(reg))(inter)
    model = Model(inputs, outputs)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(dataDict["trainX"], dataDict["trainY"], batch_size=128, nb_epoch=40, verbose=0)
    return model
def runNN(dataDict, parmDict, config):
    book = True
    if book:
        model = tutorial(dataDict)
        preds = model.predict(dataDict["testX"], batch_size=500, verbose=1)
        return np.reshape(preds, newshape=[-1,])
    '''data: dictionary holding Train, Validation and Test sets'''
    shape = dataDict["trainX"].shape[1]
    nn = buildNetwork(parmDict, shape)
    fitNetwork(dataDict, parmDict, nn, config)
    tf.keras.models.save_model(model=nn, filepath=config["modelDir"]+"NNmodel.h5")
    preds = nn.predict(dataDict["testX"])
    preds = np.reshape(preds, newshape=[-1,])
    return preds