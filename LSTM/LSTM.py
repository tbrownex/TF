import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
from calcMAPE import calcMAPE
from numpy import newaxis

def addLayers(model, width, parms):
    input_dim = 1
    return_seq = False
    use_bias = True
    bias_initializer="tom"
    
    model.add(LSTM(units=parms["L1size"],
                   activation=parms["activation"],
                   input_shape=(width, input_dim),
                   return_sequences=return_seq))
    model.add(Dense(1, activation="linear"))
    return model

def fitModel(model, trainX, trainY, parms):
    m = model.fit(
        trainX,
        trainY,
        verbose=0,
        epochs= parms["epochs"],
        batch_size = parms["batchSize"])
    return m.history["loss"][0]

def getPredictions(model, testX, testY):
    # The idea here is to get an initial segment to use for prediction. testX[0] is one segmentLength
    # After that, replace the testX with predicted values, one at a time
    testSeg = testX[0]
    testSeg = testSeg[newaxis,:,:]    # Needs to be 3D

    # Make the predictions
    predictions = []
    for _ in range(testY.shape[0]):
        pred = model.predict(testSeg, batch_size=1)[0][0]
        predictions.append(pred)
        # Shift the test segment one to the left and replace the rightmost value with the prediction
        testSeg = np.roll(testSeg, -1)
        testSeg[0,-1,0] = pred
    return predictions

def run(trainX, trainY, testX, testY, parms):
    width = trainX.shape[1]
    model = Sequential()
    model = addLayers(model, width, parms)
    model.compile(loss="mse", optimizer=parms["optimizer"])
    fitModel(model, trainX, trainY, parms)
    predictions = getPredictions(model, testX, testY)

    # calculate the error
    n = 20
    return calcMAPE(testY[:n,0], np.array(predictions[:n]))