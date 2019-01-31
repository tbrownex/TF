import pandas as pd
import numpy  as np
import os
import sys
import time

import LSTM
import jobNumber as job
from sklearn.preprocessing import normalize

epochs        = [100]
batchSize     = [16]
L1size        = [100]
optimizer     = ['adam']
activation    = ["tanh"]

'''state_size    = [4]
LR          = [0.1]
Lambda      = [0]          # Regularization parameter'''

def getConfig():
    config={}
    config["dataLoc"] = "/home/tbrownex/data/test/"
    config["fileName"] = "seasonal.csv"
    config["testPct"] = 0.2
    config["segmentLength"] = 50
    return config

def getData(config):
    return pd.read_csv(config["dataLoc"]+config["fileName"])

def editParms():
    for x in activation:
        assert x in ['tanh','softmax','elu','selu','relu','sigmoid','linear'], "Invalid Activation: %s" % x
    for x in optimizer:
        assert x in ['sgd','adam','rmsprop','adagrad','adadelta','adamax','nadam'], "Invalid Optimizer: %s" % x
        
def prepData(df, segmentLength):
    ''' Make sure the number of entries in the file is a multiple of the segment Length  '''
    extra = df.shape[0] % segmentLength
    if extra > 0:
        df = df[:-extra]
    return np.array(df["value"])

def scaleData(arr):
    arr = np.reshape(arr, newshape=[-1,1])
    arr = normalize(arr, norm="l2", axis=0)
    arr = np.reshape(arr, newshape=[-1,])
    return arr

def formatKeras(df):
    ''' Incoming data is 1D. Put it into 3D for keras where:
         2nd and 3rd dimensions are a plane with shape [segmentLength, 1]
         Stack the planes to make the 1st dimension.
         Each plane is the previous plane shifted by 1 byte.
         Skim off the last element in each plane (byte 50) for the label (y), 
         leaving 49 elements behind
         '''
    segments = []
    offset = 0
    while offset < (df.shape[0] - config["segmentLength"]):
        segment = df[offset:offset+config["segmentLength"]]
        segment = np.reshape(segment, newshape=[config["segmentLength"], 1])
        segments.append(segment)
        offset += 1
    arr = np.array(segments)
    x = arr[:, :-1]
    y = arr[:, -1, [0]]
    return x, y

def getParms():
    # "parms" holds all the combinations of hyperparameters
    return [[a,b,c,d,e] for a in epochs
             for b in batchSize
             for c in L1size
             for d in optimizer
             for e in activation]

def splitData(X,Y):
    testCount = int(X.shape[0]*(config["testPct"]))
    trainX = X[:-testCount]
    trainY = Y[:-testCount]
    testX = X[-testCount:]
    testY = Y[-testCount:]
    return trainX, trainY, testX, testY

def writeRec(segment, count, parmDict, loss):
    ''' This file stores the results for each set of parameters so you can review a series
    of runs later'''
    rec = str(segment) +"|"+ str(count) +"|"+ str(parmDict['epochs'])+"|"+ str(parmDict['batchSize'])+\
    "|"+ str(parmDict['L1size'])+ "|"+ parmDict['optimizer']+"|"+ parmDict['activation']
    rec += "|"+ str(loss) +"\n"
    summary.write(rec)

def getLoss(p):
    parmDict['epochs']    = p[0]
    parmDict['batchSize'] = p[1]
    parmDict['L1size']    = p[2]
    parmDict['optimizer'] = p[3]
    parmDict['activation']= p[4]
    return LSTM.run(trainX, trainY, testX, testY, parmDict)

if __name__ == "__main__":
    config = getConfig()
    editParms()
    
    df = getData(config)
    arr = prepData(df, config['segmentLength'])
    arr = scaleData(arr)
    X,Y = formatKeras(arr)
    trainX, trainY, testX, testY = splitData(X,Y)
    
    jobId = job.getJob()
    parmList = getParms()
    
    parmDict = {}               # holds the hyperparameter combination for one run
    start_time = time.time()
    
    count = 1
    
    with open("/home/tbrownex/summary_"+str(jobId)+".csv", 'w') as summary:
        hdr = "SegmentLength|Run|Epochs|batchSize|L1size|optimizer|activation|Error" + "\n"
        summary.write(hdr)
        
        for p in parmList:
            loss = getLoss(p)
            #jobName = "job_" + jobId +"/"+ "run_" + str(count)
            writeRec(config["segmentLength"], count, parmDict, loss)
            count +=1
    
    jobId = int(jobId)
    job.setJob(jobId+1)
    print("Job {} complete after {:,.0f} minutes".format(str(jobId), (time.time() -start_time)/60))