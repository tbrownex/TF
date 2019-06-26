import pandas as pd
import numpy  as np
import os
import sys
from tensorflow.contrib import learn

from preProcessor  import preProcess
from getConfig     import getConfig
from getModelParms import getParms
from nn            import runNN
from calcRMSE      import calcRMSE

def getData():
    return learn.datasets.load_dataset('boston')

def loadParms(x):
    ''' Load a dictionary with the hyperparameter combinations for one run '''
    d = {}
    d['l1Size']      = x[0]
    d['batchSize']   = x[1]
    d['lr']          = x[2]
    d["lambda"]      = x[3]
    d['std']         = x[4]
    d['dropout']     = x[5]
    d['dropoutTest'] = x[6]
    d['optimizer']   = x[7]
    return d

def calcMean(preds):
    ''' These are predictions over the same test data run repeatedly (one column for each run).
    Each run will produce a slightly different prediction due to "dropout" being specified. '''    
    mu    = np.mean(preds, axis=1)
    sigma = np.std(preds, axis=1)
    return mu, sigma

def evaluteSigma(actuals, mu, sigma):
    # What percent of the time was the actual within 1 or 2 StdDevs of the predicted?
    # "mu" is the mean of the predictions for a row
    Z = np.abs(actuals - mu) / sigma
    oneSigma = np.sum(Z<1) / Z.shape[0]
    twoSigma = np.sum(Z<2) / Z.shape[0]
    return np.round(oneSigma, decimals=3), np.round(twoSigma, decimals=3)

def evaluate(actuals, preds):
    ''' get the overall RMSE and see how accurate mean+sigma was '''
    mu, sigma = calcMean(preds)
    rmse = calcRMSE(actuals, mu)
    oneSigma, twoSigma = evaluteSigma(actuals, mu, sigma)
    return rmse, oneSigma, twoSigma

# This file stores the results for each set of parameters so you can review a series
# of runs later
def writeResults(results):
    with open("/home/tbrownex/results.csv", 'w') as output:
        keys = results[0][0].keys()
        hdr = ",".join(keys)
        hdr += ","+"rmse" ","+"1Sigma"","+"2Sigma"+ "\n"
        output.write(hdr)        
        
        for x in results:
            rec = ",".join([str(t) for t in x[0].values()])
            rec += ","+ str(x[1]) +","+ str(x[2]) +","+ str(x[3]) +"\n"
            output.write(rec)

if __name__ == "__main__":
    config = getConfig()
    bostonDataset = getData()
    dataDict = preProcess(bostonDataset)
    actuals = np.reshape(dataDict["testY"], newshape=[-1,])
    
    parms = getParms("NN")       # The hyper-parameter combinations to be tested
    results = []
    count = 0
    for x in parms:
        print(count)
        parmDict = loadParms(x)
        
        preds = runNN(dataDict, parmDict, config)
        
        rmse, oneSigma, twoSigma = evaluate(actuals, preds)
        tup = (parmDict, rmse, oneSigma, twoSigma)
        results.append(tup)
        count += 1
    writeResults(results)
    ''' # Write results for plotting
        results = np.stack((actuals, mu, sigma), axis=1)
        hdr = "actual,prediction,sigma"
        np.savetxt("/home/tbrownex/results.csv", results, delimiter=',', fmt='%.2f', header=hdr,comments='') '''