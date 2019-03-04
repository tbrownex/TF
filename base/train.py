import pandas as pd
import numpy  as np
import os
import sys
import time

from getArgs       import getArgs
from getConfig     import getConfig
from getData       import getData
from getModelParms import getParms
from preProcess    import preProcess
from kerasNN       import runNN
from evaluate      import evaluate

def loadParms(x):
    ''' Load a dictionary with the hyperparameter combinations for one run '''
    d = {}
    d['l1Size']      = x[0]
    d['activation']  = x[1]
    d['batchSize']   = x[2]
    d['lr']          = x[3]
    d['std']         = x[4]
    d['dropout']     = x[5]
    d['optimizer']   = x[6]
    return d

# This file stores the results for each set of parameters so you can review a series
# of runs later
def writeResults(results):
    delim = ","
    with open("/home/tbrownex/NNscores.csv", 'w') as summary:
        hdr = "L1"+delim+"activation"+delim+"batchSize"+delim+"LR"+\
        delim+"StdDev"+delim+"Dropout"+delim+"optimizer"+delim+"MAPE"+delim+"RMSE"+"\n"
        summary.write(hdr)
        
        for x in results:
            rec = str(x[0][0])+delim+str(x[0][1])+delim+str(x[0][2])+\
            delim+str(x[0][3])+delim+str(x[0][4])+delim+str(x[0][5])+\
            delim+str(x[0][6])+delim+str(x[1])+delim+str(x[2])+"\n"
            summary.write(rec)

def formatPreds(dataDict, preds):
    ''' "evaluate" module wants a DF with Predictions and the Label '''
    d = {}
    d["actual"] = dataDict["testY"]
    d["NN"]     = preds
    df = pd.DataFrame(d)
    return df

def processPreds(dataDict, preds):
    ''' Predictions were generated:
    - Format them for "evaluate"
    - get the error metrics from "evaluate"
    '''
    df = formatPreds(dataDict, preds)
    errors = evaluate(df, ensemble=False)
    mape = errors["NN"]["mape"]
    rmse = errors["NN"]["rmse"]
    print("{:<10.1%}{:.2f}".format(mape, rmse))
    return (mape, rmse)

if __name__ == "__main__":
    args   = getArgs()
    config = getConfig()

    df       = getData(config)
    df = df.sample(frac=0.3)
    dataDict = preProcess(df, config, args)
    
    parms = getParms("NN")       # The hyper-parameter combinations to be tested
    
    results = []
    count = 1
    
    start_time = time.time()
    print("\n{} parameter combinations".format(len(parms)))
    print("\n{:<10}{}".format("MAPE","RMSE"))
    
    for x in parms:
        parmDict = loadParms(x)
        
        preds = runNN(dataDict, parmDict, config)
        
        mape, rmse = processPreds(dataDict, preds)
        tup = (x, mape, rmse)
        results.append(tup)
        if count%10 == 0:
            print(count)
        
        count +=1
            
    # Write out a summary of the results
    writeResults(results)
    print("\ncomplete after {:,.0f} minutes".format((time.time() -start_time)/60))