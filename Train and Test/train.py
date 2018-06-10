import pandas as pd
import numpy  as np
import os
import sys
import time

from scale_split_data import partition
from getDataDirectory import getDir
from nn               import run
import jobNumber      as job
from split_label      import splitY

TEST_PCT = 0
VAL_PCT  = 0.20
FILENM   = 'train_CAWA.csv'

# NN hyper-parameters
l1_size       = [4]          # Count of nodes in layer 1
learning_rate = [.0003]
Lambda        = [0]          # Regularization parameter
weight        = [300]              # Degree to which Positives are weighted in the loss function
batch_size    = [128]
epochs        = [4]
activation    = ['ReLU']           # 'tanh' 'leakyReLU' 'ReLU' 'relu6' 'elu' 'crelu'

def getData():
    data_dir = getDir(client="SS", typ="ML")
    df       = pd.read_csv(data_dir+FILENM, sep="|")
    df       = df.sample(frac = 1.0)
    # No need for the HH during Training
    del df['hh_num']
    # Drop any columns with NAN values
    before = df.shape[0]
    df     = df.dropna(axis=0, how='any')
    after  = df.shape[0]
    print("{}Deleted {:,.0f} rows out of {:,.0f} due to NAN".format(
            "\n",(before-after), before))
    return(df)

def prepareData():
    '''Read the training data file and create a dictionary with keys of "train_x", "train_labels",
    "val_x" and "val_labels"'''
    df = getData()
    train, val, _ = partition(df, VAL_PCT, TEST_PCT)
    data_dict     = splitY(train, val, None)
    return data_dict

def print_sales_ratio(data_dict):
    pos = np.sum(data_dict['train_labels'][:,1])
    tot = data_dict['train_labels'].shape[0]
    print("{}Training file:   {:,.0f} rows with {:,.0f} Positives  {}:1".format(
            "\n", tot, pos, int(tot/pos)))
    pos = np.sum(data_dict['val_labels'][:,1])
    tot = data_dict['val_labels'].shape[0]
    print("Validation file: {:,.0f} rows with {:,.0f} Positives  {}:1{}".format(
            tot, pos, int(tot/pos), "\n"))

# This file stores the results for each set of parameters so you can review a series
# of runs later
def writeResults(results, job_id):
    with open("/home/tom/summary_"+str(job_id)+".txt", 'w') as summary:
        keys = results[0][1]
        hdr = "Run" +"|" + "|".join(keys)
        hdr += "|"+"Lift" + "\n"
        summary.write(hdr)        
        
        for x in results:
            rec = str(x[0]) +"|"
            rec += "|".join([str(t) for t in x[1].values()])
            rec += "|"+ str(x[2]) +"\n"         # lift
            summary.write(rec)
            
if __name__ == "__main__":
    data_dict = prepareData()
    
    print_sales_ratio(data_dict)
    print("{}{}{}".format("\n", ".....pausing 10 seconds.....","\n"))
    time.sleep(10)
    
    job_id = job.getJob()
    for x in activation:
        assert x in ['tanh', 'leakyReLU', 'ReLU', 'relu6'], "Invalid Activation: %s" % x
        
    # "parms" holds all the combinations of hyperparameters
    parms = [[a,b,c,d,e,f,g] for a in l1_size
             for b in learning_rate
             for c in Lambda
             for d in weight
             for e in batch_size
             for f in epochs
             for g in activation]
    
    parm_dict = {}                  # holds the hyperparameter combination for one run
    results = []                    # holds the hyperparemeters and results for each run
    start_time = time.time()
    
    loop = count = 1
    for x in parms:
        for i in range(loop):
            parm_dict['l1_size']       = x[0]
            parm_dict['learning_rate'] = x[1]
            parm_dict['lambda']        = x[2]
            parm_dict['weight']        = x[3]
            parm_dict['batch_size']    = x[4]
            parm_dict['epochs']        = x[5]
            parm_dict['activation']    = x[6]
            job_name = "job_" + job_id +"/"+ "run_" + str(count)
            
            lift = run(data_dict, parm_dict, job_name)
            
            tup = (count, parm_dict, lift)
            results.append(tup)
            count +=1
    
    # Write out a summary of the results
    writeResults(results, job_id)
    job_id = int(job_id)
    job.setJob(job_id+1)
    print("Job {} complete after {:,.0f} minutes".format(str(job_id), (time.time() -start_time)/60))