import pandas as pd
import numpy  as np
import os
import sys
import time

#from simpleRNN   import run
import simpleRNN
import jobNumber as job

data_length = 1000000
Y_offset    = 3

num_batches = [5000]
minibatch   = [5]
state_size  = [4]
epochs      = [1]
LR          = [0.1]

Lambda      = [0]          # Regularization parameter
activation  = ["tanh", "ReLU"]     # 'tanh' 'leakyReLU' 'ReLU' 'relu6' 'elu' 'crelu'

def editParms():
    '''Check parameters make sense'''
    for x in num_batches:
        assert data_length % x == 0, "data_length must be a multiple of num_batches"
        for y in minibatch:
            assert data_length / x % y == 0, "data width must be a multiple of minibatch"

def genData():
    X = np.array(np.random.choice(2, size=(data_length,)))
    Y = []
    for i in range(data_length):
        if X[i-Y_offset] == 1:
            Y.append(np.random.choice(2, p=[.2, .8]))
        else:
            Y.append(np.random.choice(2, p=[.5, .5]))
    return X, np.array(Y)

def writeResults(results, job_id):
    ''' This file stores the results for each set of parameters so you can review a series
    of runs later'''
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
    editParms()
    data = genData()
    job_id = job.getJob()
    for x in activation:
        assert x in ['tanh', 'leakyReLU', 'ReLU', 'relu6'], "Invalid Activation: %s" % x
        
    # "parms" holds all the combinations of hyperparameters
    parms = [[a,b,c,d,e,f] for a in num_batches
             for b in minibatch
             for c in state_size
             for d in epochs
             for e in LR
             for f in activation]
    
    parm_dict = {}               # holds the hyperparameter combination for one run
    results = []                 # holds the hyperparemeters and results for each run
    start_time = time.time()
    
    loop = count = 1
    for x in parms:
        for i in range(loop):
            parm_dict['num_batches'] = x[0]
            parm_dict['minibatch']   = x[1]
            parm_dict['state_size']  = x[2]
            parm_dict['epochs']      = x[3]
            parm_dict['LR']          = x[4]
            parm_dict['activation']  = x[5]
            job_name = "job_" + job_id +"/"+ "run_" + str(count)
            
            loss = simpleRNN.run(data, parm_dict, job_name)
            
            tup = (count, parm_dict, loss)
            results.append(tup)
            count +=1
    
    # Write out a summary of the results
    writeResults(results, job_id)
    job_id = int(job_id)
    job.setJob(job_id+1)
    print("Job {} complete after {:,.0f} minutes".format(str(job_id), (time.time() -start_time)/60))