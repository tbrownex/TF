import tensorflow as tf
import numpy  as np
import pandas as pd
import sys
import time
import glob
from pathlib import Path

from getDataDirectory import getDir

FILENM        = 'test_CAWA.csv'

def getData():
    data_dir = getDir(client="SS", typ="ML")
    df       = pd.read_csv(data_dir+FILENM, sep="|")
    df       = df.sample(frac = 1.0)
    
    # Save the HH IDs before removing
    hh = df['hh_num']
    del df['hh_num']
    
    # Remove a Label column if there is one
    try:
        del df['Label']
    except:
        pass
    
    # Drop any columns with NAN values
    before = df.shape[0]
    df     = df.dropna(axis=0, how='any')
    after  = df.shape[0]
    print("{}Deleted {:,.0f} rows out of {:,.0f} due to NAN".format(
            "\n",(before-after), before))
    return df, hh

# These are the saved models, one model per set of training parameters
def getModel():
    job = input("Enter the JOB id of the saved model:")
    run = input("Enter the RUN id of the saved model:")
    
    job_name = "SS_job_" + str(job) + "/"
    run_id   = "run_"+str(run)
    saveDir  = getDir(client="SS", typ="TB")
    model    = saveDir+job_name+run_id
    
    return model 
    
def getPredictions(model, data):
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(model+".meta")
    sess  = tf.Session()
    saver.restore(sess, model)

    predictions = sess.run("L2:0", feed_dict={"input:0": data})
    
    sess.close()
    return predictions 

def saveScores(hh, predictions):
    hh = hh.values.reshape([-1,1])
    merged = np.concatenate ((hh, predictions), axis=1)
    # Sort by "most Positive"
    idx = np.argsort(merged[:,2])          # Last columns is L2_out positive
    sort = merged[idx]
    np.savetxt(str(Path.home())+'/L2.csv', merged, delimiter="|", header="HH_ID|Neg|Pos", comments="")
    return

def sortPredictions(hh, predictions):
    ix = np.argsort(predictions, axis=0)[::-1]
    result = hh[ix[:,1]]
    result.to_csv(str(Path.home())+'/tom_scored.csv', index=False)
    return

    # The use of the dictionary is only to match what's done in "Train"
if __name__ == "__main__":
    df, hh = getData()
    data   = df.as_matrix()
    model  = getModel()

    predictions = getPredictions(model, data)
                  
    saveScores(hh, predictions)