import tensorflow as tf
import numpy  as np
import pandas as pd
import sys
import time
import glob
from pathlib import Path

from scale_split_data import partition
from getDataDirectory import getDir
from split_label      import splitY

FILENM        = 'test_CAWA.csv'

def getData():
    data_dir = getDir(client="SS", typ="ML")
    df       = pd.read_csv(data_dir+FILENM, sep="|")
    df       = df.sample(frac = 1.0)
    # No need for the HH during Testing
    del df['hh_num']
    # Drop any columns with NAN values
    before = df.shape[0]
    df     = df.dropna(axis=0, how='any')
    after  = df.shape[0]
    print("{}Deleted {:,.0f} rows out of {:,.0f} due to NAN".format(
            "\n",(before-after), before))
    return df

def prepareData():
    '''Read the test data file and create a dictionary with keys of "test_x" and "test_labels'''
    df = getData()
    data_dict = splitY(None, None, df)
    return data_dict

def print_sales_ratio(data_dict):
    pos = np.sum(data_dict['test_labels'][:,1])
    tot = data_dict['test_labels'].shape[0]
    print("{}Test file:   {:,.0f} rows with {:,.0f} Positives  {}:1".format(
            "\n", tot, pos, int(tot/pos)))

# These are the saved models, one model per set of training parameters
def get_models(job):
    job_name = "SS_job_" + str(job) + "/"
    saveDir = getDir(client="SS", typ="TB")
    for model in glob.glob(saveDir+job_name+'*.meta'):
        yield model

def score_model(model, data_dict, scores):
    tf.reset_default_graph()
    
    saver = tf.train.import_meta_graph(model)
    sess = tf.Session()
    mod = model.rstrip(".meta")
    saver.restore(sess, mod)
    
    predictions = sess.run("L2:0", feed_dict={"input:0": data_dict['test_x']})
    
    # Evaluate
    # count the number of True Positives in the top 20%
    K     = int(np.floor(predictions.shape[0] * .2))
    idx   = np.argpartition(predictions[:,1], -K)[-K:]
    topK  = data_dict['test_labels'][idx][:,1].sum()
    my_rr = topK / K
    lift = my_rr/true_rr-1
    
    print("{}Score for model {}: {:.2%}{}".format("\n", mod[mod.find("job"):],lift, "\n"))
    
    # Get the Run number from the model name
    start = model.find("run_")
    run_id = model[start+4:model.find(".meta")]
    
    # Save the results
    d = {"topK": topK, "lift": lift, "K": K}
    scores[run_id] = d
    sess.close()
    return scores

    # The use of the dictionary is only to match what's done in "Train"
if __name__ == "__main__":
    data_dict = prepareData()
    
    print_sales_ratio(data_dict)
    print("{}{}{}".format("\n", ".....pausing 10 seconds.....","\n"))
    time.sleep(10)
    
    # get the true response rate to compare to my predictions
    true_rr = data_dict['test_labels'][:,1].sum() / data_dict['test_labels'].shape[0]
    
    scores={}
    # Get the name of the saved model to use
    job = input("Enter the job ID of the saved model:")
    for model in get_models(job):
        scores = score_model(model, data_dict, scores)
        
    # Get the location for storing the results
    home = str(Path.home())
    
    with open(home+"/"+str(job)+"_test_scores.csv", 'w') as results:
        rec = "Run #" +"|"+ "topK" +"|"+ "lift" +"\n"
        results.write(rec)
        for k,v in scores.items():
            rec = str(k) +"|"+ str(v['topK']) +"|"+ str(v['lift']) +"\n"
            results.write(rec)
   