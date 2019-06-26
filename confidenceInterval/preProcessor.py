import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preProcess(bostonDataset):
    X = bostonDataset.data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    Y = bostonDataset.target
    Y = np.reshape(Y, newshape=[-1,1])
    
    d = {}
    d["trainX"], d["testX"], d["trainY"], d["testY"] = train_test_split(X, Y, test_size=0.2, random_state=42)
    return d