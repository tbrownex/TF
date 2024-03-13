from sklearn.preprocessing import MinMaxScaler
import numpy as np

import constants

def normalize(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    # Save the min/max to rescale the predictions later
    scalerParms = np.array([scaler.data_min_, scaler.data_range_])
    np.save(constants.SCALERFILENAME, scalerParms, allow_pickle=False)
    return data

def getSequences(data, seqLength):
    sequences = []
    for i in range(0, len(data) - seqLength):
        seq = data[i:i + seqLength]
        sequences.append(seq)
    return sequences

def prep(data, cfg, norm=True):
    if norm:
        data = normalize(data)
    data = getSequences(data, cfg['seqLength'])
    data = np.array(data)
    numBatches = int(len(data)/cfg['batchSize'])
    return data[:numBatches*cfg['batchSize']]