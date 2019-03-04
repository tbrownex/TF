import itertools
from tensorflow import keras

def getParms(typ):
    if typ == "RF":
        nEstimators      = [50]
        min_samples_split = [25]
        max_depth         = [10]
        min_samples_leaf  = [4]
        max_features      = [ 0.9]
        return list(itertools.product(nEstimators,
                                      min_samples_split,
                                      max_depth,
                                      min_samples_leaf,
                                      max_features))
    elif typ == "XGB":
        nEstimators       = [100]
        learningRate      = [2e-2]
        maxDepth          = [5]
        min_child_weight  = [8]
        colsample_bytree  = [0.55]
        subsampling       = [1.0]
        gamma             = [0]
        return list(itertools.product(nEstimators,
                                      learningRate,
                                      maxDepth,
                                      min_child_weight,
                                      colsample_bytree,
                                      subsampling,
                                      gamma))
    elif typ == "NN":
        L1Size       = [24, 48]
        activation   = ["tanh", "relu"]
        batchSize    = [32, 64]
        learningRate = [1e-3, 1e-4]
        std          = [0.1, 0.25]
        dropout      = [0.25, 0.4]
        optimizer    = ["Adam"]
        return list(itertools.product(L1Size,
                                      activation,
                                      batchSize,
                                      learningRate,
                                      std,
                                      dropout,
                                      optimizer))
    elif typ == "AE":     # AutoEncoder for outlier detection
        L1Size       = [18]
        L2Size       = [12]
        activation   = ["tanh"]
        batchSize    = [32]
        learningRate = [2e-3]
        std          = [0.5]
        dropout      = [0.4]
        optimizer    = ["Adam"]
        return list(itertools.product(L1Size,
                                      L2Size,
                                      activation,
                                      batchSize,
                                      learningRate,
                                      std,
                                      dropout,
                                      optimizer))