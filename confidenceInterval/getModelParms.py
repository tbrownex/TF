import itertools
from tensorflow import keras

def getParms(typ):
    if typ == "NN":
        L1Size       = [60]
        batchSize    = [64]
        learningRate = [1e-1, 5e-2]
        Lambda       = [1e-3]
        std          = [0.1]
        dropout      = [0.33]
        dropoutTest  = [0.05]
        optimizer    = ["Adam"]
        return list(itertools.product(L1Size,
                                      batchSize,
                                      learningRate,
                                      Lambda,
                                      std,
                                      dropout,
                                      dropoutTest,
                                      optimizer))