import numpy as np

def createSinWave(numSamples, cfg):
    tmpList = list()
    for _ in range(cfg['numFeatures']):
        # Randomly drawn frequency and phase
        freq = np.random.uniform(0, 0.1)
        phase = np.random.uniform(0, 0.1)
        tmp = [np.sin(freq * j + phase) for j in range(numSamples)]
        tmpList.append(tmp)
    # Normalize to [0,1]
    data = np.asarray(tmpList) + 1
    return np.transpose(data * 0.5)