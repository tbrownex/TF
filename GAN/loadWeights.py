import keras

def loadWeights(model, config, typ):
    ''' Models are saved as either 'gen_Final' (generator) or  disc'_Final' (discriminator) as their suffix  '''
    filename = typ + "_Final.h5"
    return model.load_weights(config["modelDir"]+filename)