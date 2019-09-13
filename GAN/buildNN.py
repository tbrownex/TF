import tensorflow as tf
import keras
import keras.layers as L

def addConv(nn, size):
    nn.add(keras.layers.Conv2D(
        filters=size,
        kernel_size=3,
        strides=1,
        padding="same",
        data_format="channels_last",
        use_bias=True))
def addELU(nn):
    nn.add(keras.layers.ELU(alpha=0.1))    
def addPool(nn):
    nn.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
def addDense(nn, size):
    nn.add(L.Dense(units=size, activation="elu"))
def transpose(nn, size, kernelSize):
    nn.add(L.Conv2DTranspose(filters=size, kernel_size=kernelSize, strides=(1,1), activation="elu", padding='valid'))

def buildGenerator(config, imgShape):
    nn = keras.models.Sequential()
    nn.add(L.InputLayer([config["codeSize"]], name='noise'))
    addDense(nn, 10*8*8)
    nn.add(L.Reshape((8,8,10)))
    transpose(nn, 64, kernelSize=(5,5))
    transpose(nn, 64, kernelSize=(5,5))
    nn.add(L.UpSampling2D(size=(2,2)))
    transpose(nn, 32, kernelSize=3)
    transpose(nn, 32, kernelSize=3)
    transpose(nn, 32, kernelSize=3)
    nn.add(L.Conv2D(3,kernel_size=3,activation=None))
    return nn

def buildDiscriminator(config, imgShape):
    nn = keras.models.Sequential()
    nn.add(L.InputLayer(imgShape))
    
    addConv(nn, 32)
    addELU(nn)
    addPool(nn)
    addConv(nn, 64)
    addELU(nn)
    addPool(nn)
    addConv(nn, 128)
    addELU(nn)
    addPool(nn)
    addConv(nn, 256)
    addELU(nn)
    addPool(nn)
    
    nn.add(L.Flatten())
    nn.add(L.Dense(256,activation='tanh'))
    nn.add(L.Dense(2,activation=tf.nn.log_softmax))
    return nn
    
def buildNN(config, typ):
    assert typ in ["generator", "discriminator"], "Invalid code type"
    imgShape=(config["dimX"], config["dimY"], 3)
    
    if typ == "generator":
        gen = buildGenerator(config, imgShape)
        assert gen.output_shape[1:] == imgShape, "generator has invalid shape"
        return gen
    else:
        disc = buildDiscriminator(config, imgShape)
        return disc