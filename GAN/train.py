import time
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras import layers as L

from getArgs       import getArgs
from getConfig     import getConfig
from getData       import getData
from buildNN     import buildNN
from saveModel import saveModel
from loadWeights import loadWeights

tf.logging.set_verbosity(tf.logging.ERROR)

EPOCHS = 80

def getNoiseBatch(config):
    # This is the starting point to generate an image
    return np.random.normal(size=(config["batchSize"], config["codeSize"])).astype('float32')
    
def process(images, config, args):
    gen = buildNN(config, "generator")
    disc = buildNN(config, "discriminator")
    
    ''' If "epochStart" is non-zero that means previous runs have saved models. So load those weights '''
    if args.epochStart > 0:
        '''
        There is a "model.load" method that I could not get working; seems to be a bug within keras. The workaround
        is to build the network (buildGAN) then load_weights
        '''
        loadWeights(gen, config, "gen")
        loadWeights(disc, config, "disc")
    
    imgShape=(config["dimX"], config["dimY"], 3)
    noise = tf.placeholder('float32',[None,config["codeSize"]])     # Input to the generator
    realData = tf.placeholder('float32',[None,]+list(imgShape))   # Input to the discriminator (actual images)
    
    logp_real = disc(realData)
    logp_gen = disc(gen(noise))
    
    # Loss functions: column 0 is Real probability; column 1 is fake
    discLoss = -tf.reduce_mean(logp_real[:,1] + logp_gen[:,0])
    genLoss = -tf.reduce_mean(logp_gen[:,1])
    #regularize
    discLoss += tf.reduce_mean(disc.layers[-1].kernel**2)
    #optimizers
    discOptimizer =  tf.train.GradientDescentOptimizer(1e-3).minimize(discLoss,var_list=disc.trainable_weights)
    genOptimizer = tf.train.AdamOptimizer(1e-4).minimize(genLoss,var_list=gen.trainable_weights)    
    
    batch = config["batchSize"]    # only to use a shorter variable name
    numTrainingBatches = int(images.shape[0] / batch)
    print('{} epochs of {} iterations with batch size {}'.format(EPOCHS, numTrainingBatches, batch))
    
    start = time.time()
    print("{:<8}{:<12}".format("Epoch", "Loss"))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.epochStart, args.epochStart+EPOCHS):
            print("running epoch ", epoch)
            np.random.shuffle(images)
            for idx in range(numTrainingBatches):
                imageBatch = images[idx*batch : idx*batch+batch]
                feedDict = { realData:imageBatch, noise: getNoiseBatch(config) }
                ''' Each batch should be trained by either the Discriminator or the Generator. You want to go back-and-forth so neither 
                gets a big lead. Do that by choosing which one randomly.
                Usually either the Discriminator or Generator trains faster (Generator in this case) so I have biased the selection '''
                if np.random.randint(100) > 60:
                    sess.run(genOptimizer, feedDict)
                else:
                    sess.run(discOptimizer, feedDict)
                
            if epoch%10 == 0 and epoch > 0:
                loss = sess.run(discLoss, feedDict)
                print("{:<8}{:<8.2}".format("Loss:", loss))
                saveModel(config, gen, "gen", epoch)
                saveModel(config, disc, "disc", epoch)
        
        saveModel(config, gen, "gen", "Final")
        saveModel(config, disc, "disc", "Final")
    
    elapsed = (time.time()-start)/60
    print("Done after {:.0f} minutes ({:.0f} minutes per 100 epochs)".format(elapsed, elapsed/EPOCHS*100 ))
    
if __name__ == "__main__":
    args     = getArgs()
    config  = getConfig()
    images = getData(config)
    process(images, config, args)