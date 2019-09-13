''' A dictionary object holding key parameters'''

__author__ = "The Hackett Group"

def getConfig():
    d = {}
    d["dataLoc"]    = "/home/tbrownex/data/CoursERA/Week4/GAN/"
    #d["labelsFile"] = "lfw_attributes.txt"
    d["batchSize"] = 64
    d["cropX"] = 80
    d["cropY"] = 80
    d["dimX"] = 36
    d["dimY"] = 36
    d["codeSize"] = 256
    #d["logLoc"]     = "/home/tbrownex/"
    #d["logFile"]    = "cifar10.log"
    #d["logDefault"] = "info"
    #d["TBdir"] = "/home/tbrownex/TF/TB/"
    d["modelDir"] = "/home/tbrownex/TF/models/"
    return d