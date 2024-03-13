import os, warnings

def setEnv():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings('ignore') 
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU