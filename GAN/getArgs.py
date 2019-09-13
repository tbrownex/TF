''' Get the command line arguments:
    Mandatory
    - remOutliers: whether to remove outliers from Training
    - genFeature: whether to use a generated feature or not
    
    Optional arguments
    - log: the logging level (overrides "config")
    - normalize: the algorithm to use when normalizing the input'''
    
__author__ = "The Hackett Group"

import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    '''parser.add_argument("genFeatures", \
                        choices=['Y','N'], \
                        help="Generate additional features or not")'''
    parser.add_argument('epochStart',  type=int, 
                    help='starting Epoch number')
    return parser.parse_args()