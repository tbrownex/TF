''' Prepare the data for modeling:
    - Identify any static columns (single value) and remove them
    - Convert ordinal values to one-hot
    - Split data into Train & Test
    - Split the features from the label
    - (optional) Normalize the data
    - (optional) Remove outliers
    - Shuffle the Training data
    '''
__author__ = "Tom Browne"

from normalizeData import normalize
from analyzeCols   import analyzeCols
from splitData     import splitData
from splitLabel    import splitLabel
from convertOrdinals import convertOrdinals
#from removeOutliers import removeOutliers
#from genFeatures import genFeatures

def removeCols(df):
    cols   = df.columns
    remove = analyzeCols(df)
    keep = [col for col in cols if col not in remove]
    df = df[keep]
    return df

def preProcess(df, config, args):
    df = removeCols(df)
    '''if args.genFeatures == "Y":
        print("\nGenerating features")
        df = genFeatures(train, test)
    '''
    df = convertOrdinals(df)
    dataDict = splitData(df, config)
    dataDict = splitLabel(dataDict, config)
    
    if config["normalize"]:
        print(" - Normalizing the data")
        dataDict = normalize(dataDict, "Std")
    else:
        print(" - Not normalizing the data")
        
    if args.Outliers == "Y":
        print(" - Removing outliers")
        dataDict = removeOutliers(dataDict)    
    else:
        print(" - Not removing outliers")
    # Shuffle the training data
    dataDict["trainX"] = dataDict["trainX"].sample(frac=1).reset_index(drop=True)
    return dataDict