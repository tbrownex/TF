import pandas as pd

def convertOrdinals(df):
    ''' incoming data has all ordinal and binary columns so need to one-hot them '''
    # All the ordinals are "int" dtype
    ordinals = df.select_dtypes(include=int)
    # "get_dummies" only operates on strings/objects
    ordinals = ordinals.astype(str)
    # One-hot them
    ordinals = pd.get_dummies(ordinals)
    # Add the label
    ordinals["MeanRunTime"] = df["MeanRunTime"]
    return ordinals