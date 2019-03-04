import numpy as np
from calcMAPE import calcMAPE
from calcRMSE import calcRMSE
'''
    Calculate the MAPE and RMSE of predictions vs actuals from the Test data.
    "predictions" is a dataframe with "unit" as the index. Columns are:
        - first column is the Baseline prediction using only the mean failure rate
        - next few columns are the predictions from the different algos
        - last column holds the actuals
        
    "Unit" is indexed so we process the data a Unit at a time. That's how this would run in Production: you
    would be looking at a single machine's output and making a forecast for that machine, in isolation from
    the others.
    
    The method of evaluation is to look at the most recent prediction, which is the last entry for each unit
    '''
    
def getError(df, col):
    errors = {}
    actuals = []
    preds   = []           
    grp = df.groupby(level=0)
    for unit, data in grp:
        data = data.tail(1)                # get the last row
        preds.append(data[col].iloc[0])
        actuals.append(data["actual"].iloc[0])
    actuals = np.array(actuals)
    preds   = np.array(preds)
    errors["mape"] = calcMAPE(actuals, preds)
    errors["rmse"] = calcRMSE(actuals, preds)
    return errors

def getEnsemble(df):
    '''
    This is the "official" prediction: the average of the other algos
    '''
    errors = {}
    actuals = []
    preds   = []
    
    grp = df.groupby(level=0)
    for unit, data in grp:
        data = data.tail(1)                # get the last 10 rows for the Regression data
        actuals.append(data["actual"].iloc[-1])
        del data["actual"]                    # Don't include the actual in the ensemble of predictions
        del data["Baseline"]                  # Don't want the Baseline included either
        ensemble = data.mean(axis=1)          # axis=1 gets the mean across rows of different algos
        preds.append(ensemble.iloc[-1])
    actuals = np.array(actuals)
    preds   = np.array(preds)
    errors["mape"] = calcMAPE(actuals, preds)
    errors["rmse"] = calcRMSE(actuals, preds)
    return errors

def evaluate(predDF, ensemble):
    '''
    You're either optimizing an algo...not an ensemble 
    OR you're running all the algos...an ensemble
    '''
    errors = {}
    if ensemble:
        for col in ["Baseline", "RF", "NN", "XGB"]:
            errors[col] = getError(predDF, col)
        errors["ensemble"] = getEnsemble(predDF)
    else:
        cols = set(predDF.columns) - set(["actual", "unit"])  # This will get the type of algo run e.g. "RF"
        for col in cols:
            errors[col] = getError(predDF, col)
    return errors