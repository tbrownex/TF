import numpy as np

def splitY(train, val, test):
    '''Inputs are dataframes with last column being the Label.
    Separate the Label from the data and put into distinct data dictionary keys.
    Assume that if you have Train then you will have Val'''
    LABELS   = np.array([0,1])
    data_dict = {}
    
    if train is not None:
        train = train.as_matrix()
        val   =   val.as_matrix()
    
        # First the features
        data_dict['train_x'] = train[:,:-1]
        data_dict['val_x']   =   val[:,:-1]
    
    if test is not None:
        test = test.as_matrix()
        data_dict['test_x'] = test[:,:-1]
    
    # Now the Labels: convert them to "one-hot" vectors
    if train is not None:
        train_y = train[:,-1]
        val_y   =   val[:,-1]
    
        data_dict['train_labels'] = (LABELS == train_y[:, None]).astype(np.float32)
        data_dict['val_labels']   = (LABELS == val_y[:,   None]).astype(np.float32)
    
    if test is not None:
        test_y = test[:,-1]
        data_dict['test_labels']   = (LABELS == test_y[:,   None]).astype(np.float32)
    
    return data_dict