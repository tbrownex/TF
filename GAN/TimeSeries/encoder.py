import tensorflow as tf

def encoder(cfg):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(
        units=cfg['numUnits'],
        return_sequences=True,
        input_shape=(cfg['seqLength'], cfg['numFeatures'])))
    model.add(tf.keras.layers.LSTM(
        units=cfg['numUnits'],
        return_sequences=True))
    model.add(tf.keras.layers.LSTM(
        units=cfg['numUnits'],
        return_sequences=False))
    model.add(tf.keras.layers.Dense(
        units=cfg['numUnits'],
        activation='sigmoid'))
    return model