import tensorflow as tf

def decoder(cfg, inputShape):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(
        units=cfg['numUnits'],
        return_sequences=True,
        input_shape=(inputShape)))
    model.add(tf.keras.layers.LSTM(
        units=cfg['numUnits'],
        return_sequences=True))
    model.add(tf.keras.layers.LSTM(
        units=cfg['numUnits'],
        return_sequences=False))
    model.add(tf.keras.layers.Dense(
        units=cfg['numFeatures'],
        activation='sigmoid'))
    return model