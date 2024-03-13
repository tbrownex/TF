from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.synthesizers.timeseries import TimeGAN

def createGAN (cfg):
    gamma=1
    
    gan_args = ModelParameters(batch_size=cfg['batchSize'],
                           lr=cfg['learningRate'],
                           noise_dim=cfg['batchSize'])
    gan = TimeGAN(
        model_parameters=gan_args,
        hidden_dim=cfg['numUnits'],
        seq_len=cfg['seqLength'],
        n_seq=cfg['numFeatures'],
        gamma=cfg['gamma'])
    return gan