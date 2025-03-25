import numpy as np
from skimage.metrics import structural_similarity as cal_ssim

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def MAPE(pred, true):
    return np.mean(np.abs(pred-true)/np.abs(true)) * 100



def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    mape = MAPE(pred, true)

    return mse, mae, mape