# -*- coding: utf-8 -*-

from sklearn.datasets import make_circles
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def model_acc_loss():
    Training_loss=np.load('Data/Training_loss.npy')
    Training_accuracy=np.load('Data/Training_accuracy.npy')
    validation_loss=np.load('Data/validation_loss.npy')
    validation_accuracy=np.load('Data/validation_accuracy.npy')
    return Training_loss,validation_loss,Training_accuracy,validation_accuracy
    
    
