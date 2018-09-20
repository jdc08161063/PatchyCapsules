import pickle
import numpy as np
from numpy import random as nprand

def unpickle(file):
    with open(file, 'rb') as fo:
        dictio = pickle.load(fo, encoding='bytes')
    return dictio

def load_image_data(data_path):
    if 'cifar' in data_path:
        dictio_data = unpickle(data_path)
        X = dictio_data[b'data'].reshape([10000,3,32,32]).transpose(0,2,3,1)
        y = np.array(dictio_data[b'labels'])
        print('shape : ', X.shape)
        return X,y


def subsample(X_train, y_train, ratio=0.1):
    idx_size = X_train.shape[0]
    idx = nprand.choice(range(idx_size),round(idx_size*ratio))
    sample_y = y_train[idx]
    sample_x = X_train[idx]
    return sample_x,sample_y