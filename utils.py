import pickle
import numpy as np

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