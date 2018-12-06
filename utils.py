import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import csv
import math
import pandas as pd
import os

def plot_log(filename, show=True):

    data = pd.read_csv(filename)

    fig = plt.figure(figsize=(6,8))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(311)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(313)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    fig.add_subplot(312)
    for key in data.keys():
        if key.find('loss') >= 0 and key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Validation loss')


    fig.savefig('result/log.png')
    # if show:
    #     plt.show()


def get_accuracy_results(filename): #, index): # show=True):
    df = pd.read_csv(filename) #,index_col=0)
    df = df.loc[:, ['epoch', 'capsnet_acc', 'val_capsnet_acc']]
    results = df.iloc[-1, :] #.val_capsnet_acc
    results.epoch = results.epoch + 1
    results.name = None

    return results



def save_results_to_csv(file_to_append, file_name):
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, index_col=0)
        df = df.append(file_to_append)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(file_name)
    else:
        file_to_append.to_csv(file_name)


def indices_to_one_hot(number, nb_classes, label_dummy=-1):
    """Convert an iterable of indices to one-hot encoded labels."""

    if number == label_dummy:
        return np.zeros(nb_classes)
    else:
        return np.eye(nb_classes)[number]



def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


if __name__=="__main__":
    plot_log('result/log.csv')



