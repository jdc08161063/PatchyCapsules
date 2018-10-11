import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from time import time
import argparse
from collections import defaultdict

import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D,Dropout
from keras.models import Sequential
from keras import layers, models, optimizers
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

from utils import combine_images
from utils import plot_log
from utils import save_results_to_csv

from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
sys.path.append('./PatchyTools/')
from PatchyConverter import PatchyConverter
from DropboxLoader import DropboxLoader
from CapsuleParameters import CapsuleParameters
from CapsuleParameters import CapsuleTrainingParameters


DIR_PATH = os.environ['GAMMA_DATA_ROOT']
RESULTS_PATH = os.path.join(DIR_PATH, 'Results/CapsuleSans/CNN_Caps_comparison.csv')


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))


class ConvNetPatchy(object):
    def __init__(self, graph_tensor, graph_labels):
        self.graph_tensor = graph_tensor
        self.graph_labels = graph_labels
        self.name = 'cnn'
        self.input_shape = graph_tensor.shape[1:]
        self.n_class = len(np.unique(graph_labels))

    def split_data(self, random_state=42, test_size=0.10):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.graph_tensor, self.graph_labels,
                                                                                test_size=test_size,
                                                                                random_state=random_state)
        self.y_train = pd.get_dummies(self.y_train).values
        self.y_test = pd.get_dummies(self.y_test).values

    def build_graph(self, ):
        K.clear_session()
        self.model = Sequential()
        self.model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=self.input_shape,kernel_initializer='glorot_uniform'))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(8, kernel_size=(5, 5), activation='relu',kernel_initializer='glorot_uniform'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu',kernel_initializer='glorot_uniform'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.n_class, activation='softmax'))

    def train_model(self, epochs=200, batch_size=100, verbose=0):
        start = time()
        self.epochs = epochs
        self.history = AccuracyHistory()
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(decay=1e-6),
                           metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=(self.x_test, self.y_test),
                       callbacks=[self.history])
        self.final_acc = self.history.acc[-1]
        self.final_val_acc = self.history.val_acc[-1]
        self.training_time = time() - start



if __name__ == "__main__":

    # Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='name_of the dataset', default='MUTAG')
    #parser.add_argument('-w', help='width for patchy', default=18)
    parser.add_argument('-k', help='receptive field for patchy', default=10)
    parser.add_argument('-e', help='number of epochs', default=400)
    parser.add_argument('-f', help='number of different folds', default=10)
    parser.add_argument('-s', dest='save', help='saving by default', action='store_true')
    parser.add_argument('-r', dest='relabelling', help='reshuffling takes place', action='store_true')
    parser.add_argument('-nr', dest='relabelling', help='no reshuffling takes place', action='store_false')
    parser.set_defaults(relabelling=True)
    parser.set_defaults(save=False)


    # parser.add_argument('-sampling_ratio', help='ratio to sample on', default=0.2)

    # Parsing arguments:
    args = parser.parse_args()

    # Arguments:
    dataset_name = args.n
    #width = int(args.w)
    receptive_field = int(args.k)
    epochs = int(args.e)
    n_folds = int(args.f)
    relabeling = args.relabelling
    save = args.save


    # Getting the data:
    # dataset_name = 'PTC_FM'
    # width = 25
    # receptive_field = 10
    graph_converter = PatchyConverter(dataset_name, receptive_field)
    if relabeling:
        graph_converter.relabel_graphs()
    graph_tensor = graph_converter.graphs_to_Patchy_tensor()

    avg_nodes_per_graph = graph_converter.avg_nodes_per_graph

    # Getting the labels:
    dropbox_loader = DropboxLoader(dataset_name)
    graph_labels = dropbox_loader.get_graph_label()
    graph_labels = np.array(graph_labels.graph_label)


    val_acc = []
    training_time = []
    for j in range(n_folds):
        # Starting Conv net with Patchy
        patchy_cnn = ConvNetPatchy(graph_tensor, graph_labels)
        patchy_cnn.split_data(random_state=j)
        patchy_cnn.build_graph()
        patchy_cnn.train_model(epochs=epochs)
        print('Fold {} completed with val acc : {}'.format(j, patchy_cnn.final_val_acc))

        val_acc.append(patchy_cnn.final_val_acc)
        training_time.append(patchy_cnn.training_time)


    mean_acc = np.mean(val_acc)
    std_acc = np.std(val_acc)

    mean_time = np.mean(training_time)
    std_time = np.std(training_time)

    results_dd = defaultdict(list)
    results_dd['Mean accuracy'].append(mean_acc)
    results_dd['Std accuracy'].append(std_acc)
    results_dd['Relabelling'].append(relabeling)
    results_dd['Dataset'].append(dataset_name)
    results_dd['Number of graphs'].append(graph_tensor.shape[0])
    results_dd['Number of epochs'].append(epochs)
    results_dd['Number of folds'].append(n_folds)
    results_dd['Avg number of nodes'].append(avg_nodes_per_graph)
    results_dd['Mean training time'].append(mean_time)
    results_dd['Std training time'].append(std_time)
    results_dd['Model'].append('Patchy + CNN')

    results_df = pd.DataFrame(results_dd)
    if save == True:
        save_results_to_csv(results_df, RESULTS_PATH)


























