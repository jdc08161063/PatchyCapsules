#!/usr/bin/env python3
# coding: utf-8
"""
Implementation of Capsule Networks:
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from time import time
from datetime import datetime
from PIL import Image
import argparse
from collections import defaultdict

from keras import layers, models, optimizers
from keras import backend as K

K.set_image_data_format('channels_last')
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.losses import categorical_crossentropy

from utils import plot_log,save_results_to_csv
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

sys.path.append('./PatchyTools/')
from PatchyConverter import PatchyConverter

from DropboxLoader import DropboxLoader
from CapsuleParameters import CapsuleParameters
from CapsuleParameters import CapsuleTrainingParameters
from GraphClassifier import GraphClassifier

DIR_PATH = os.environ['GAMMA_DATA_ROOT']
RESULTS_PATH = os.path.join(DIR_PATH, 'Results/CapsuleSans/CNN_Caps_comparison_BC.csv')


if __name__ == "__main__":

    # Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='name_of the dataset', default='MUTAG')
    #parser.add_argument('-w', help='width for patchy', default=18)
    parser.add_argument('-k', help='receptive field for patchy', default=10)
    parser.add_argument('-e', help='number of epochs', default=400)
    #parser.add_argument('-c', help='number of classes', default=2)
    parser.add_argument('-f', help='number of different folds', default=10)
    parser.add_argument('-s', dest='save', help='saving by default', action='store_true')
    parser.add_argument('-r', dest='relabelling', help='reshuffling takes place', action='store_true')
    parser.add_argument('-nr', dest='relabelling', help='no reshuffling takes place', action='store_false')
    parser.set_defaults(relabelling=True)
    parser.set_defaults(save=True)

    # parser.add_argument('-sampling_ratio', help='ratio to sample on', default=0.2)

    # Parsing arguments:
    args = parser.parse_args()

    # Arguments:
    dataset_name = args.n
    #width = int(args.w)
    receptive_field = int(args.k)
    relabelling = args.relabelling
    epochs = int(args.e)
    n_folds = int(args.f)
    save = args.save
    #n_class = int(args.c)

    # print('relabelling:')
    # print('')
    # print(relabeling)

    # dataset_name = 'MUTAG'
    # width = 18
    # receptive_field = 10

    # Converting Graphs into Matrices:
    graph_converter = PatchyConverter(dataset_name, receptive_field)
    if relabelling:
        graph_converter.relabel_graphs()

    graph_tensor = graph_converter.graphs_to_patchy_tensor()
    avg_nodes_per_graph = graph_converter.avg_nodes_per_graph

    # Getting the labels:
    dropbox_loader = DropboxLoader(dataset_name)
    graph_labels = dropbox_loader.get_graph_label()
    graph_labels = np.array(graph_labels.graph_label)
    n_class = len(np.unique(graph_labels))

    # Capsule Architecture Parameters:
    capsule_params = CapsuleParameters()

    # First conv layer: 'filters', kernel_size)
    conv_layer_name = 'conv_layer'
    conv_layer_params = {}
    conv_layer_params['filters'] = 256
    conv_layer_params['kernel_size'] = 9
    conv_layer_params['strides'] = [1, 1]
    conv_layer_params['padding'] = 'VALID'
    conv_layer_params['activation'] = 'relu'
    conv_layer_params['name'] = 'conv1'

    capsule_params.add_params(conv_layer_params, conv_layer_name)

    # First Capsule Layer:
    # [num_output_caps, caps_len,'filters',kernel_size,strides,padding]
    caps_layer_name = 'caps_layer'
    caps_layer_params = {}
    caps_layer_params['filters'] = 256
    caps_layer_params['kernel_size'] = 2
    caps_layer_params['strides'] = [2, 2]
    caps_layer_params['padding'] = 'VALID'
    caps_layer_params['padding'] = 'VALID'
    caps_layer_params['n_channels'] = 32
    caps_layer_params['dim_capsule'] = 8
    caps_layer_params['name'] = 'caps_layer'
    capsule_params.add_params(caps_layer_params, caps_layer_name)

    # Digit Capsule Layer:
    digit_layer_name = 'digitcaps_layer'
    digit_layer_params = {}
    digit_layer_params['n_channels'] = 10
    digit_layer_params['dim_capsule'] = 16
    digit_layer_params['name'] = 'digitcaps'
    capsule_params.add_params(digit_layer_params, digit_layer_name)

    # Capsule Decoder:
    decoder_layer = 'decoder_layer'
    decoder_params = {}
    decoder_params['first_dense'] = 256  # 250 #512
    decoder_params['second_dense'] = 512
    decoder_params['name'] = 'decoder'
    capsule_params.add_params(decoder_params, decoder_layer)

    # Training Hyperparameters:

    args_train = CapsuleTrainingParameters()
    args_train.batch_size = 100
    if not os.path.exists(args_train.save_dir):
        os.makedirs(args_train.save_dir)

    # Generate list of parameter sets::
    list_parameter_sets = []
    #list_parameter_sets.append(args_train)

    n_epoch_list = [epochs]  # , 150, 200]
    lr_list = [0.001]  # , 0.005]#[0.0005, 0.001]#, 0.005]
    lr_decay_list = [0.9]  # , 0.75]#[0.25,0.4]#, 0.75, 1.5]
    lam_recon_list = [0.392]  # ,0.7] # [0.25,0.4,0.55]

    for n_epoch in n_epoch_list:
        for lr in lr_list:
            for lr_decay in lr_decay_list:
                for lam_recon in lam_recon_list:
                    training_params = CapsuleTrainingParameters(epochs=n_epoch,
                                                                lr=lr,
                                                                lr_decay=lr_decay,
                                                                lam_recon=lam_recon)
                    list_parameter_sets.append(training_params)

    #list_parameter_sets[-1].plot_log = True

    # Default parameters:
    print('Training in {} parameter sets'.format(len(list_parameter_sets)))


    fold_set = []

    for j in range(n_folds):
        fold_set.append(train_test_split(graph_tensor,
                                         graph_labels,
                                         test_size=0.10,
                                         random_state=j))

    results_df = []
    val_acc = []
    training_time=[]
    for i, parameter_set in enumerate(list_parameter_sets):
        for j in range(n_folds):

            x_train, x_test, y_train, y_test = fold_set[j]
            data = ((x_train, y_train), (x_test, y_test))
            input_shape = x_train.shape[1:]
            parameter_set.add_fold(i)

            patchy_classifier = GraphClassifier(input_shape,n_class)
            patchy_classifier.build_the_graph(capsule_params)
            ##
            #patchy_classifier.train_model.summary()

            ##
            patchy_classifier.train(data, parameter_set)

            training_time.append(patchy_classifier.training_time)
            val_acc.append(patchy_classifier.results.val_capsnet_acc)

            #if i == 0:
            results_df.append(pd.DataFrame(patchy_classifier.results))
            #else:
            print('Fold number {} trained '.format(j + 1))



    #OLD results:
    # results_df = pd.concat(results_df, 1)
    # results_df.columns = list(range(len(results_df.columns)))
    # results_df = results_df.transpose()
    #
    # # Saving results:
    # time_now = datetime.now()
    # time_now = '_'.join([str(time_now.date()).replace('-', '_'), str(time_now.hour), str(time_now.minute)])
    # results_path = DIR_PATH + 'Results/CapsuleSans/{}_results_{}_{}.csv'.format(dataset_name, time_now, exp_name)


    # results_df.to_csv(results_path)
    #

    mean_acc = np.mean(val_acc)
    std_acc = np.std(val_acc)

    mean_time = np.mean(training_time)
    std_time = np.std(training_time)

    results_dd = defaultdict(list)
    results_dd['Mean accuracy'].append(mean_acc)
    results_dd['Std accuracy'].append(std_acc)
    results_dd['Relabelling'].append(relabelling)
    results_dd['Dataset'].append(dataset_name)
    results_dd['Number of graphs'].append(graph_tensor.shape[0])
    results_dd['Number of epochs'].append(epochs)
    results_dd['Number of folds'].append(n_folds)
    results_dd['Avg number of nodes'].append(avg_nodes_per_graph)
    results_dd['Mean training time'].append(mean_time)
    results_dd['Std training time'].append(std_time)
    results_dd['Model'].append('Patchy + Capsule Net')


    results_df = pd.DataFrame(results_dd)



    if save == True:
        #print('getting here')
        save_results_to_csv(results_df, RESULTS_PATH)






