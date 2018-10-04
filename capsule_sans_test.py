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

from keras import layers, models, optimizers
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

from utils import plot_log
from utils import get_accuracy_results

from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
sys.path.append('./PatchyTools/')
from GraphConverter import GraphConverter
from DropboxLoader import DropboxLoader
from CapsuleParameters import CapsuleParameters
from CapsuleParameters import CapsuleTrainingParameters
from GraphClassifier import GraphClassifier

DIR_PATH = os.environ['GAMMA_DATA_ROOT']




if __name__ == "__main__":

    # Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', help='dataset to train on', default='cifar')
    parser.add_argument('-sampling_ratio', help='ratio to sample on', default=0.2)


    # Parsing arguments:
    args = parser.parse_args()
    dataset = args.dataset
    subsample_ratio = float(args.sampling_ratio)


    # setting the hyper parameters

    args = CapsuleTrainingParameters()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    # (x_train, y_train), (x_test, y_test) = load_cifar()  # load_mnist()
    # Getting the training data:


    dataset_name = 'MUTAG'
    width = 18
    receptive_field = 10
    PatchyConverter = GraphConverter(dataset_name, width, receptive_field)
    mutag_tensor = PatchyConverter.graphs_to_Patchy_tensor()
    # plt.imshow(mutag_tensor[0,:,:,2])

    # Getting the labels:
    dropbox_loader = DropboxLoader(dataset_name)
    mutag_labels = dropbox_loader.get_graph_label()
    mutag_labels = np.array(mutag_labels.graph_label)

    x_train, x_test, y_train, y_test = train_test_split(mutag_tensor,
                                                        mutag_labels,
                                                        test_size=0.10,
                                                        random_state=1)

    data = ((x_train, y_train), (x_test, y_test))

    input_shape = x_train.shape[1:]

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
    decoder_params['first_dense'] = 512  # 250 #512
    decoder_params['second_dense'] = 1024
    decoder_params['name'] = 'decoder'

    capsule_params.add_params(decoder_params, decoder_layer)

    # Generate list of parameter sets::
    list_parameter_sets = []
    n_epoch_list = [200]  # , 150, 200]
    lr_list = [0.001]  # , 0.005]#[0.0005, 0.001]#, 0.005]
    lr_decay_list = [0.85, 0.95]  # , 0.75]#[0.25,0.4]#, 0.75, 1.5]
    lam_recon_list = [0.25, 0.4, 0.55]

    for n_epoch in n_epoch_list:
        for lr in lr_list:
            for lr_decay in lr_decay_list:
                for lam_recon in lam_recon_list:
                    training_params = CapsuleTrainingParameters(epochs=n_epoch,
                                                                lr=lr,
                                                                lr_decay=lr_decay,
                                                                lam_recon=lam_recon)
                    list_parameter_sets.append(training_params)

    # num_params_list = len(list_parameter_sets)
    list_parameter_sets.append(args)
    list_parameter_sets[-1] = CapsuleTrainingParameters(epochs=n_epoch,
                                                        lr=lr,
                                                        lr_decay=lr_decay,
                                                        plot_log=True)

    print('Training in {} parameter sets'.format(len(list_parameter_sets)))
    for i, parameter_set in enumerate(list_parameter_sets):

        patchy_classifier = GraphClassifier(input_shape)
        patchy_classifier.build_the_graph(capsule_params)
        patchy_classifier.train(data, parameter_set)

        if i == 0:
            results_df = pd.DataFrame(patchy_classifier.results)
        else:
            results_df = pd.concat([results_df, patchy_classifier.results], 1)

        print('Set of parameters {} trained '.format(i + 1))

    # Adding index :
    results_df.columns = list(range(len(results_df.columns)))
    results_df = results_df.transpose()

    # Saving results:
    time_now = datetime.now()
    time_now = '_'.join([str(time_now.date()).replace('-', '_'), str(time_now.hour), str(time_now.minute)])

    results_path = DIR_PATH + 'Results/CapsuleSans/results_hp_search_{}.csv'.format(time_now)
    results_df.to_csv(results_path)

