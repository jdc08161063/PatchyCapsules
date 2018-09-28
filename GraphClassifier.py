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

DIR_PATH = os.environ['GAMMA_DATA_ROOT'] + 'Samples/'

class GraphClassifier(object):
    def __init__(self,input_shape, n_class=2, routings=3):
        # Fixed initialization parameters:
        self.input_shape = input_shape
        self.n_class = n_class
        self.routings = routings



    def import_data(self,data):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data

        #assert(self.input_shape == x_train.shape[1:], 'input shape doesnt match ')

        self.y_train = pd.get_dummies(self.y_train).values
        self.y_test = pd.get_dummies(self.y_test).values

    def import_nn_parameters(self,params):
        self.conv_layer = params.get_layer_params('conv_layer')
        self.primary_caps_layer = params.get_layer_params('caps_layer')
        self.digit_caps_layer = params.get_layer_params('digitcaps_layer')
        self.decoder_layer = params.get_layer_params('decoder_layer')




    def build_the_graph(self,params):
        """
        A Capsule Network on MNIST.
        :param input_shape: data shape, 3d, [width, height, channels]
        :param n_class: number of classes
        :param routings: number of routing iterations
        :return: Two Keras Models, the first one used for training, and the second one for evaluation.
                `eval_model` can also be used for training.
        """
        self.import_nn_parameters(params)

        start = time()
        x = layers.Input(shape=self.input_shape)

        # Layer 1: Just a conventional Conv2D layer
        #params_conv_layer = self.params[0]

        conv1 = layers.Conv2D(filters=self.conv_layer['filters'],
                              kernel_size=self.conv_layer['kernel_size'],
                              strides=self.conv_layer['strides'],
                              padding=self.conv_layer['padding'],
                              activation=self.conv_layer['activation'],
                              name=self.conv_layer['activation'])(x)
                              # filters=128,
                              # kernel_size=9,
                              # strides=1,
                              # padding='valid',
                              # activation='relu',
                              # name='conv1')(x)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primarycaps = PrimaryCap(conv1,
                                 dim_capsule=self.primary_caps_layer['dim_capsule'],
                                 n_channels=self.primary_caps_layer['n_channels'],
                                 kernel_size=self.primary_caps_layer['kernel_size'],
                                 strides=self.primary_caps_layer['strides'],
                                 padding=self.primary_caps_layer['padding'])
                                 # dim_capsule=8,
                                 # n_channels=32,
                                 # kernel_size=2,
                                 # strides=2,
                                 # padding='valid')

        # Layer 3: Capsule layer. Routing algorithm works here.
        digitcaps = CapsuleLayer(num_capsule=self.n_class,
                                 dim_capsule=self.digit_caps_layer['dim_capsule'],
                                 #/dim_capsule = 16
                                 routings=self.routings,
                                 name=self.digit_caps_layer['name'])(primarycaps)

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        # If using tensorflow, this will not be necessary. :)
        out_caps = Length(name='capsnet')(digitcaps)

        # Decoder network.
        y = layers.Input(shape=(self.n_class,))
        masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
        masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

        # Shared Decoder model in training and prediction
        decoder = models.Sequential(name='decoder')
        decoder.add(layers.Dense(self.decoder_layer['first_dense'], activation='relu', input_dim=self.digit_caps_layer['dim_capsule'] * self.n_class))
        decoder.add(layers.Dense(self.decoder_layer['second_dense'], activation='relu'))
        # decoder.add(layers.Dense(128, activation='relu', input_dim=16 * self.n_class))
        # decoder.add(layers.Dense(256, activation='relu'))
        decoder.add(layers.Dense(np.prod(self.input_shape), activation='sigmoid'))
        decoder.add(layers.Reshape(target_shape=self.input_shape, name='out_recon'))

        # Models for training and evaluation (prediction)
        train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
        eval_model = models.Model(x, [out_caps, decoder(masked)])

        # manipulate model
        noise = layers.Input(shape=(self.n_class, self.digit_caps_layer['dim_capsule'])) # 16
        noised_digitcaps = layers.Add()([digitcaps, noise])
        masked_noised_y = Mask()([noised_digitcaps, y])
        manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
        self.train_model = train_model
        self.eval_model = eval_model
        self.manipulate_model = manipulate_model
        print('time to generate the model: {}'.format(time()-start))
        return train_model, eval_model, manipulate_model



    def margin_loss(self,y_true, y_pred):
        """
        Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
        :param y_true: [None, n_classes]
        :param y_pred: [None, num_capsule]
        :return: a scalar loss value.
        """
        L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
            0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

        return K.mean(K.sum(L, 1))

    def train_generator(self,x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])


    def train(self, data, args):
        """
        Training a CapsuleNet
        :param model: the CapsuleNet model
        :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
        :param args: arguments
        :return: The trained model
        """
        self.import_data(data)

        # if not hasattr(self, 'train_model'):
        #     self.build_the_graph()
        # time:
        start = time()
        # callbacks
        self.log_file = args.save_dir + '/log.csv'

        log = callbacks.CSVLogger(self.log_file)
        tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                                   batch_size=args.batch_size, histogram_freq=int(args.debug))
        checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                               save_best_only=True, save_weights_only=True, verbose=1)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

        # compile the model
        self.train_model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=[self.margin_loss, 'mse'],
                      loss_weights=[1., args.lam_recon],
                      metrics={'capsnet': 'accuracy'})


        # Training without data augmentation:
        #print('shape validation : ', np.array([[self.x_test, self.y_test], [self.y_test, self.x_test]]).shape)
        if args.data_augmentation == False:
            self.train_model.fit([self.x_train, self.y_train], [self.y_train, self.x_train], batch_size=args.batch_size, epochs=args.epochs,
                   validation_data=[[self.x_test, self.y_test], [self.y_test, self.x_test]], callbacks=[log, tb, checkpoint, lr_decay])
            #print('Evaluation: ',self.train_model.predict([[self.x_test, self.y_test], [self.y_test, self.x_test]]))
        else:
            # Begin: Training with data augmentation ---------------------------------------------------------------------#
            # Training with data augmentation. If shift_fraction=0., also no augmentation.
            self.train_model.fit_generator(generator=self.train_generator(self.x_train, self.y_train, args.batch_size, args.shift_fraction),
                                steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                                epochs=args.epochs,
                                validation_data=[[self.x_test, self.y_test], [self.y_test, self.x_test]],
                                callbacks=[log, tb, checkpoint, lr_decay])
            # End: Training with data augmentation -----------------------------------------------------------------------#
            self.train_model.save_weights(args.save_dir + '/trained_model.h5')
        print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)


        # Save the results:
        if args.plot_log == True:
            plot_log(self.file_to_save, show=True)
        self.training_time = time() - start
        self.results = get_accuracy_results(self.log_file)
        self.results['time'] = self.training_time


        #return self.train_model


if __name__ == "__main__":


    # setting the hyper parameters

    args = CapsuleTrainingParameters()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    #(x_train, y_train), (x_test, y_test) = load_cifar()  # load_mnist()
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
                                                        test_size=0.10)#,
                                                        #random_state=42)

    data = ((x_train,y_train),(x_test, y_test))

    input_shape = x_train.shape[1:]

    capsule_params= CapsuleParameters()

    # First conv layer: 'filters', kernel_size)
    conv_layer_name = 'conv_layer'
    conv_layer_params = {}
    conv_layer_params['filters'] = 256
    conv_layer_params['kernel_size'] = 9
    conv_layer_params['strides'] = [1, 1]
    conv_layer_params['padding'] = 'VALID'
    conv_layer_params['activation'] = 'relu'
    conv_layer_params['name'] = 'conv1'

    capsule_params.add_params(conv_layer_params,conv_layer_name)

    # First Capsule Layer:
    # [num_output_caps, caps_len,'filters',kernel_size,strides,padding]
    caps_layer_name = 'caps_layer'
    caps_layer_params = {}
    caps_layer_params['filters'] = 256
    caps_layer_params['kernel_size'] = 2
    caps_layer_params['strides'] = [2,2]
    caps_layer_params['padding'] = 'VALID'
    caps_layer_params['padding'] = 'VALID'

    caps_layer_params['n_channels'] = 32
    caps_layer_params['dim_capsule'] = 8
    caps_layer_params['name'] = 'caps_layer'

    capsule_params.add_params(caps_layer_params,caps_layer_name)

    # Digit Capsule Layer:
    digit_layer_name = 'digitcaps_layer'
    digit_layer_params = {}
    digit_layer_params['n_channels'] = 10
    digit_layer_params['dim_capsule'] = 16
    digit_layer_params['name'] = 'digitcaps'
    capsule_params.add_params(digit_layer_params,digit_layer_name )

    # Capsule Decoder:
    decoder_layer = 'decoder_layer'
    decoder_params = {}
    decoder_params['first_dense'] = 512
    decoder_params['second_dense'] = 1024
    decoder_params['name'] = 'decoder'


    capsule_params.add_params(decoder_params,decoder_layer)

    # Generate list of parameter sets::
    list_parameter_sets = []
    n_epoch_list = [50]#, 100, 150]
    lr_list = [0.0005]#, 0.001, 0.005]
    lr_decay_list = [0.25,0.4]#, 0.4, 0.75, 1.5]
    for n_epoch in n_epoch_list:
        for lr in lr_list:
            for lr_decay in lr_decay_list:
                capsule_params = CapsuleTrainingParameters(epochs=n_epoch, lr=lr, lr_decay=lr_decay)
                list_parameter_sets.append(capsule_params)


    #num_params_list = len(list_parameter_sets)
    list_parameter_sets[-1]= CapsuleTrainingParameters(epochs=n_epoch,
                                                       lr=lr,
                                                       lr_decay=lr_decay,
                                                       plot_log=True)

    for i,parameter_set in enumerate(list_parameter_sets):

        patchy_classifier = GraphClassifier(input_shape)
        patchy_classifier.build_the_graph(capsule_params)
        patchy_classifier.train(data, args)

        if i == 0:
            results_df = patchy_classifier.results
        else:
            results_df = pd.concat(results_df,patchy_classifier.results,1)


    # Saving results:
    time_now = datetime.now()
    time_now = '_'.join([str(time_now.date()).replace('-', '_'), str(time_now.hour), str(time_now.minute)])

    results_path = DIR_PATH + 'Results/results_hp_search_{}'.format(time_now)
    results_df.to_csv(results_path)

