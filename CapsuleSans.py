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
from PIL import Image

from keras import layers, models, optimizers
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

from utils import combine_images
from utils import plot_log

from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
sys.path.append('./PatchyTools/')
from GraphConverter import GraphConverter
from DropboxLoader import DropboxLoader
from CapsuleParameters import CapsuleParameters
from CapsuleParameters import CapsuleTrainingParameters



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
        # unpacking the data
        # (x_train, y_train), (x_test, y_test) = data

        self.import_data(data)

        # if not hasattr(self, 'train_model'):
        #     self.build_the_graph()

        # callbacks
        log = callbacks.CSVLogger(args.save_dir + '/log.csv')
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
        plot_log(args.save_dir + '/log.csv', show=True)

        #return self.train_model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-' * 30 + 'Begin: test' + '-' * 30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()


def manipulate_latent(model, data, args):
    print('-' * 30 + 'Begin: manipulate' + '-' * 30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:, :, dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)



if __name__ == "__main__":


    # setting the hyper parameters

    args = CapsuleTrainingParameters()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    #(x_train, y_train), (x_test, y_test) = load_cifar()  # load_mnist()
    # Getting the training data:


    dataset_name = 'DD'
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


    #args.data_augmentation = True
    patchy_classifier = GraphClassifier(input_shape)
    patchy_classifier.build_the_graph(capsule_params)
    patchy_classifier.train(data, args)


