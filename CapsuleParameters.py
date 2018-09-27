#!/usr/bin/env python3

import os


class CapsuleParameters(object):
    def __init__(self):
        self.layer_parameters = {}

    def add_params(self, parameters_dict, layer_name):
        self.layer_parameters[layer_name] = parameters_dict

    def get_layer_params(self, layer_name):
        return self.layer_parameters[layer_name]
    # def get_conv_params(self,layer):
    #     return layer['filters'],layer['kernel_size'], \
    #            layer['strides'],layer['padding']

    # def get_caps_params(self,layer):
    #     #layer = self.layer_parameters[layer_name]
    #     return layer['n_channels'], \
    #            layer['dim_capsule']


class CapsuleTrainingParameters(object):
    def __init__(self,
                 epochs=100,
                 batch_size=100,
                 lr=0.001,
                 lr_decay=0.9,
                 lam_recon=0.392,
                 routing=5,
                 shift_fraction=0.1,
                 debug=False,
                 save_dir='./result',
                 data_augmentation=False,
                 testing=False,
                 digit=5,
                 weights=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.lam_recon = lam_recon
        self.debug = debug
        self.routing = routing
        self.shift_fraction = shift_fraction
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.data_augmentation = data_augmentation
        self.testing = testing
        self.digit = digit
        self.weights = weights


if __name__ == '__main__':
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
    caps_layer_params['kernel_size'] = 9
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
    decoder_params['first_dense'] = 512
    decoder_params['second_dense'] = 1024
    decoder_params['name'] = 'decoder'

    capsule_params.add_params(decoder_params, decoder_layer)
