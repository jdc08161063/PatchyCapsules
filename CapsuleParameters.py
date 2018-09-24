#!/usr/bin/env python3


# class CapsuleLayer(object):
#     """
#     This class creates dynamic capsule Layers
#     """
#     def __init__(self, num_caps_out, caps_len, with_routing=False):
#
#         self.num_caps_out = num_caps_out
#         self.caps_len = caps_len
#         self.with_routing = with_routing
#



class CapsuleParameters(object):
    def __init__(self):
        self.layer_parameters = []

    def add_params(self,parameters):
        self.layer_parameters.append(parameters)

    def get_conv_params(self,layer):
        return layer['num_filters'],layer['kernel_size'], \
               layer['strides'],layer['padding']
    def get_caps_params(self,layer):
        return layer['caps_num_out'], \
               layer['caps_len']


if __name__=='__main__':

    capsule_params= CapsuleParameters()

    # First conv layer: (num_filters, kernel_size)
    conv_layer_params = {}
    conv_layer_params['num_filters'] = 256
    conv_layer_params['kernel_size'] = 9
    conv_layer_params['strides'] = [1, 1]
    conv_layer_params['padding'] = 'VALID'

    capsule_params.add_params(conv_layer_params)

    # First Capsule Layer:
    # [num_output_caps, caps_len, num_filters,kernel_size,strides,padding]
    caps_layer_params = {}
    caps_layer_params['num_filters'] = 256
    caps_layer_params['kernel_size'] = 9
    caps_layer_params['strides'] = [2,2]
    caps_layer_params['padding'] = 'VALID'

    caps_layer_params['caps_num_out'] = 32
    caps_layer_params['caps_len'] = 8

    capsule_params.add_params(caps_layer_params)

    # Digit Capsule Layer:
    digit_layer_params = {}
    digit_layer_params['caps_num_out'] = 10
    digit_layer_params['caps_len'] = 16

    capsule_params.add_params(digit_layer_params)

    # Capsule Decoder:
    decoder_params = {}
    decoder_params['first_layer'] = 512
    decoder_params['second_layer'] = 1024

    capsule_params.add_params(decoder_params)

