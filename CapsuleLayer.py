#!/usr/bin/env python3
import tensorflow as tf
import numpy as np


def squash(vector):
    '''Squashing function.
    Args:
        vector: A 4-D tensor with shape [batch_size, num_caps, vec_len, 1],
    Returns:
        A 4-D tensor with the same shape as vector but
        squashed in 3rd and 4th dimensions.
    '''
    vec_abs = tf.sqrt(tf.reduce_sum(tf.square(vector)))  # a scalar
    scalar_factor = tf.square(vec_abs) / (1 + tf.square(vec_abs))
    vec_squashed = scalar_factor * tf.divide(vector, vec_abs)  # element-wise
    return(vec_squashed)

def routing(u_hat, b_IJ, num_iter):
    # Stopping the routing:
    u_hat_stopped = tf.stop_gradient(u_hat, name='u_hat_stopped')
    print('u_hat shape: ',u_hat_stopped.shape)
    # Routing
    with tf.name_scope('routing'):
        for r_iter in range(num_iter):
            c = tf.nn.softmax(b_IJ,axis=2)
            #assert c.get_shape().as_list() == [5000,1152,10,1,1]
            if r_iter == num_iter-1:
                s_j = tf.reduce_sum(tf.multiply(c,u_hat),axis = 1, keepdims = True)
                v = squash(s_j)
            else:
                s_j = tf.reduce_sum(tf.multiply(c,u_hat_stopped),axis = 1,keepdims=True)
                v = squash(s_j)
                #v_tiled = tf.tile(v,[1, 1152,1,1,1])
                v_tiled = tf.tile(v, [1, u_hat.shape[1].value, 1, 1, 1])
                a = tf.matmul(u_hat_stopped, v_tiled, transpose_a=True)
                b_IJ = b_IJ + a
#         print('c shape: ',c.shape)
#         print('s_j shape: ',s_j.shape)
#         print('v shape: ',v.shape)
#         print('a shape: ',a.shape)
#         print('b_IJ shape: ',b_IJ.shape)
    return v,b_IJ


class CapsuleLayer(object):
    """
    This class creates dynamic capsule Layers
    """
    def __init__(self, num_caps_out, caps_len, with_routing=False):

        self.num_caps_out = num_caps_out
        self.caps_len = caps_len
        self.with_routing = with_routing






    def generate_capsule(self,previous_layer,num_filters=256,kernel_size=9,strides=[2,2],padding='VALID'):
        with tf.name_scope('caps'):
            caps = tf.layers.conv2d(previous_layer,
                                    filters=num_filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding)
            print('caps shape: ', caps.shape)
            caps_size = int(caps.shape[1])
            u_i = tf.reshape(caps, shape=[-1, 32 * caps_size * caps_size, 8, 1])
            # u_i = tf.reshape(caps, shape=[-1, 32 * 6 * 6, 8, 1])
            # caps2 = tf.layers.conv2d(caps1,filters=8,kernel_size=9,strides=[2,2],padding='VALID')
            u_i = squash(u_i)
            print('u_i shape: ', u_i.shape)
            # a_caps1 = squash(caps1)

            return u_i

    def generate_output_capsule(self):
        return



class LayerParameters(object):
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

    capsule_params= LayerParameters()

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



