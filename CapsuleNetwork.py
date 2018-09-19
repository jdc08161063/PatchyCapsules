#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../PatchyCapsules')
from utils_caps import load_image_data

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

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

class CapsuleNetwork(object):

    def __init__(self,batch_size,\
                      num_inputs,\
                      num_outputs,\
                      num_iter):
        # Some constants:
        self.batch_size = batch_size  # None#X_train.shape[0]
        # n_batches = int(X_train.shape[0]/batch_size)
        self.num_inputs = num_inputs # 128
        self.channels = 1
        self.height, self.width = 28, 28
        # variables:
        self.num_iter = num_iter
        self.num_outputs = num_outputs


    def set_training_parameters(self,X_train,print_config = False):
        self.train_size = X_train.shape[0]
        self.height,self.width = X_train.shape[1:3]
        if len(X_train.shape) == 3:
            self.num_channels = 1
        else:
            self.num_channels = X_train.shape[-1]

        if print_config == True:
            print('shape of training set : ', X_train.shape)




    def train_model(self,X_train,y_train, X_test, y_test, num_epochs = 10):

        self.set_training_parameters(X_train)
        num_batches = n_batches = int(X_train.shape[0]/self.batch_size)
        num_batches_test = int(X_test.shape[0] / self.batch_size)


        # Create Placeholders:
        self.X_place, self.y_place = self.set_placeholders(self.batch_size)

        # Define forward pass:
        self.build_capsule_arch(self.batch_size,self.X_place)

        print('Capsule architecture built')
        loss = self.calculate_total_loss(self.X_place, self.y_place,\
                                         y_train, self.batch_size)
        print('Loss computed')

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer()
            self.training_op = self.optimizer.minimize(loss)

        with tf.name_scope('evalute_on_training'):
            self.train_error = []


        with tf.Session() as sess:

            sess.run(tf.initialize_all_variables())

            for epoch in range(num_epochs):

                print('Epoch {} :'.format(epoch))
                # Shuffling and dividing into batches:
                shuffled_idx = np.random.permutation(self.train_size)
                X_batches = np.array_split(X_train[shuffled_idx], num_batches)
                y_batches = np.array_split(y_train[shuffled_idx], num_batches)

                y_pred_train = np.array([])
                # SGD:
                for X_batch, y_batch in zip(X_batches, y_batches):
                    sess.run(self.training_op, feed_dict={self.X_place: X_batch,
                                                          self.y_place: y_batch})

                    y_pred_batch = sess.run(self.get_predictions(self.v),
                                            feed_dict={self.X_place: X_batch, self.y_place: y_batch})

                    #print('y_pred_batch shape : ', y_pred_batch.shape)
                    #print('y_pred_train shape : ', y_pred_train.shape)
                    y_pred_train= np.hstack([y_pred_train, y_pred_batch])

                train_accuracy = sum(y_pred_train == y_train[shuffled_idx]) / len(y_pred_train)
                print('accuracy train: {}'.format(train_accuracy))


                # Testing :
                X_batches_test = np.array_split(X_test, num_batches_test)
                y_batches_test = np.array_split(y_test, num_batches_test)
                y_pred_test = np.array([])

                for X_batch, y_batch in zip(X_batches_test, y_batches_test):
                    y_pred_batch = sess.run(self.get_predictions(self.v),
                                            feed_dict={self.X_place: X_batch, self.y_place: y_batch})

                    y_pred_test = np.hstack([y_pred_test, y_pred_batch])

                test_accuracy = sum(y_pred_test == y_test) / len(y_pred_test)
                print('accuracy test: {}'.format(test_accuracy))


    def set_placeholders(self,batch_size):
        # Initialize graph:
        reset_graph()
        # Placeholders:
        X = tf.placeholder(shape=(batch_size, self.height, self.width, self.channels), dtype=tf.float32)
        y = tf.placeholder(tf.int32, shape=(batch_size), name="y")
        return X,y


    def build_capsule_arch(self,batch_size, X_place_train):
        # Building the graph:
        conv1 = self.build_first_cnn_layer(X_place_train)
        u_i = self.build_first_capsule_layer(conv1)
        self.v, self.b_IJ = self.build_digit_caps_layer(batch_size, u_i)


    def evaluate(self, X, y, sess):
        with tf.name_scope('evaluate'):

            v_pred = sess.run(self.v,feed_dict={self.X_place:X})


    def get_predictions(self, v):
        reshaped_v = tf.squeeze(self.v)
        v_norm = tf.sqrt(tf.reduce_sum(tf.square(reshaped_v), axis=2))
        #print('y norm shape : ', v_norm.shape)
        v_softmax = tf.nn.softmax(v_norm, axis=1)
        #print('y soft shape : ', v_softmax.shape)
        y_pred = tf.argmax(v_norm, axis=1)
        #print('y_pred shape : ', y_pred.shape)
        return y_pred







    # Building the architecture:

    def build_first_cnn_layer(self, X, num_filters= 256, kernel_size= 9):
        # First Conv layer:
        with tf.name_scope('cnn'):
            conv = tf.layers.conv2d(X, filters=num_filters, kernel_size=kernel_size, strides=[1, 1], padding='VALID')
            print('first cnn layer shape : ',conv.shape)
            return conv

    def build_first_capsule_layer(self,conv):
        # First Capsule LAyer:
        with tf.name_scope('caps'):
            caps = tf.layers.conv2d(conv, filters=256, kernel_size=9, strides=[2, 2], padding='VALID')
            print('caps shape: ', caps.shape)
            caps_size = int(caps.shape[1])
            u_i = tf.reshape(caps, shape=[-1, 32 * caps_size * caps_size, 8, 1])
            #u_i = tf.reshape(caps, shape=[-1, 32 * 6 * 6, 8, 1])
            # caps2 = tf.layers.conv2d(caps1,filters=8,kernel_size=9,strides=[2,2],padding='VALID')
            u_i = squash(u_i)
            print('u_i shape: ', u_i.shape)
            # a_caps1 = squash(caps1)

            return u_i




    def build_digit_caps_layer(self, batch_size, u_i):
        u_i_size = u_i.shape[1].value
        with tf.variable_scope('final_layer'):

            w_initializer = np.random.normal(size=[1, u_i_size , 10, 8, 16], scale=0.01)
            W = tf.Variable(w_initializer, dtype=tf.float32)

            # repeat W with batch_size times to shape [batch_size, 1152, 8, 16]
            W = tf.tile(W, [batch_size, 1, 1, 1, 1])  #  -1 instead of Batch size
            print('W shape: ', W.shape)
            # calc u_ahat
            # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 16, 1]
            u_i = tf.reshape(u_i, shape=(batch_size, -1, 1, u_i.shape[-2].value, 1))  #  -1 instead of Batch size
            u_i = tf.tile(u_i, [1, 1, 10, 1, 1])
            print('u_i shape: ', u_i.shape)
            u_hat = tf.matmul(W, u_i, transpose_a=True)
            print('u_hat shape: ', u_hat.shape)

            # Routing:
            with tf.variable_scope('routing'):
                # Initialize constants:
                b_IJ = tf.zeros([batch_size, u_i.shape[1].value, 10, 1, 1], dtype=np.float32)
                print('b_IJ shape: ', b_IJ.shape)
                v, b_IJ = routing(u_hat, b_IJ, self.num_iter)
                print('After routing : ')
                print('v shape: ', v.shape)
                return v, b_IJ

    def calculate_total_loss(self, X_place, y_place, y_train, batch_size): # delete v
        margin_loss = self.calculate_margin_loss(y_place)
        reg_loss = self.calculate_reg_loss(X_place, y_train, batch_size)
        # Total loss:
        with tf.name_scope('Loss'):
            # margin_loss = calculate_margin_loss(y_batch,v,n_outputs)
            loss = tf.reduce_sum(margin_loss) + reg_loss
            return loss


    def calculate_margin_loss(self, y_place):
        # def calculate_margin_loss(y_batch,v,n_outputs):
        m_pos = 0.9
        m_neg = 0.1
        lambda_const = 0.5

        with tf.name_scope('margin_loss'):
            reshaped_v = tf.squeeze(self.v)
            # v_norm = tf.map_fn(lambda x: tf.norm(x,axis=1), v)
            v_norm = tf.sqrt(tf.reduce_sum(tf.square(reshaped_v), axis=2))
            # v_softmax = tf.nn.softmax(v_norm, axis =1)
            # y_pred = tf.argmax(v_norm,axis=1)
            # t_k = tf.equal(y_pred,y_train)
            t_k = tf.one_hot(y_place, self.num_outputs)

            print('t_k shape :', t_k.shape)
            # Loss:
            max_l = tf.maximum(tf.cast(0, tf.float32), tf.square(m_pos - v_norm))
            max_r = tf.maximum(tf.cast(0, tf.float32), tf.square(v_norm - m_neg))

            print('max_l shape: ', max_l.shape)

            margin_loss = tf.multiply(t_k, tf.square(max_l)) + lambda_const * tf.multiply((1 - t_k), tf.square(max_r))
            return margin_loss



    def calculate_reg_loss(self, X_place, y_train, batch_size): #delete v as input
        # Regularizer Decoder:
        #tf.global_variables_initializer()
        with tf.name_scope('mask'):
            # Masking:
            v_list = []
            for i, j in zip(range(batch_size), y_train):  # y_train ?
                v_list.append(tf.reshape(tf.squeeze(self.v)[i][j, :], [1, 16]))
            v_masked = tf.concat(v_list, axis=0)

        with tf.name_scope('decoder'):
            # 2 FC Relu:
            dec1 = tf.layers.dense(inputs=v_masked, units=512, activation=tf.nn.relu)
            dec2 = tf.layers.dense(inputs=dec1, units=512, activation=tf.nn.relu)
            # 1 FC Sigmoid:
            dec3 = tf.layers.dense(inputs=dec2, units=self.num_inputs, activation=tf.nn.sigmoid)
            #loss_reg = tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(X, [batch_size, num_inputs * num_channels]) - dec3)))
            loss_reg = tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(X_place, [batch_size, self.num_inputs*self.num_channels]) - dec3)))

            return loss_reg




if __name__ == "__main__":
    # Downloading mnist dataset:

    dataset = 'mnist'
    #dataset = 'cifar'

    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        # Train set
        X_train = X_train.astype(np.float32).reshape(-1, 28, 28, 1) / 255.0
        y_train = y_train.astype(np.int32)
        # Test set:
        X_test = X_test.astype(np.float32).reshape(-1, 28, 28, 1) / 255.0
        y_test = y_test.astype(np.int32)
        training_size = 2500
        # Validation set:
        X_train, X_valid = X_train[:training_size], X_train[training_size:]
        y_train, y_valid = y_train[:training_size], y_train[training_size:]

    else:
        train_file_path = '../../others/CIFAR10-img-classification-tensorflow/cifar-10-batches-py/data_batch_1'
        test_file_path = '../../others/CIFAR10-img-classification-tensorflow/cifar-10-batches-py/test_batchs'
        X_train, y_train = load_image_data(train_file_path)
        X_test, y_test = load_image_data(train_file_path)

        # Some constants:

        # n_batches = int(X_train.shape[0]/batch_size)
        num_inputs = 28 * 28
        channels = 1
        # n_output_conv1 = (20,20,256)
        height, width = 28, 28
        # variables:



    batch_size = 250
    height, width, num_channels = X_train.shape[1:]
    num_inputs = height * width
    num_outputs = 10
    num_iter = 5

    print('TensorFlow Version: {}'.format(tf.__version__)) 
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    caps = CapsuleNetwork(batch_size,num_inputs,\
                          num_outputs,num_iter)


    #
    caps.train_model(X_train,y_train,X_test,y_test, num_epochs = 20)





