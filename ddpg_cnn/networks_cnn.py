import tensorflow as tf
import numpy as np


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def hist_summaries(*args):
    return tf.merge_summary([tf.histogram_summary(t.name, t) for t in args])


def init_weight(shape, fanin=None, value = None):
    if value != None:
        return tf.constant(value,shape= shape)

    v = 2 / np.sqrt(fanin)

    # tf.random_uniform(shape, minval=-v, maxval=v)
    return tf.truncated_normal(shape, mean= 0, stddev=v)

class cnn_config:
    def __init__(self, dimO):
        self.input_maps = 3

        self.kernel_size = 4
        self.no_kernels_conv1 = 32
        self.no_kernels_conv2 = 32
        self.no_kernels_conv3 = 32

        self.num_fc1 = 200
        self.num_fc2 = 200


        self.fanin_conv1 = self.kernel_size*self.kernel_size*self.input_maps
        print('fanin_conv1:', self.fanin_conv1)
        self.fanin_conv2 = self.kernel_size*self.kernel_size*self.no_kernels_conv1
        self.fanin_conv3 = self.kernel_size*self.kernel_size*self.no_kernels_conv2
        self.fanin_fc1 = dimO[0]/8*dimO[1]/8*self.no_kernels_conv3
        self.fanin_fc2 = self.num_fc1
        self.fanin_fc3 = self.num_fc2

        # 4*4*32*3 + 4*4*32*32 + 4*4*32*32 + 32*8*8*200 + 200*200
        print('total number of weights: ', self.input_maps * self.kernel_size * self.kernel_size * self.no_kernels_conv1 +
              self.no_kernels_conv1 * self.kernel_size * self.kernel_size * self.no_kernels_conv2 +
              self.no_kernels_conv2 * self.kernel_size * self.kernel_size * self.no_kernels_conv3 +
              self.no_kernels_conv3 * dimO[0] * dimO[1] / 8 / 8 * self.num_fc1 +
              self.num_fc1 * self.num_fc2)

def theta_mu(dimA, c):

    with tf.variable_scope("theta_mu"):
        return [tf.Variable(init_weight([c.kernel_size, c.kernel_size, c.input_maps, c.no_kernels_conv1], fanin= c.fanin_conv1), name='W1_conv'),
                tf.Variable(init_weight([c.no_kernels_conv1], value= 0.), name='b1_conv'),

                tf.Variable(init_weight([c.kernel_size, c.kernel_size, c.no_kernels_conv1, c.no_kernels_conv2], fanin= c.fanin_conv2), name='W2_conv'),
                tf.Variable(init_weight([c.no_kernels_conv2], value= 0.), name='b2_conv'),

                tf.Variable(init_weight([c.kernel_size, c.kernel_size, c.no_kernels_conv2, c.no_kernels_conv3], fanin= c.fanin_conv3), name='W3_conv'),
                tf.Variable(init_weight([c.no_kernels_conv3], value= 0.), name='b3_conv'),

                tf.Variable(init_weight([c.fanin_fc1, c.num_fc1], fanin=c.fanin_fc1), name='W_fc1'),
                tf.Variable(init_weight([c.num_fc1], value= 0.), name='b_fc1'),

                tf.Variable(init_weight([c.num_fc1, c.num_fc2], fanin= c.fanin_fc2), name='W_fc2'),
                tf.Variable(init_weight([c.num_fc2], value= 0.), name='b_fc2'),

                tf.Variable(init_weight([c.num_fc1, dimA], fanin= c.fanin_fc3), name='W_fc3'),
                tf.Variable(init_weight([dimA], value= 0.), name='b_fc3'),
                ]


def mu_net(obs, theta, c, name='mu_net'):
    with tf.variable_op_scope([obs], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        h1 = tf.nn.relu(conv2d(h0, theta[0]) + theta[1])
        h2 = tf.nn.relu(conv2d(h1, theta[2]) + theta[3])
        h3 = tf.nn.relu(conv2d(h2, theta[4]) + theta[5])
        h_3_flat = tf.reshape(h3, [-1, c.fanin_fc1])
        h4 = tf.nn.relu(tf.matmul(h_3_flat, theta[6]) + theta[7])
        h5 = tf.nn.relu(tf.matmul(h4, theta[8]) + theta[9])
        h6 = tf.nn.tanh(tf.matmul(h5, theta[10]) + theta[11])

        summary = hist_summaries(h0, h1, h2, h3, h4, h5, h6)
        return h6, summary


def theta_q(dimO, dimA, c):

    with tf.variable_scope("theta_q"):
        return [tf.Variable(init_weight([c.kernel_size, c.kernel_size, c.input_maps, c.no_kernels_conv1], fanin= c.fanin_conv1), name='W1_conv'),
                tf.Variable(init_weight([c.no_kernels_conv1], value= 0.), name='b1_conv'),

                tf.Variable(init_weight([c.kernel_size, c.kernel_size, c.no_kernels_conv1, c.no_kernels_conv2], fanin= c.fanin_conv2), name='W2_conv'),
                tf.Variable(init_weight([c.no_kernels_conv2], value= 0.), name='b2_conv'),

                tf.Variable(init_weight([c.kernel_size, c.kernel_size, c.no_kernels_conv2, c.no_kernels_conv3], fanin= c.fanin_conv3), name='W3_conv'),
                tf.Variable(init_weight([c.no_kernels_conv3], value= 0.), name='b3_conv'),

                tf.Variable(init_weight([c.no_kernels_conv3 * dimO[0]/8 * dimO[1]/8 + dimA, c.num_fc1], fanin= c.fanin_fc1), name='W_fc1'),
                tf.Variable(init_weight([c.num_fc1], value= 0.), name='b_fc1'),

                tf.Variable(init_weight([c.num_fc1, c.num_fc2], fanin= c.fanin_fc2), name='W_fc2'),
                tf.Variable(init_weight([c.num_fc2], value= 0.), name='b_fc2'),

                tf.Variable(init_weight([c.num_fc1, 1], fanin= c.fanin_fc3), name='W_fc3'),
                tf.Variable(init_weight([1], value= 0.), name='b_fc3'),
                ]

def q_net(obs, act, theta, c, name='qnet'):

    with tf.variable_op_scope([obs], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        h1 = tf.nn.relu(conv2d(h0, theta[0]) + theta[1])
        h2 = tf.nn.relu(conv2d(h1, theta[2]) + theta[3])
        h3 = tf.nn.relu(conv2d(h2, theta[4]) + theta[5])
        h_3_flat = tf.reshape(h3, [-1, c.fanin_fc1])
        h_3_concat = tf.concat(1, [h_3_flat, act])
        h4 = tf.nn.relu(tf.matmul(h_3_concat, theta[6]) + theta[7])
        h5 = tf.nn.relu(tf.matmul(h4, theta[8]) + theta[9])
        h6 = tf.identity(tf.matmul(h5, theta[10]) + theta[11])

        summary = hist_summaries(h0, h1, h2, h3, h4, h5, h6)
        return h6, summary


def exponential_moving_averages(theta, tau=0.001):
    ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
    update = ema.apply(theta)  # also creates shadow vars
    averages = [ema.average(x) for x in theta]
    return averages, update

