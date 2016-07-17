import tensorflow as tf
import numpy as np


def hist_summaries(*args):
    return tf.merge_summary([tf.histogram_summary(t.name, t) for t in args])

def fanin_init(shape, fanin=None):
    fanin = fanin or shape[0]
    v = 1 / np.sqrt(fanin)
    return tf.random_uniform(shape, minval=-v, maxval=v)


def theta_hidden(dimO, l1, l2):
    with tf.variable_scope("theta_hidden"):
        return [tf.Variable(fanin_init([dimO, l1]), name='1w'),
                tf.Variable(fanin_init([l1], dimO), name='1b'),
                tf.Variable(fanin_init([l1, l2]), name='2w'),
                tf.Variable(fanin_init([l2], l1), name='2b'),
                ]


def hidden_layers(obs, theta, name='hidden'):
    with tf.variable_op_scope([obs], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        h1 = tf.nn.relu(tf.matmul(h0, theta[0]) + theta[1], name='h1')
        h2 = tf.nn.relu(tf.matmul(h1, theta[2]) + theta[3], name='h2')

        summary = hist_summaries(h0, h1, h2)
        return h2, summary


def theta_fc(inp_dim, out_dim):
    with tf.variable_scope("theta_hidden"):
        return [tf.Variable(fanin_init([inp_dim, out_dim]), name='1w'),
                tf.Variable(fanin_init([out_dim], inp_dim), name='1b')
                ]


def fc_layer(input, theta, act_func, name):

    with tf.variable_op_scope([input], name, name):
        h1 = act_func(tf.matmul(input, theta[0]) + theta[1], name='h1')
        summary = hist_summaries(h1)
        return h1, summary


def exponential_moving_averages(theta, tau=0.001):
    ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
    update = ema.apply(theta)  # also creates shadow vars
    averages = [ema.average(x) for x in theta]
    return averages, update

