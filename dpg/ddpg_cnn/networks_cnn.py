import tensorflow as tf
import numpy as np




def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

learning_rate = tf.placeholder(tf.float32, shape=[])

global_step = tf.Variable(0, name='global_step', trainable=False)

# losses
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]), name= 'cross_entropy')


######


def hist_summaries(*args):
    return tf.merge_summary([tf.histogram_summary(t.name, t) for t in args])

def fanin_init(shape, fanin=None):
    fanin = fanin or shape[0]
    v = 1 / np.sqrt(fanin)
    return tf.random_uniform(shape, minval=-v, maxval=v)
#
# def fanin_init(shape, fanin=None):
#         if fanin != None:
#             return tf.constant(0.,shape= shape)
#
#         fanin = fanin or shape[0]
#         v = 1 / np.sqrt(fanin)
#
#         # tf.random_uniform(shape, minval=-v, maxval=v)
#         return tf.truncated_normal(shape, v)


def theta_hidden(dimO, l1, l2):
    with tf.variable_scope("theta_hidden"):
        return [tf.Variable(fanin_init([dimO, l1]), name='1w'),
                tf.Variable(fanin_init([l1], dimO), name='1b'),
                tf.Variable(fanin_init([l1, l2]), name='2w'),
                tf.Variable(fanin_init([l2], l1), name='2b'),
                ]


def theta_network(obs, theta, name='hidden'):
    with tf.variable_op_scope([obs], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        h1 = tf.nn.relu(conv2d(h0, theta[0]) + theta[1])
        h2 = tf.nn.relu(conv2d(h0, theta[2]) + theta[3])
        h_2_flat = tf.reshape(h2, [-1, 7*7*64])
        h3 = tf.nn.relu(tf.matmul(h_2_flat, theta[4]) + theta[5])
        h4 = tf.nn.relu(tf.matmul(h3, theta[6]) + theta[7])

        summary = hist_summaries(h0, h1, h2, h_2_flat, h3, h4)
        return h4, summary

def q_network(obs, act, theta, name='hidden'):
    with tf.variable_op_scope([obs], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        h1 = tf.nn.relu(conv2d(h0, theta[0]) + theta[1])
        h2 = tf.nn.relu(conv2d(h0, theta[2]) + theta[3])
        h_2_flat = tf.reshape(h2, [-1, 7*7*64])
        h_2_concat = tf.concat(1, [h_2_flat, act])
        h3 = tf.nn.relu(tf.matmul(h_2_concat, theta[4]) + theta[5])
        h4 = tf.nn.relu(tf.matmul(h3, theta[6]) + theta[7])

        summary = hist_summaries(h0, h1, h2, h3, h4)
        return h4, summary

def q_net(self,obs, act, theta, name="qfunction"):
    with tf.variable_op_scope([obs, act], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        h1 = tf.nn.relu(tf.matmul(h0, theta[0]) + theta[1], name='h1')
        h1a = tf.concat(1, [h1, act])
        h2 = tf.nn.relu(tf.matmul(h1a, theta[2]) + theta[3], name='h2')
        qs = tf.matmul(h2, theta[4]) + theta[5]
        q = tf.squeeze(qs, [1], name='h3-q')

        summary = self.hist_summaries(h0, h1, h2, q)
        return q, summary


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

