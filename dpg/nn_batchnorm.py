from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS

import numpy as np

np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import time
import math
import cPickle
import gym as gym


class nn_batchnorm():

    def __init__(self, input_low, input_high, car):

        self.input_high = input_high
        self.input_low = input_low

        self.sigma = np.identity(2)*0.5

        self.num_outputs = 1
        self.num_inputs = 2
        self.batchsize = 50

        self.sess = tf.InteractiveSession()
        self.num_neurons_layer1 = 100
        self.num_neurons_layer2 = 100

        # parameteres for network training
        batchsize = 50
        self.batch_train_data = np.zeros((batchsize,2))
        self.batch_labels = np.zeros(batchsize).reshape((batchsize,1))
        self.batchindex = 0

        self.step = 0

        self.summaries_dir = './regressiontest/batchnorm'

        self.car1 = car

    def add_to_batch(self, state, mu):

        self.batch_train_data[self.batchindex,:] = state
        self.batch_labels[self.batchindex] = mu
        self.batchindex += 1

        if self.batchindex == 50:
            self.batchindex = 0
            self.pefrom_train_step()

    def plot_learned_function(self):

        print('plotting the mu() policy learned by NN')

        resolution = 50

        # values to evaluate policy at
        x_range = np.linspace(self.input_high[0], self.input_low[0], resolution)
        v_range = np.linspace(self.input_high[1], self.input_low[1], resolution)

        # get actions in a grid
        vals = np.zeros((resolution, resolution))
        for i, x in enumerate(x_range):
            for j, v in enumerate(v_range):
                x_ = np.array([x,v],dtype=np.float32).reshape((1,2))
                vals[j,i]= self.eval_trained_function(x_)[0]

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_range, v_range)
        ax.plot_surface(X, Y, vals, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=True)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_zlabel("action")
        plt.show()


    def initialize_training(self, sess):

        # Create a multilayer model.

        # Input placehoolders
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, self.num_inputs], name='x-input')
            self.y_ = tf.placeholder(tf.float32, [None, self.num_outputs], name='y-input')

        # We can't initialize these variables to 0 - the network will get stuck.
        def weight_variable(shape):
            """Create a weight variable with appropriate initialization."""
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            """Create a bias variable with appropriate initialization."""
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def variable_summaries(var, name):
            """Attach a lot of summaries to a Tensor."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.scalar_summary('mean/' + name, mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
                tf.scalar_summary('sttdev/' + name, stddev)
                tf.scalar_summary('max/' + name, tf.reduce_max(var))
                tf.scalar_summary('min/' + name, tf.reduce_min(var))
                tf.histogram_summary(name, var)

        def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
            """Reusable code for making a simple neural net layer.

            It does a matrix multiply, bias add, and then uses relu to nonlinearize.
            It also sets up name scoping so that the resultant graph is easy to read,
            and adds a number of summary ops.
            """
            # Adding a name scope ensures logical grouping of the layers in the graph.
            with tf.name_scope(layer_name):
                # This Variable will hold the state of the weights for the layer
                with tf.name_scope('weights'):
                    weights = weight_variable([input_dim, output_dim])
                    variable_summaries(weights, layer_name + '/weights')
                with tf.name_scope('biases'):
                    biases = bias_variable([output_dim])
                    variable_summaries(biases, layer_name + '/biases')
                with tf.name_scope('Wx_plus_b'):
                    preactivate = tf.matmul(input_tensor, weights) + biases
                    tf.histogram_summary(layer_name + '/pre_activations', preactivate)
                activations = act(preactivate, 'activation')
                tf.histogram_summary(layer_name + '/activations', activations)
                return activations

        self.phase_train = tf.placeholder(tf.bool, name='phase_train')

        def batch_norm(x,shape_x, scope):
            """
            Batch normalization on convolutional maps.
            Args:
                x:           Tensor, 4D BHWD input maps
                shape_x:       integer, depth of input maps
                scope:       string, variable scope
            Return:
                normed:      batch-normalized maps
            """
            with tf.variable_scope(scope):
                beta = tf.Variable(tf.constant(0.0, shape=[shape_x]),
                                             name='beta', trainable=True)
                gamma = tf.Variable(tf.constant(1.0, shape=[shape_x]),
                                              name='gamma', trainable=True)
                batch_mean, batch_var = tf.nn.moments(x, axes= [0], name='moments')
                ema = tf.train.ExponentialMovingAverage(decay=0.9)

                def mean_var_with_update():
                    ema_apply_op = ema.apply([batch_mean, batch_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(batch_mean), tf.identity(batch_var)

                # summary_batch_mean = ema.average(batch_mean)
                # summary_var_mean = ema.average(batch_var)
                # variable_summaries(summary_batch_mean, scope + '/batch_mean')
                # variable_summaries(summary_var_mean, scope + '/batch_mean')

                mean, var = tf.cond(self.phase_train,
                                    mean_var_with_update,
                                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
                normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

            return normed, mean, var, beta, gamma

        x_bn, mean, var, beta, gamma = batch_norm(self.x, shape_x= self.num_inputs, scope='input_bn')

        with tf.name_scope('hidden_1'):
            a1 = nn_layer(x_bn, self.num_inputs, self.num_neurons_layer1, 'layer1')
            a1_bn, _, _, _, _ = batch_norm(a1, shape_x= self.num_neurons_layer1, scope= 'layer_1_bn')


        with tf.name_scope('hidden_2'):
            a2 = nn_layer(a1_bn, self.num_neurons_layer1, self.num_neurons_layer2, 'layer2')
            a2_bn, _, _, _, _ = batch_norm(a2, shape_x= self.num_neurons_layer1, scope= 'layer_2_bn')

        self.y = nn_layer(a2_bn, self.num_neurons_layer2, self.num_outputs, 'layer3', act=tf.identity)

        with tf.name_scope('mse'):
            squ_diff = tf.pow(self.y_- self.y, 2)
            with tf.name_scope('total'):
                self.mean_squared_error = tf.reduce_mean(squ_diff)
            tf.scalar_summary('mean squared error', self.mean_squared_error)

        self.learning_rate = tf.placeholder(tf.float32, shape= [], name='learning_rate')
        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mean_squared_error)


        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        self.merged = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(self.summaries_dir, sess.graph)
        tf.initialize_all_variables().run()

    def pefrom_train_step(self):

        l_r = 1e-3

        def feed_dict():

            xs = self.batch_train_data

            ys = self.batch_labels

            return {self.x: xs, self.y_: ys, self.phase_train: True, self.learning_rate: l_r}

        summary, _, mse_val = self.sess.run([self.merged, self.train_step, self.mean_squared_error],feed_dict=feed_dict())

        if self.step%10 == 0:
            self.train_writer.add_summary(summary, self.step )

        self.step += 1

        if self.step % 500 == 0:
            print('result after minibatch no. {} : mean squared error: {}'.format(self.step, mse_val))
            print('batch train data', self.batch_train_data)
            print('batch train labels', self.batch_labels)
            self.plot_learned_function()

            num_steps= self.test_learned_policy()
            print('episode length using learned policy:',num_steps)

    def eval_trained_function(self, input):
        return self.y.eval(feed_dict= {self.x : input ,self.phase_train: False})

    def test_learned_policy(self, limit=20000, enable_render=False):

        def apply_limits(action):
            if action < self.car1.action_limits[0]:
                action = self.car1.action_limits[0]
            if action > self.car1.action_limits[1]:
                action = self.car1.action_limits[1]
            return action

        episode = []
        state = self.car1.env.reset()

        count = 0
        done = False

        while ( not done ):

            if len(episode)>limit:
                return episode

            count += 1

            print('state',state)
            action = self.eval_trained_function(state.reshape((1,2)))[0]
            print(action)
            action = apply_limits(action)

            state, reward, done, info = self.car1.env.step(action)

            if enable_render:
                self.car1.env.render()
                # print("step no. {}".format(count))

        return count

    def main(self):
        if tf.gfile.Exists(self.summaries_dir):
            tf.gfile.DeleteRecursively(self.summaries_dir)
        tf.gfile.MakeDirs(self.summaries_dir)

        self.initialize_training(self.sess)