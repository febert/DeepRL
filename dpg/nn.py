from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', './regressiontest/regressionlogs', 'Summaries directory')

import numpy as np

np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import time
import math
import cPickle
import gym as gym


class nn():

    def __init__(self):

        self.target_func_range_max = (3,3)
        self.target_func_range_min = (0,0)

        self.sigma = np.identity(2)*0.5

        self.num_outputs = 1
        self.num_inputs = 2
        self.batchsize = 50

        self.sess = tf.InteractiveSession()
        self.num_neurons_layer1 = 100
        self.num_neurons_layer2 = 100

        self.keep_prob_val =  0.9

        # parameteres for network training
        batchsize = 50
        self.batch_train_data = np.zeros((batchsize,2))
        self.batch_labels = np.zeros(batchsize).reshape((batchsize,1))
        self.batchindex = 0

        self.step = 0

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
        x_range = np.linspace(self.target_func_range_min[0], self.target_func_range_max[0], resolution)
        v_range = np.linspace(self.target_func_range_min[1], self.target_func_range_max[1], resolution)

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

        def make_input_output_histogramm(inp, labels):
            x_slice = tf.slice(input_= inp, begin= [0,0], size= [-1,1])
            v_slice = tf.slice(input_= inp, begin= [0,1], size= [-1,1])

            tf.histogram_summary('x_input', x_slice)
            tf.histogram_summary('v_input', v_slice)
            tf.histogram_summary('labels hist', labels)

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

        self.keep_prob = tf.placeholder(dtype= tf.float32, shape= [], name= 'keep_prob')

        def rescale_input(input):
            c = tf.constant([1,10], shape=[1, 2], dtype=tf.float32)
            c_broadcast = tf.tile(c, [tf.shape(input)[0], 1])
            return tf.mul(input, c_broadcast)

        self.x_rescaled = rescale_input(self.x)
        make_input_output_histogramm(self.x_rescaled, self.y_)

        with tf.name_scope('hidden_1'):
            a1 = nn_layer(self.x_rescaled, self.num_inputs, self.num_neurons_layer1, 'layer1')
            with tf.name_scope('dropout'):
                dropped1 = tf.nn.dropout(a1, self.keep_prob)

        with tf.name_scope('hidden_2'):
            a2 = nn_layer(dropped1, self.num_neurons_layer1, self.num_neurons_layer2, 'layer2')
            with tf.name_scope('dropout'):
                dropped2 = tf.nn.dropout(a2, self.keep_prob)

        self.y = nn_layer(dropped2, self.num_neurons_layer2, self.num_outputs, 'layer3', act=tf.identity)

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
        self.train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
        self.test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
        tf.initialize_all_variables().run()

    def pefrom_train_step(self):

        l_r = 1e-3

        def feed_dict():

            xs = self.batch_train_data

            ys = self.batch_labels
            k = self.keep_prob_val
            return {self.x: xs, self.y_: ys, self.keep_prob:k, self.learning_rate: l_r}

        summary, _, mse_val = self.sess.run([self.merged, self.train_step, self.mean_squared_error],feed_dict=feed_dict())

        if self.step%10 == 0:
            self.train_writer.add_summary(summary, self.step )

        self.step += 1

        if self.step % 500 == 0:
            print('result after minibatch no. {} : {}'.format(self.step, mse_val))
            print('batch train data', self.batch_train_data)
            print('batch train labels', self.batch_labels)
            self.plot_learned_function()


    def eval_trained_function(self, input):
        return self.y.eval(feed_dict= {self.x : input,  self.keep_prob: 1.0})

    def main(self):
        if tf.gfile.Exists(FLAGS.summaries_dir):
            tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
        tf.gfile.MakeDirs(FLAGS.summaries_dir)

        self.initialize_training(self.sess)