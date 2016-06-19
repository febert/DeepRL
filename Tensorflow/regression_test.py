from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
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



class regression_test():

    def __init__(self):

        self.target_func_range_max = (3,3)
        self.target_func_range_min = (0,0)

        self.num_inputs = 2
        self.num_outputs = 1
        self.batchsize = 100

        self.sess = tf.InteractiveSession()
        self.num_neurons_layer1 = 100
        self.num_neurons_layer2 = 100


    def target_function(self, state, addnoise=True):

        mu_out1 = np.array([1, 1])
        mu_out2 = np.array([2, 2])

        sig1 =  np.identity(2)*0.1
        sig2 =  np.identity(2)*0.1


        def func1(input, mu, sig):
            delta = input - mu
            return np.exp(-0.5*delta.dot(np.linalg.inv(sig)).dot(delta)) / np.sqrt(2*np.pi*np.linalg.det(sig))



        if addnoise:
            noise = np.random.randn(1) * 0.1
        else:
            noise = 0


        out = func1(input= state, mu= mu_out1, sig= sig1) + func1(input= state, mu= mu_out2, sig= sig2) + noise

        return out

    def getbatch(self,train):

        xs = np.zeros((self.batchsize, self.num_inputs))
        ys = np.zeros((self.batchsize, self.num_outputs))

        for i in range(self.batchsize):
            xs[i] = np.array([ np.random.uniform(self.target_func_range_min[0],self.target_func_range_max[0],1),
                               np.random.uniform(self.target_func_range_min[1],self.target_func_range_max[1],1) ]).squeeze()
            if train:
                ys[i] = self.target_function(xs[i], addnoise= True)  # training is performed on a noisy version of the function
            else:
                ys[i] = self.target_function(xs[i], addnoise= False) # use the clear function when performing testing

        return xs, ys


    def plot_target_function(self):

        resolution = 50

        # values to evaluate policy at
        x_range = np.linspace(self.target_func_range_min[0], self.target_func_range_max[0], resolution)
        v_range = np.linspace(self.target_func_range_min[1], self.target_func_range_max[1], resolution)

        # get actions in a grid
        vals = np.zeros((resolution, resolution))
        for i, x in enumerate(x_range):
            for j, v in enumerate(v_range):
                vals[i,j]= self.target_function((x,v), addnoise= False)

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_range, v_range)
        ax.plot_surface(X, Y, vals, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=True)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_zlabel("action")
        plt.show()

    def plot_learned_function(self):

        resolution = 50

        # values to evaluate policy at
        x_range = np.linspace(self.target_func_range_min[0], self.target_func_range_max[0], resolution)
        v_range = np.linspace(self.target_func_range_min[1], self.target_func_range_max[1], resolution)

        # get actions in a grid
        vals = np.zeros((resolution, resolution))
        for i, x in enumerate(x_range):
            for j, v in enumerate(v_range):
                x_ = np.array([x,v],dtype=np.float32).reshape((1,2))
                vals[i,j]= self.eval_trained_function(x_)[0]

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_range, v_range)
        ax.plot_surface(X, Y, vals, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=True)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_zlabel("action")
        plt.show()


    def train(self, sess):

        # Create a multilayer model.

        # Input placehoolders
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, self.num_inputs], name='x-input')
            self.y_ = tf.placeholder(tf.float32, [None, self.num_outputs], name='y-input')

        # with tf.name_scope('input_reshape'):
        #     image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        #     tf.image_summary('input', image_shaped_input, 10)

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



        self.keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope('hidden_1'):
            a1 = nn_layer(self.x, self.num_inputs, self.num_neurons_layer1, 'layer1')
            with tf.name_scope('dropout'):
                self.keep_prob1 = tf.placeholder(tf.float32)
                tf.scalar_summary('dropout_keep_probability_layer1', self.keep_prob1)
                dropped1 = tf.nn.dropout(a1, self.keep_prob1)

        with tf.name_scope('hidden_2'):
            a2 = nn_layer(dropped1, self.num_neurons_layer1, self.num_neurons_layer2, 'layer2')
            with tf.name_scope('dropout'):
                self.keep_prob2 = tf.placeholder(tf.float32)
                tf.scalar_summary('dropout_keep_probability_layer2', self.keep_prob2)
                dropped2 = tf.nn.dropout(a2, self.keep_prob2)

        self.y = nn_layer(dropped2, self.num_neurons_layer2, self.num_outputs, 'layer3', act=tf.identity)

        with tf.name_scope('mse'):
            squ_diff = tf.pow(self.y_- self.y, 2)
            with tf.name_scope('total'):
                mean_squared_error = tf.reduce_mean(squ_diff)
            tf.scalar_summary('mean squared error', mean_squared_error)


        learning_rate = tf.placeholder(tf.float32, shape= [], name='learning_rate')
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(mean_squared_error)

        # with tf.name_scope('accuracy'):
        #     with tf.name_scope('correct_prediction'):
        #         correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #     with tf.name_scope('accuracy'):
        #         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #     tf.scalar_summary('accuracy', accuracy)

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
        tf.initialize_all_variables().run()

        # Train the model, and also write summaries.
        # Every 10th step, measure test-set accuracy, and write test summaries
        # All other steps, run train_step on training data, & add training summaries

        def feed_dict(train):
            """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
            if train:
                xs, ys = self.getbatch(train)
                k = FLAGS.dropout
            else:
                xs, ys = self.getbatch(train)
                k = 1.0
            return {self.x: xs, self.y_: ys, self.keep_prob1: k, self.keep_prob2: k}

        num_epochs = 10
        steps_per_epoch = 1000
        for epoch in range(num_epochs):
            l_r = 10**np.linspace(-3,-5,num_epochs)[epoch]
            print("epoch %d, learning rate %g"%(epoch,l_r))

            for step in range(steps_per_epoch):

                if step % 10 == 0:  # Record summaries and test-set accuracy
                    tmpdict = feed_dict(train= False)
                    tmpdict[learning_rate] = l_r
                    summary, acc = sess.run([merged, mean_squared_error], feed_dict=tmpdict)
                    test_writer.add_summary(summary, step + epoch*steps_per_epoch)
                    print('mean_squared_error at step %s: %s' % (step + epoch*steps_per_epoch, acc))
                else:  # Record train set summaries, and train
                    if step % 100 == 99:  # Record execution stats
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        tmpdict = feed_dict(train= True)
                        tmpdict[learning_rate] = l_r
                        summary, _ = sess.run([merged, train_step],
                                              feed_dict=tmpdict,
                                              options=run_options,
                                              run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, 'step%d' %  (step + epoch*steps_per_epoch))
                        train_writer.add_summary(summary, step + epoch*steps_per_epoch)
                        print('Adding run metadata for', step + epoch*steps_per_epoch)
                    else:  # Record a summary
                        tmpdict = feed_dict(train= True)
                        tmpdict[learning_rate] = l_r
                        summary, _ = sess.run([merged, train_step],feed_dict=tmpdict)
                        train_writer.add_summary(summary, step + epoch*steps_per_epoch)

                if (step + epoch*steps_per_epoch) %1000 == 0:
                    self.plot_learned_function()
                    self.plot_target_function()

    def eval_trained_function(self, input):
        return self.y.eval(feed_dict= {self.x : input,  self.keep_prob1: 1.0, self.keep_prob2: 1.0})

    def main(self):
        if tf.gfile.Exists(FLAGS.summaries_dir):
            tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
        tf.gfile.MakeDirs(FLAGS.summaries_dir)

        self.train(self.sess)


if __name__ == '__main__':

    t1 = regression_test()
    t1.main()
    t1.plot_target_function()
    t1.plot_learned_function()