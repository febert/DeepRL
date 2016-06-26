from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', './regressiontest/regression_batchnormalized', 'Summaries directory')

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

        self.target_func_range_max = (3,30)
        self.target_func_range_min = (0,0)

        self.num_inputs = 2
        self.num_outputs = 1
        self.batchsize = 50

        self.sess = tf.InteractiveSession()
        self.num_neurons_layer1 = 100
        self.num_neurons_layer2 = 100

        self.summaries_dir = './regressiontest/regression_batchnormalized'

    def target_function(self, state, addnoise=True):

        mu_out1 = np.array([1, 10])
        mu_out2 = np.array([2, 20])

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
                vals[j,i]= self.target_function((x,v), addnoise= False)

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
                eval = self.eval_trained_function(x_)[0]
                # print(eval)
                vals[j,i]= eval[0]


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

        self.phase_train = tf.placeholder(tf.bool, name='phase_train')

        def batch_norm(x,shape_x, scope):
            """
            Batch normalization on convolutional maps.
            Args:
                x:           Tensor, 4D BHWD input maps
                n_out:       integer, depth of input maps
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
                mean_squared_error = tf.reduce_mean(squ_diff)
            tf.scalar_summary('mean squared error', mean_squared_error)


        learning_rate = tf.placeholder(tf.float32, shape= [], name='learning_rate')
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(mean_squared_error)


        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
        tf.initialize_all_variables().run()

        # Train the model, and also write summaries.
        # Every 10th step, measure test-set accuracy, and write test summaries
        # All other steps, run train_step on training data, & add training summaries

        def feed_dict(if_train):
            """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
            if if_train:
                xs, ys = self.getbatch(if_train)
                select_train = True
            else:
                xs, ys = self.getbatch(if_train)
                select_train = False

            return {self.x: xs, self.y_: ys, self.phase_train: select_train}

        num_epochs = 3
        steps_per_epoch = 500
        for epoch in range(num_epochs):

            l_r = 10**np.linspace(-2,-4,num_epochs)[epoch]
            print("epoch %d, learning rate %g"%(epoch,l_r))

            for step in range(steps_per_epoch):

                if step % 100 == 0:  # Record summaries and test-set accuracy
                    tmpdict = feed_dict(if_train= False)
                    tmpdict[learning_rate] = l_r

                    summary, acc ,retrive_xbn, r_mean, r_var, r_beta, r_gamma = sess.run( [merged, mean_squared_error, x_bn,mean, var, beta, gamma ], feed_dict=tmpdict)
                    test_writer.add_summary(summary, step + epoch*steps_per_epoch)
                    print('mean_squared_error at step %s: %s' % (step + epoch*steps_per_epoch, acc))

                    print('input data:',tmpdict[self.x])
                    print('mean', r_mean)
                    print('var', r_var)
                    print('beta', r_beta)
                    print('gamma', r_gamma)

                    print('retrieve xbn:', retrive_xbn)

                else:  # Record train set summaries, and train
                    tmpdict = feed_dict(if_train= True)
                    tmpdict[learning_rate] = l_r
                    summary, _ , r_y, retrive_xbn, r_mean, r_var, r_beta, r_gamma = sess.run([merged, train_step, self.y, x_bn,mean, var, beta, gamma],feed_dict=tmpdict)
                    train_writer.add_summary(summary, step + epoch*steps_per_epoch)

                    # print('input data train:',tmpdict[self.x])
                    # print('output data',  r_y)
                    # print('mean', r_mean)
                    # print('var', r_var)
                    # print('beta', r_beta)
                    # print('gamma', r_gamma)
                    #
                    # print('retrieve xbn:', retrive_xbn)

                if (step + epoch*steps_per_epoch) %500 == 0:
                    self.plot_learned_function()
                    self.plot_target_function()

    def eval_trained_function(self, input):
        return self.y.eval(feed_dict= {self.x : input ,self.phase_train: False})

    def main(self):
        if tf.gfile.Exists(self.summaries_dir):
            tf.gfile.DeleteRecursively(self.summaries_dir)
        tf.gfile.MakeDirs(self.summaries_dir)

        self.train(self.sess)

if __name__ == '__main__':

    t1 = regression_test()
    t1.main()
    t1.plot_learned_function()
    t1.plot_target_function()

