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


class ddpg():

    def __init__(self,
                 input_low,
                 input_high,
                 environment = 'MountainCarContinuous-v0',
                 ):

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
        self.batch_train_data = np.zeros((batchsize, 2))
        self.batch_labels = np.zeros(batchsize).reshape((batchsize, 1))
        self.batchindex = 0

        self.step = 0

        self.summaries_dir = './regressiontest/batchnorm'

        self.max_replay_size = 1e3 #number of minibatch transition to be stored in replay buffer


        self.env = gym.make(environment)
        self.select_env = environment

        self.action_limits = (self.env.action_space.low, self.env.action_space.high)
        print('action limits', self.action_limits)
        actionmean = (self.action_limits[0]+ self.action_limits[1])/2

        self.episode_lengths = []
        self.test_lengths = []

    def add_to_replay(self, state, mu):

        self.batch_train_data[self.batchindex, :] = state
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

    def run_episode(self, it, enable_render=False, limit=5000, target = True):

        episode = []
        state = self.env.reset()

        count = 0
        done = False

        while ( not done ):

            if len(episode)>limit:
                return episode

            count += 1
            state = np.squeeze(state)  # convert (2,1) array in to (2,)
            if target:
                action = self.mu(state)
            else:
                action = self.beta(state)

            # action = self.apply_limits(action)
            # print('action',action)

            state_prime, reward, done, info = self.env.step(action)

            self.add_to_replay(state, action, reward, state_prime)
            state = state_prime

            if it > 0 or count > 100*self.batchsize:
                self.train_networks()

            if enable_render:
                self.env.render()

        return count


    def start_training(self, max_episodes=1000, dataname ='unnamed_data', save = False, max_episode_length = 20000):
        # fig = plt.figure()

        for it in range(max_episodes):

            # run episode
            episode = self.run_episode(it, target= False, enable_render=False, limit= max_episode_length,)

            self.episode_lengths.append(len(episode))

            if it % 5 == 0:
                # perform a test run with the target policy:
                self.test_lengths.append(len(self.run_episode(target= True, enable_render=False)))


            if (it+1)%1 == 0:
                # output training info
                print("Finished run #{}".format(it + 1))
                print("with a sigma exploration of  {}".format(self.sigma_b))
#                print("and learning rate of {}".format(self.alpha))
                print("lasted {0} steps".format(len(episode)))

                if self.select_env == 'MountainCarContinuous-v0':
                    print("theta")
                    print(self.theta)
                    print('last v', self.v)
                    print("beta: ")
                    print("mu: ")
                    self.plot_policy(mode= 'deterministic')


            # print('sum tile features ', tile_features_mat[idx].sum())
            print('max theta', self.theta.max())
            print('min theta', self.theta.min())


            if (it+1)%10 == 0:
                self.plot_training()
                self.plot_testing()

            if save:
                self.savedata(dataname=dataname)

        return self.theta

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

        class nn_layer():

            def __init__(self,input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):

                with tf.name_scope(layer_name):
                    # This Variable will hold the state of the weights for the layer
                    with tf.name_scope('weights'):
                        self.weights = weight_variable([input_dim, output_dim])
                        variable_summaries(self.weights, layer_name + '/weights')
                    with tf.name_scope('biases'):
                        self.biases = bias_variable([output_dim])
                        variable_summaries(self.biases, layer_name + '/biases')
                    with tf.name_scope('Wx_plus_b'):
                        preactivate = tf.matmul(input_tensor, self.weights) + self.biases
                        tf.histogram_summary(layer_name + '/pre_activations', preactivate)
                    self.activations = act(preactivate, 'activation')
                    tf.histogram_summary(layer_name + '/activations', self.activations)

                    self.ema = tf.train.ExponentialMovingAverage(decay=0.9)
                    self.ema = self.ema.apply([self.weights,self.biases])

            def make_nn_layer(self):
                return self.activations

        class nn_layer_prime():

            def __init__(self, input_tensor, nn_, input_dim, output_dim, layer_name, act=tf.nn.relu):

                with tf.name_scope(layer_name):
                    # This Variable will hold the state of the weights for the layer
                    with tf.name_scope('weights'):
                        self.weights = nn_.ema.average(nn_.weights)
                        variable_summaries(self.weights, layer_name + '/weights')
                    with tf.name_scope('biases'):
                        self.biases = nn_.ema.average(nn_.biases)
                        variable_summaries(self.biases, layer_name + '/biases')
                    with tf.name_scope('Wx_plus_b'):
                        preactivate = tf.matmul(input_tensor, self.weights) + self.biases
                        tf.histogram_summary(layer_name + '/pre_activations', preactivate)
                    self.activations = act(preactivate, 'activation')
                    tf.histogram_summary(layer_name + '/activations', self.activations)

            def make_nn_layer(self):
                return self.activations

        with tf.name_scope('mu_net'):

            with tf.name_scope('hidden_1'):
                n1 = nn_layer(self.x, self.num_inputs, self.num_neurons_layer1, 'layer1')
                a1 = n1.make_nn_layer()

            with tf.name_scope('hidden_2'):
                n2 = nn_layer(a1, self.num_neurons_layer1, self.num_neurons_layer2, 'layer2')
                a2 = n2.make_nn_layer()

            self.y_mu = nn_layer(a2, self.num_neurons_layer2, self.num_outputs, 'layer3', act=tf.identity)

        with tf.name_scope('mu_net_prime'):

            with tf.name_scope('hidden_1'):
                n1_prime = nn_layer_prime(self.x, n1,self.num_inputs, self.num_neurons_layer1, 'layer1')
                a1_prime = n1_prime.make_nn_layer()

            with tf.name_scope('hidden_2'):
                n2_prime = nn_layer_prime(a1_prime, n2, self.num_neurons_layer1, self.num_neurons_layer2, 'layer2')
                a2_primnn_layer = n2.make_nn_layer()

            self.y_mu_prime = nn_layer(a2_primnn_layer, self.num_neurons_layer2, self.num_outputs, 'layer3', act=tf.identity)



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

    def get_train_batch(self):

        #getting random transition from the replay memory:
        indices = np.random.choice(self.replay_size)

        for i in range(self.batchsize):



    def train_networks(self):

        l_r = 1e-3

        def feed_dict():

            xs, ys = self.get_train_batch()

            ys = self.batch_labels

            return {self.x: xs, self.y_: ys, self.learning_rate: l_r}

        summary, _, mse_val = self.sess.run([self.merged, self.train_step, self.mean_squared_error],feed_dict=feed_dict())

        if self.step % 10 == 0:
            self.train_writer.add_summary(summary, self.step)

        self.step += 1

        if self.step % 500 == 0:
            print('result after minibatch no. {} : mean squared error: {}'.format(self.step, mse_val))
            print('batch train data', self.batch_train_data)
            print('batch train labels', self.batch_labels)
            self.plot_learned_function()

            num_steps= self.test_learned_policy()
            print('episode length using learned policy:',num_steps)

    def mu(self, input):
        return self.y.eval(feed_dict= {self.x : input ,self.phase_train: False})

    def q(self, input):
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
