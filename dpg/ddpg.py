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

from collections import deque
from ornstein_uhlenbeck import ornstein_uhlenbeck


class ddpg():

    def __init__(self,
                 environment = 'MountainCarContinuous-v0',
                 ):

        self.gamma = 0.9

        self.sigma = np.identity(2)*0.5


        self.num_outputs = 1
        self.action_dim = 1
        self.state_dim = 2
        self.batch_size = 50
        self.samples_count = 0

        self.sess = tf.InteractiveSession()
        self.num_neurons_layer1 = 100
        self.num_neurons_layer2 = 100


        self.step = 0

        self.summaries_dir = './regressiontest/ddpg'

        replay_memory_size = 1e6 #number of transitions to be stored in replay buffer

        self.train_lengths = []
        self.test_lengths = []

        self.replay_memory = deque(maxlen=replay_memory_size)

        # environment specific:
        self.env = gym.make(environment)
        self.select_env = environment

        self.action_limits = (self.env.action_space.low, self.env.action_space.high)
        print('action limits', self.action_limits)
        actionmean = (self.action_limits[0]+ self.action_limits[1])/2

        self.ou_process = ornstein_uhlenbeck(ndim= 1, theta= 0.15, sigma= .3, delta_t= 1)

        self.obs_low = self.env.observation_space.low
        self.obs_high = self.env.observation_space.high


    def plot_learned_mu(self):

        print('plotting the mu() policy learned by NN')

        resolution = 50
        x_range = np.linspace(self.obs_low[0], self.obs_high[0], resolution)
        v_range = np.linspace(self.obs_low[1], self.obs_high[1], resolution)

        # get actions in a grid
        vals = np.zeros((resolution, resolution))
        for i, x in enumerate(x_range):
            for j, v in enumerate(v_range):
                x_ = np.array([x,v],dtype=np.float32).reshape((1,2))
                vals[j,i]= self.mu(x_)[0]

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_range, v_range)
        ax.plot_surface(X, Y, vals, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=True)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_zlabel("action")
        plt.show()

    def run_episode(self, enable_render=False, limit=5000, test_run = True):

        state = self.env.reset()
        self.ou_process.reset()

        count = 0
        done = False

        while ( not done ):

            if count>limit:
                return count

            count += 1
            state = np.squeeze(state).reshape((1,2))  # convert (2,1) array in to (2,)
            if test_run:
                action = self.mu(state)
            else:
                action = self.mu(state) + self.ou_process.ou_step()

            state_prime, reward, done, info = self.env.step(action)

            state = state_prime

            if not test_run:
                self.replay_memory.append((state, action, reward, state_prime, done))
                self.samples_count += 1
                if len(self.replay_memory) > 1e2* self.batch_size and self.samples_count % self.batch_size/2:
                    self.train_networks()

            if enable_render:
                self.env.render()

        return count


    def start_training(self, max_episodes=1000, dataname ='unnamed_data', save = False, max_episode_length = 10000):

        for it in range(max_episodes):

            print('starting episode no',it )

            # run episode
            episode_length = self.run_episode(test_run= False, enable_render=False, limit= max_episode_length)
            self.train_lengths.append(episode_length)

            if (it+1) % 100 == 0:
                # perform a test run with the target policy:
                self.test_lengths.append(self.run_episode(test_run= True, enable_render=False))

            if (it+1)%100 == 0:
                self.plot_episode_lengths(train= True)
                self.plot_episode_lengths(train= False)

    def initialize_training(self, sess):

        # Create a multilayer model.

        # Input placehoolders
        with tf.name_scope('input'):
            self.x_state = tf.placeholder(tf.float32, [None, self.state_dim], name='x-input')
            self.x_action = tf.placeholder(tf.float32, [None, self.action_dim], name='x-input')
            self.y_qtargets = tf.placeholder(tf.float32, [None, self.num_outputs], name='y-input')

        # We can't initialize these variables to 0 - the network will get stuck.
        def weight_variable(shape,initrange):
            """Create a weight variable with appropriate initialization."""
            #initial = tf.truncated_normal(shape, stddev=0.1)
            initial = tf.random_uniform(shape,initrange[0],initrange[1])
            return tf.Variable(initial)

        def bias_variable(shape,initrange, init_bias_rnd):
            """Create a bias variable with appropriate initialization."""
            if init_bias_rnd:
                initial = tf.random_uniform(shape,initrange[0],initrange[1])
            else:
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

            def __init__(self,input_tensor, input_dim, output_dim, init_range, layer_name, act=tf.nn.relu, init_bias_rnd= False):

                with tf.name_scope(layer_name):
                    # This Variable will hold the state of the weights for the layer
                    with tf.name_scope('weights'):
                        self.weights = weight_variable([input_dim, output_dim],init_range)
                        variable_summaries(self.weights, layer_name + '/weights')
                    with tf.name_scope('biases'):
                        self.biases = bias_variable([output_dim],init_range, init_bias_rnd=False,  )
                        variable_summaries(self.biases, layer_name + '/biases')
                    with tf.name_scope('Wx_plus_b'):
                        preactivate = tf.matmul(input_tensor, self.weights) + self.biases
                        tf.histogram_summary(layer_name + '/pre_activations', preactivate)
                    self.activations = act(preactivate, 'activation')
                    tf.histogram_summary(layer_name + '/activations', self.activations)

                    self.ema = tf.train.ExponentialMovingAverage(decay=0.9)
                    self.ema_ap_op = self.ema.apply([self.weights,self.biases])

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

        mean = tf.constant([-0.3, 0], name='batch_mean')
        variance = tf.constant([0.81, 0.0049], name='batch_variance')
        x_state_normed = tf.nn.batch_normalization(self.x_state, mean, variance, None, None, 1e-8)

        with tf.name_scope('mu_net'):

            with tf.name_scope('hidden_1'):
                initrange = (-1/np.sqrt(self.state_dim), 1/np.sqrt(self.state_dim))
                n1 = nn_layer(x_state_normed, self.state_dim, self.num_neurons_layer1, initrange, 'layer1')
                a1 = n1.make_nn_layer()

            with tf.name_scope('hidden_2'):
                initrange = (-1/np.sqrt(self.num_neurons_layer1), 1/np.sqrt(self.num_neurons_layer1))
                n2 = nn_layer(a1, self.num_neurons_layer1, self.num_neurons_layer2,initrange, 'layer2')
                a2 = n2.make_nn_layer()

            initrange = (-1e-4, 1e-4)
            n3 = nn_layer(a2, self.num_neurons_layer2, self.num_outputs, initrange, layer_name= 'layer3', act=tf.nn.tanh, init_bias_rnd= True)
            n3_out = n3.make_nn_layer()
            self.y_mu =  tf.add(n3_out,tf.constant(1.0, dtype=tf.float32, shape= [1]))


        with tf.name_scope('mu_net_prime'):

            with tf.name_scope('hidden_1'):
                n1_prime = nn_layer_prime(x_state_normed, n1, self.state_dim, self.num_neurons_layer1, 'layer1')
                a1_prime = n1_prime.make_nn_layer()

            with tf.name_scope('hidden_2'):
                n2_prime = nn_layer_prime(a1_prime, n2, self.num_neurons_layer1, self.num_neurons_layer2, 'layer2')
                a2_prime = n2_prime.make_nn_layer()

            n3_prime= nn_layer_prime(a2_prime,n3, self.num_neurons_layer2, self.num_outputs, 'layer3', act=tf.nn.tanh)
            n3_out_prime = n3_prime.make_nn_layer()
            self.y_mu_prime = tf.add(n3_out_prime,tf.constant(1.0, dtype=tf.float32, shape= [1]))

        with tf.name_scope('Q_net'):

            with tf.name_scope('hidden_1'):
                initrange = (-1/np.sqrt(self.state_dim), 1/np.sqrt(self.state_dim))
                n1q = nn_layer(x_state_normed, self.state_dim, self.num_neurons_layer1,initrange, 'layer1')
                a1q = n1q.make_nn_layer()

            with tf.name_scope('hidden_2'):
                initrange = (-1/np.sqrt(self.num_neurons_layer1), 1/np.sqrt(self.num_neurons_layer1))
                conc = tf.concat(concat_dim=1, values=[a1q, self.x_action] , name='concat')
                n2q = nn_layer(conc, self.num_neurons_layer1+ self.action_dim, self.num_neurons_layer2, initrange, 'layer2')
                a2q = n2q.make_nn_layer()

            initrange = (-1e-3, 1e-3)
            n3q = nn_layer(a2q, self.num_neurons_layer2, self.num_outputs, initrange, layer_name= 'layer3', act=tf.identity)
            y_q = n3q.make_nn_layer()

        with tf.name_scope('Q_net_prime'):

            with tf.name_scope('hidden_1'):
                n1_primeq = nn_layer_prime(x_state_normed, n1q, self.state_dim + self.action_dim, self.num_neurons_layer1, 'layer1')
                a1_primeq = n1_primeq.make_nn_layer()

            with tf.name_scope('hidden_2'):
                conc = tf.concat(concat_dim=1, values=[a1_primeq, self.x_action] , name='concat')
                n2_primeq = nn_layer_prime(conc, n2q, self.num_neurons_layer1, self.num_neurons_layer2, 'layer2')
                a2_primeq = n2_primeq.make_nn_layer()

            n3_primeq= nn_layer_prime(a2_primeq,n3, self.num_neurons_layer2, self.num_outputs, 'layer3', act=tf.identity)
            self.y_q_prime = n3_primeq.make_nn_layer()


        self.learning_rate_actor = tf.placeholder(tf.float32, shape= [], name='learning_rate')
        self.learning_rate_critic = tf.placeholder(tf.float32, shape= [], name='learning_rate')

        ## set up training the Q-Function
        ql2 = .01

        theta_q = [n1q.weights,n1q.biases,n2q.weights,n2q.biases,n3q.weights,n3q.biases]
        squ_diff = tf.pow(self.y_qtargets - y_q, 2)
        wd_q = tf.add_n([ql2 * tf.nn.l2_loss(var) for var in theta_q])  # weight decay
        self.q_loss = tf.reduce_mean(squ_diff) + wd_q
        tf.scalar_summary('Qfunc mean squared error', self.q_loss)
        self.qtrain_step = tf.train.AdamOptimizer(self.learning_rate_critic).minimize(self.q_loss)

        ## set up training the mu-Function
        opt = tf.train.AdamOptimizer(self.learning_rate_actor)
        theta_mu = [n1.weights,n1.biases,n2.weights,n2.biases,n3.weights,n3.biases]   #these are all variables from the Q-network
        mu_loss = -tf.reduce_mean(y_q, 0)
        tf.scalar_summary('Qfunc mean squared error', self.q_loss)
        grads_vars_p = opt.compute_gradients(mu_loss,theta_mu)

        ema_updates = [n1.ema_ap_op, n2.ema_ap_op, n3.ema_ap_op, n1q.ema_ap_op, n2q.ema_ap_op, n3q.ema_ap_op]
        with tf.control_dependencies(ema_updates+[self.qtrain_step]):  #combining all train steps:
            self.mu_train_step = opt.apply_gradients(grads_vars_p)


        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        self.merged = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(self.summaries_dir, sess.graph)
        tf.initialize_all_variables().run()

    def get_train_batch(self):

        #selecting transitions randomly from the replay memory:
        indices =  np.random.randint(0, len(self.replay_memory), [self.batch_size])
        transition_batch = [self.replay_memory[i] for i in indices]

        states = np.asarray([transition_batch[i][0] for i in range(self.batch_size)])
        actions = np.asarray([transition_batch[i][1] for i in range(self.batch_size)])
        rewards = np.asarray([transition_batch[i][2] for i in range(self.batch_size)])
        states_prime = np.asarray([transition_batch[i][3] for i in range(self.batch_size)])
        term2 = np.asarray([transition_batch[i][4] for i in range(self.batch_size)])

        Qprime = self.eval_Qnet_prime(states_prime.squeeze(), self.mu_prime(states_prime.squeeze()))
        Qprime = np.asarray(Qprime).squeeze()

        term2 = 1- term2 #inverting, now 0 means terminated

        targets = rewards + self.gamma * Qprime * term2

        return states, actions, targets

    def train_networks(self):

        lr_actor = 1e-3
        lr_critic = 1e-4   #according to Silver dpg-paper the critic should be faster !!

        def feed_dict():

            xs_state, xs_action, y_qtargets = self.get_train_batch()

            return {self.x_state: xs_state.squeeze(),
                    self.x_action: xs_action.squeeze().reshape(self.batch_size,1),
                    self.y_qtargets: y_qtargets.squeeze().reshape(self.batch_size,1),
                    self.learning_rate_actor: lr_actor,
                    self.learning_rate_critic: lr_critic,
                    }

        dict_ = feed_dict()
        summary, _, mse_val = self.sess.run([self.merged, self.mu_train_step, self.q_loss], feed_dict= dict_)

        if self.step % 10 == 0:
            self.train_writer.add_summary(summary, self.step)

        self.step += 1

        if self.step % 100 == 0:
            print('result after minibatch no. {} : mean squared error: {}'.format(self.step, mse_val))
            print('batch train data states', dict_[self.x_state])
            print('batch train data actions', dict_[self.x_action])
            print('batch train data actions', dict_[self.y_qtargets])
            self.plot_learned_mu()

            num_steps= self.run_episode(test_run=True)
            print('episode length using learned policy:',num_steps)

    def mu(self, state):
        return self.y_mu.eval(feed_dict= {self.x_state : state})

    def mu_prime(self, state):
        return self.y_mu_prime.eval(feed_dict= {self.x_state : state})

    def eval_Qnet_prime(self, state, action):
        return self.y_q_prime.eval(feed_dict= {self.x_state : state, self.x_action: action})


    def plot_episode_lengths(self, train):

        fig = plt.figure()
        if train:
            plt.plot(self.train_lengths)
        else:
            plt.plot(self.test_lengths)

        plt.yscale('log')

        plt.xlabel("episodes")
        plt.ylabel("timesteps")

        plt.show()

    def main(self):
        if tf.gfile.Exists(self.summaries_dir):
            tf.gfile.DeleteRecursively(self.summaries_dir)
        tf.gfile.MakeDirs(self.summaries_dir)

        self.initialize_training(self.sess)
        self.start_training()

if __name__ == '__main__':

    car = ddpg()
    car.main()
