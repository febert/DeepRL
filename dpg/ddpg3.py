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

from matplotlib.colors import LogNorm


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
        self.l1 = 400  #neurons layer 1
        self.l2 = 300  #neurons layer 2

        self.step = 0

        self.summaries_dir = './logging/ddpg'

        replay_memory_size = 5e5 #number of transitions to be stored in replay buffer
        self.warmup = 5e4

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

    def run_episode(self, enable_render=False, limit=5000, test_run = True):

        state = self.env.reset()
        self.ou_process.reset()

        count = 0
        done = False

        while ( not done ):

            if count>limit:
                return count

            count += 1
            state = np.squeeze(state).reshape((1,2))  # convert (2,1) array in to (2,1)
            if test_run:
                action = self.eval_mu(state)
            else:
                action = self.eval_mu(state) + self.ou_process.ou_step()

            action = self.apply_limits(action)
            action = action.squeeze()
            state_prime, reward, done, _ = self.env.step(action)

            if not test_run:
                self.replay_memory.append((state, action, reward, state_prime, done))
                self.samples_count += 1

                if (len(self.replay_memory) > self.warmup) and (self.samples_count % (self.batch_size/2) == 0):
                    self.train_networks()

            state = state_prime

            if enable_render:
                self.env.render()

        return count


    def start_training(self, max_episodes=1000, dataname ='unnamed_data', save = False, max_episode_length = 10000):

        for it in range(max_episodes):

            print('starting episode no',it )
            print('replay size',self.samples_count)

            # run episode
            episode_length = self.run_episode(test_run= False, enable_render=False, limit= max_episode_length)
            self.train_lengths.append(episode_length)

            # if it% 5== 0:
            #     self.plot_learned_mu()
            #     self.plot_replay_memory_2d_state_histogramm()

            if (it+1) % 5 == 0:
                # perform a test run with the target policy:
                self.test_lengths.append(self.run_episode(test_run= True, enable_render=False))

            if (it+1)%10 == 0:
                self.plot_episode_lengths(train= True)
                self.plot_episode_lengths(train= False)

    def hist_summaries(self,*args):
        return tf.merge_summary([tf.histogram_summary(t.name, t) for t in args])

    def fanin_init(self, shape, fanin=None):
        fanin = fanin or shape[0]
        v = 1 / np.sqrt(fanin)
        return tf.random_uniform(shape, minval=-v, maxval=v)

    def create_theta_p(self,dimO, dimA):
        with tf.variable_scope("theta_p"):
            return [tf.Variable(self.fanin_init([dimO, self.l1]), name='1w'),
                    tf.Variable(self.fanin_init([self.l1], dimO), name='1b'),
                    tf.Variable(self.fanin_init([self.l1, self.l2]), name='2w'),
                    tf.Variable(self.fanin_init([self.l2], self.l1), name='2b'),
                    tf.Variable(tf.random_uniform([self.l2, dimA], -3e-3, 3e-3), name='3w'),
                    tf.Variable(tf.random_uniform([dimA], -3e-3, 3e-3), name='3b')]

    def mu_net(self, obs, theta, name='policy'):
        with tf.variable_op_scope([obs], name, name):
            h0 = tf.identity(obs, name='h0-obs')
            h1 = tf.nn.relu(tf.matmul(h0, theta[0]) + theta[1], name='h1')
            h2 = tf.nn.relu(tf.matmul(h1, theta[2]) + theta[3], name='h2')
            h3 = tf.identity(tf.matmul(h2, theta[4]) + theta[5], name='h3')
            action = tf.nn.tanh(h3, name='h4-action')

            action_add =  tf.add(action,tf.constant(1.0, dtype=tf.float32, shape= [1]))

            summary = self.hist_summaries(h0, h1, h2, h3, action_add)
            return action_add, summary

    def create_theta_q(self,dimO, dimA):
        with tf.variable_scope("theta_q"):
            return [tf.Variable(self.fanin_init([dimO, self.l1]), name='1w'),
                    tf.Variable(self.fanin_init([self.l1], dimO), name='1b'),
                    tf.Variable(self.fanin_init([self.l1 + dimA, self.l2]), name='2w'),
                    tf.Variable(self.fanin_init([self.l2], self.l1 + dimA), name='2b'),
                    tf.Variable(tf.random_uniform([self.l2, 1], -3e-4, 3e-4), name='3w'),
                    tf.Variable(tf.random_uniform([1], -3e-4, 3e-4), name='3b')]

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

    def exponential_moving_averages(self, theta, tau=0.001):
        ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
        update = ema.apply(theta)  # also creates shadow vars
        averages = [ema.average(x) for x in theta]
        return averages, update

    def initialize_training(self, sess):

        self.theta_mu = self.create_theta_p(self.state_dim, self.action_dim)
        self.theta_q = self.create_theta_q(self.state_dim, self.action_dim)
        self.theta_mu_prime, update_mu_averages = self.exponential_moving_averages(self.theta_mu, 0.001)
        self.theta_q_prime, update_q_averages = self.exponential_moving_averages(self.theta_q, 0.001)

        self.state = tf.placeholder(tf.float32, [None, self.state_dim], name='x-states')

        mean = tf.constant([-0.3, 0], name='batch_mean')
        variance = tf.constant([0.81, 0.0049], name='batch_variance')
        self.state_normed = tf.nn.batch_normalization(self.state, mean, variance, None, None, 1e-8)

        self.mu, sum_p = self.mu_net(self.state_normed, self.theta_mu)

        pl2 = .0
        ql2 = .0 # .01  #set weight decay
        self.learning_rate_actor = tf.placeholder(tf.float32, shape= [], name='actor_learning_rate')
        self.learning_rate_critic = tf.placeholder(tf.float32, shape= [], name='critic_learning_rate')


        # test
        q, sum_q = self.q_net(self.state_normed, self.mu, self.theta_q)
        # training
        # policy loss
        meanq = tf.reduce_mean(q, 0)
        wd_p = tf.add_n([pl2 * tf.nn.l2_loss(var) for var in self.theta_mu])  # weight decay
        self.mu_loss = -meanq + wd_p
        # policy optimization
        optim_p = tf.train.AdamOptimizer(learning_rate=self.learning_rate_actor)
        grads_and_vars_p = optim_p.compute_gradients(self.mu_loss, var_list=self.theta_mu)
        optimize_p = optim_p.apply_gradients(grads_and_vars_p)
        with tf.control_dependencies([optimize_p]):
            self.mu_train_step = tf.group(update_mu_averages)

        # q optimization
        self.act_train = tf.placeholder(tf.float32, [None, self.action_dim], name='actions')
        self.rew = tf.placeholder(tf.float32, [None, 1], name='rewards')
        self.state_prime = tf.placeholder(tf.float32, [None, self.state_dim], name='states_prime')
        state_prime_normed = tf.nn.batch_normalization(self.state_prime, mean, variance, None, None, 1e-8)
        self.term2 = tf.placeholder(tf.bool, [None, 1], name='term2')

        # q
        q_train, sum_qq = self.q_net(self.state_normed, self.act_train, self.theta_q)

        # q targets
        act2, sum_p2 = self.mu_net(state_prime_normed, theta=self.theta_mu_prime)
        self.q2, sum_q2 = self.q_net(state_prime_normed, act2, theta=self.theta_q_prime)
        q_target = tf.stop_gradient(tf.select(self.term2, self.rew, self.rew + self.gamma * tf.reshape(self.q2, [self.batch_size,1]) ))

        # q loss
        td_error = q_train - q_target
        ms_td_error = tf.reduce_mean(tf.square(td_error), 0)
        wd_q = tf.add_n([ql2 * tf.nn.l2_loss(var) for var in self.theta_q])  # weight decay
        self.q_loss = ms_td_error + wd_q

        # q optimization
        optim_q = tf.train.AdamOptimizer(learning_rate=self.learning_rate_critic)
        grads_and_vars_q = optim_q.compute_gradients(self.q_loss, var_list=self.theta_q)
        optimize_q = optim_q.apply_gradients(grads_and_vars_q)
        with tf.control_dependencies([optimize_q]):
            self.q_train_step = tf.group(update_q_averages)


        # logging
        log_obs = [] if self.state_dim > 20 else [tf.histogram_summary("obs/" + str(i), self.state[:,i]) for i in range(self.state_dim)]
        log_act = [] if self.action_dim > 20 else [tf.histogram_summary("act/inf" + str(i), self.mu[:, i]) for i in
                                           range(self.action_dim)]
        log_act2 = [] if self.action_dim > 20 else [tf.histogram_summary("act/train" + str(i), self.act_train[:, i]) for i in
                                            range(self.action_dim)]
        log_misc = [sum_p, sum_qq, tf.histogram_summary("td_error", td_error)]
        log_grad = [self.grad_histograms(grads_and_vars_p), self.grad_histograms(grads_and_vars_q)]
        log_train = log_obs + log_act + log_act2 + log_misc + log_grad


        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        self.merged = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(self.summaries_dir, sess.graph)
        tf.initialize_all_variables().run()

    def get_train_batch(self):

        #selecting transitions randomly from the replay memory:
        indices =  np.random.randint(0, len(self.replay_memory), [self.batch_size])
        transition_batch = [self.replay_memory[i] for i in indices]

        states = np.asarray([transition_batch[i][0].squeeze() for i in range(self.batch_size)])
        actions = np.asarray([transition_batch[i][1] for i in range(self.batch_size)])
        rewards = np.asarray([transition_batch[i][2] for i in range(self.batch_size)])
        states_prime = np.asarray([transition_batch[i][3].squeeze() for i in range(self.batch_size)])
        term2 = np.asarray([transition_batch[i][4] for i in range(self.batch_size)])

        return states, actions, rewards, states_prime, term2

    def train_networks(self):

        lr_actor = 1e-5
        lr_critic = 1e-4   #according to Silver dpg-paper the critic should be faster !!

        def feed_dict():

            states, action, reward, states_prime, term2 = self.get_train_batch()

            return {self.state: states.squeeze().reshape(self.batch_size,2),
                    self.act_train: action.squeeze().reshape(self.batch_size,1),
                    self.rew: reward.squeeze().reshape(self.batch_size,1),
                    self.state_prime: states_prime.squeeze().reshape(self.batch_size,2),
                    self.term2: term2.squeeze().reshape(self.batch_size,1),
                    self.learning_rate_actor: lr_actor,
                    self.learning_rate_critic: lr_critic,
                    }


        dict_ = feed_dict()
        summary, _, mse_val = self.sess.run([self.merged, self.q_train_step, self.q_loss], feed_dict= dict_)
        summary, _, _ = self.sess.run([self.merged, self.mu_train_step, self.mu_loss], feed_dict= dict_)

        if self.step % 10 == 0:
            self.train_writer.add_summary(summary, self.step)

        if self.step % 100 == 0:
            print('result after minibatch no. {} : mean squared error: {}'.format(self.step, mse_val))
            # print('batch train data states', dict_[self.x_states])
            # print('batch train data actions', dict_[self.x_action])
            self.plot_learned_mu()
            self.plot_replay_memory_2d_state_histogramm()

            # print('qs: ', self.q.eval(feed_dict = {self.x_states: dict_[self.x_states], self.x_action: dict_[self.x_action]}))

            # print('batch train data targets', dict_[self.q_targets])

        self.step += 1

    def eval_mu(self, state):
        return self.mu.eval(feed_dict= {self.state : state})


    def apply_limits(self,action):

        if action < self.action_limits[0]:
            action = self.action_limits[0]

        if action > self.action_limits[1]:
            action = self.action_limits[1]

        return action


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


    def plot_replay_memory_2d_state_histogramm(self):
        if self.state_dim == 2:
            rm=np.array(self.replay_memory)
            states, _,_,_,_ = zip(*rm)
            states_np = np.array(states)
            states_np = np.squeeze(states_np)

            x,v = zip(*states_np)
            plt.hist2d(x, v, bins=40, norm=LogNorm())
            plt.xlabel("position")
            plt.ylabel("velocity")
            plt.colorbar()
            plt.show()

    # def plot_q_func(self):
    #     if self.state_dim == 2:
    #         rm=np.array(self.replay_memory)
    #         states, _,_,_,_ = zip(*rm)
    #         states_np = np.array(states)
    #         states_np = np.squeeze(states_np)
    #
    #         x,v = zip(*states_np)
    #         plt.hist2d(x, v, bins=40, norm=LogNorm())
    #         plt.xlabel("position")
    #         plt.ylabel("velocity")
    #         plt.colorbar()
    #         plt.show()

    def plot_learned_mu(self):

        print('plotting the mu() policy learned by NN')

        resolution = 20
        x_range = np.linspace(self.obs_low[0], self.obs_high[0], resolution)
        v_range = np.linspace(self.obs_low[1], self.obs_high[1], resolution)

        # get actions in a grid
        vals = np.zeros((resolution, resolution))
        for i, x in enumerate(x_range):
            for j, v in enumerate(v_range):
                x_ = np.array([x,v],dtype=np.float32).reshape((1,2))
                vals[j,i]= self.eval_mu(x_)[0]

        print('muvals', vals)

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_range, v_range)
        ax.plot_surface(X, Y, vals, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=True)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_zlabel("action")
        plt.show()

    def grad_histograms(self,grads_and_vars):
        s = []
        for grad, var in grads_and_vars:
            s.append(tf.histogram_summary(var.op.name + '', var))
            s.append(tf.histogram_summary(var.op.name + '/gradients', grad))
        return tf.merge_summary(s)

if __name__ == '__main__':

    car = ddpg()
    car.main()
