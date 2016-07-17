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


import time
import math
import cPickle
import gym as gym

from collections import deque
from ornstein_uhlenbeck import ornstein_uhlenbeck


import naf_net
import filter_env
import plotting

class NAF():

    def __init__(self,
                 environment = 'MountainCarContinuous-v0',
                 # environment = 'InvertedPendulum-v1',
                 ):

        self.gamma = 0.99
        lr = 1e-3 #learning rate

        self.sess = tf.InteractiveSession()
        self.l1 = 100  #neurons layer 1
        self.l2 = 100  #neurons layer 2

        self.step = 0  # number of SGD-steps alredy taken

        self.summaries_dir = './logging/ddpg'

        replay_memory_size = 5e5 #number of transitions to be stored in replay buffer
        self.warmup = 5e4

        self.train_lengths = []
        self.test_lengths = []

        self.replay_memory = deque(maxlen=replay_memory_size)

        # environment specific:
        self.env_f = filter_env.makeFilteredEnv(gym.make(environment))
        self.select_env = environment

        self.num_outputs = 1
        self.action_dim = self.env_f.action_space.shape[0]
        self.state_dim = self.env_f.observation_space.shape[0]

        print('state dim', self.state_dim)
        print('action dim', self.action_dim)

        self.batch_size = 32
        self.samples_count = 0
        self.ou_process = ornstein_uhlenbeck(ndim= 1, theta= 0.15, sigma= .2, delta_t= 1)


        ####### Initialize the Networks: ######

        self.state = tf.placeholder(tf.float32, [None, self.state_dim], name='x-states')
        self.action = tf.placeholder(tf.float32, [None, self.action_dim], name='x-action')

        neurons_layer1 = 200
        neurons_layer2 = 200
        theta_hidden = naf_net.theta_hidden(self.state_dim, neurons_layer1, neurons_layer2)
        self.hidden_out, _ = naf_net.hidden_layers(self.state, theta_hidden, name= 'hidden_net')

        theta_v = naf_net.theta_fc(neurons_layer2, 1)
        self.V, _ = naf_net.fc_layer(self.hidden_out, theta_v, tf.identity, 'v_layer')

        theta_l = naf_net.theta_fc(neurons_layer2, int(self.action_dim*(self.action_dim+1)/2))
        l, _ = naf_net.fc_layer(self.hidden_out, theta_l, tf.identity,'l_layer')

        # theta_mu = naf_net.theta_fc(neurons_layer2, self.action_dim)

        theta_mu = [tf.Variable(tf.random_uniform((neurons_layer2,self.action_dim),-1e-3, 1e-3), name='1w'),
                    tf.Variable(tf.random_uniform([self.action_dim],-1e-3, 1e-3), name='1b')]
        self.mu, _ = naf_net.fc_layer(self.hidden_out, theta_mu, tf.tanh, 'mu_layer')

        #prime net:

        self.state_prime = tf.placeholder(tf.float32, [None, self.state_dim], name='x-states_prime')

        theta_hidden_prime, update_hidden = naf_net.exponential_moving_averages(theta_hidden, 0.001)

        self.hidden_out_prime, _ = naf_net.hidden_layers(self.state, theta_hidden_prime, name= 'hidden_net_prime')

        theta_v_prime, update_theta_v = naf_net.exponential_moving_averages(theta_v, 0.001)
        V_prime, _ = naf_net.fc_layer(self.hidden_out, theta_v_prime, tf.identity, 'v_prime_layer')


        #creating the P matrix:
        pivot = 0
        rows = []
        for idx in xrange(self.action_dim):
            count = self.action_dim - idx

            diag_elem = tf.exp(tf.slice(l, (0, pivot), (-1, 1)))
            non_diag_elems = tf.slice(l, (0, pivot + 1), (-1, count - 1))
            row = tf.pad(tf.concat(1, (diag_elem, non_diag_elems)), ((0, 0), (idx, 0)))
            rows.append(row)

            pivot += count

        L = tf.transpose(tf.pack(rows), (1, 2, 0))
        P = tf.batch_matmul(L, tf.transpose(L, (0, 2, 1)))

        tmp = tf.expand_dims(self.action - self.mu, -1)
        A = -tf.batch_matmul(tf.transpose(tmp, (0, 2, 1)), tf.batch_matmul(P, tmp)) / 2
        A = tf.reshape(A, [-1, 1])

        with tf.name_scope('Q'):
            self.Q = A + self.V

        with tf.name_scope('optimization'):
            self.rew = tf.placeholder(tf.float32, [None, self.action_dim], name='reward')

            V_prime_stopped = tf.stop_gradient(V_prime)
            q_target = self.rew + self.gamma*V_prime_stopped
            self.loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(q_target), tf.squeeze(self.Q)), name='td_error_loss')

            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            grads_and_vars = optimizer.compute_gradients(self.loss, var_list=theta_hidden + theta_v + theta_mu + theta_l)

            with tf.control_dependencies([update_hidden, update_theta_v]):
                self.train_step = optimizer.apply_gradients(grads_and_vars)

        # logging
        log_obs = [] if self.state_dim > 20 else [tf.histogram_summary("obs/" + str(i), self.state[:, i]) for i in
                                                  range(self.state_dim)]
        log_act = [] if self.action_dim > 20 else [tf.histogram_summary("act/inf" + str(i), self.mu[:, i]) for i in
                                                   range(self.action_dim)]
        log_act2 = [] if self.action_dim > 20 else [tf.histogram_summary("act/train" + str(i), self.action[:, i]) for
                                                    i in range(self.action_dim)]

        log_grad = [plotting.grad_histograms(grads_and_vars)]

        self.log_all = tf.merge_summary(log_obs + log_act + log_act2)

        plotting.hist_summaries(*list(theta_v_prime + theta_hidden_prime))
        tf.scalar_summary('mean squared tderror', self.loss)


        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        self.merged = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(self.summaries_dir, self.sess.graph)
        tf.initialize_all_variables().run()

    def eval_mu(self, state):
        return self.mu.eval(feed_dict= {self.state : state})

    def eval_q(self, state, action):
        return self.Q.eval(feed_dict= {self.state : state, self.action : action})

    def eval_V(self, state):
        return self.V.eval(feed_dict= {self.state : state})


    def run_episode(self, enable_render=False, limit=5000, test_run = True):

        state = self.env_f.reset()
        self.ou_process.reset()

        count = 0
        done = False

        while ( not done ):

            if count>limit:
                return count

            count += 1
            state = np.squeeze(state).reshape((1,self.state_dim))  # convert (2,1) array in to (2,1)

            if test_run:
                action = self.eval_mu(state)
            else:
                action = self.eval_mu(state) + self.ou_process.ou_step()

            state_prime, reward, done, _ = self.env_f.step(action)

            if not test_run:
                self.replay_memory.append((state, action, reward, state_prime, done))
                self.samples_count += 1

                # if (len(self.replay_memory) > self.warmup) and (self.samples_count % (self.batch_size/2) == 0):
                if len(self.replay_memory) > self.warmup:
                    self.train_networks()

            state = state_prime

            if enable_render:
                self.env_f.render()

        return count


    def run(self, maxstep = 1e6, max_episode_length = 10000):

        it = 0
        while self.step < maxstep:
            # if it % 1000 == 0:
            #     print('replay size',len(self.replay_memory))

            # run episode
            episode_length = self.run_episode(test_run= False, enable_render=False, limit= max_episode_length)
            self.train_lengths.append(episode_length)

            if (it+1)% 10== 0:
                plotting.plot_replay_memory_2d_state_histogramm(self.replay_memory)

            if (it+1) % 5 == 0:
                # perform a test run with the target policy:
                self.test_lengths.append(self.run_episode(test_run= True, enable_render=False))

            if (it+1) % 10 == 0:
                plotting.plot_episode_lengths(self.train_lengths)
                plotting.plot_episode_lengths(self.test_lengths)

            it+=1


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


        def feed_dict():

            states, action, reward, states_prime, term2 = self.get_train_batch()

            return {self.state: states.squeeze().reshape(self.batch_size, self.state_dim),
                    self.action: action.squeeze().reshape(self.batch_size,1),
                    self.rew: reward.squeeze().reshape(self.batch_size,1),
                    self.state_prime: states_prime.squeeze().reshape(self.batch_size, self.state_dim),
                    }

        dict_ = feed_dict()
        summary, _, mse_val = self.sess.run([self.log_all, self.train_step, self.loss], feed_dict= dict_)

        if self.step % 10 == 0:
            self.train_writer.add_summary(summary, self.step)

        if self.step % 1000 == 0:
            print('result after minibatch no. {} : mean squared error: {}'.format(self.step, mse_val))
            # print('batch train data states', dict_[self.x_states])
            # print('batch train data actions', dict_[self.x_action])
            plotting.plot_learned_func(self.eval_mu, self.env_f)
            plotting.plot_learned_func(self.eval_V, self.env_f)
            plotting.plot_q_func(self.eval_q, self.env_f)

            # print('qs: ', self.q.eval(feed_dict = {self.x_states: dict_[self.x_states], self.x_action: dict_[self.x_action]}))

            # print('batch train data targets', dict_[self.q_targets])

        self.step += 1


    def main(self):
        if tf.gfile.Exists(self.summaries_dir):
            tf.gfile.DeleteRecursively(self.summaries_dir)
        tf.gfile.MakeDirs(self.summaries_dir)

        self.run()


if __name__ == '__main__':
    car = NAF()
    car.main()
