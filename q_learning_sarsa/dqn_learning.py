from __future__ import print_function
import numpy as np

np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import time
import math
import cPickle
import gym as gym

import qnn

class q_learning():
    def __init__(self,
                 gamma=0.99,
                 init_epsilon=1.0,
                 end_epsilon=0.1,
                 update_epsilon=True,
                 exploration_decrease_length = 1e6,
                 policy_mode='deterministic',
                 environment='MountainCar-v0',
                 # environment = 'Acrobot-v0',
                 lambda_=0.5,
                 plot_resolution=30,
                 nn_size_hidden = [300,400,400],
                 nn_batch_size = 50,
                 nn_learning_rate = 1e-4,
                 qnn_target = 'q-learning', # 'sarsa'
                 replay_memory_size = 1e6,
                 descent_method = 'grad'
                 ):

        self.env = gym.make(environment)
        self.select_env = environment

        self.num_actions = self.env.action_space.n
        self.prob_distrib = np.zeros(self.num_actions)
        self.statedim = self.env.observation_space.shape[0]

        # lengths of all the played episodes
        self.episode_lengths = []

        # lengths of all the tested episodes
        self.test_lengths = []


        self.plot_resolution = plot_resolution

        # w parametrizes a value for every action for each state
        # It is a single row, but is divided into as many sections as actions there are
        # Each of these sections gets multiplied by a different section of the features array
        self.eligibility_vector_theta = np.zeros(self.num_tile_features * self.num_actions)
        self.w = np.zeros(self.num_tile_features * self.num_actions)  # weights for q-function estimator

        self.lambda_ = lambda_

        ## stochastic or deterministic softmax-based actions
        self.policy_mode = policy_mode

        ## exploration parameters
        # too much exploration is wrong!!!
        self.epsilon = init_epsilon  # explore probability
        self.init_epsilon = init_epsilon
        self.end_epsilon = end_epsilon
        self.exploration_decrease_length = exploration_decrease_length
        self.update_epsilon = update_epsilon
        self.total_runs = 0.
        # too long episodes give too much negative reward!!!!
        # self.max_episode_length = 1000000
        # ----> Use gamma!!!!! TODO: slower decrease?
        self.gamma = gamma  # similar to 0.9

        if qnn_target == 'q-learning':
            self.is_a_prime_external = False
        elif qnn_target == 'sarsa':
            self.is_a_prime_external = True
        else:
            throw('ValueError')
        # simultaneous evaluation through neural network
        self.qnn = qnn.qnn(self.statedim,
                           self.num_actions,
                           size_hidden=nn_size_hidden,
                           batch_size=nn_batch_size,
                           learning_rate=nn_learning_rate,
                           is_a_prime_external=self.is_a_prime_external,
                           replay_memory_size=replay_memory_size,
                           descent_method=descent_method
                           )

        print('lambda', self.lambda_)
        print('using environment', environment)
        print('qnn target', qnn_target, self.is_a_prime_external, self.qnn.is_a_prime_external)


    # epsilon-greedy but deterministic or stochastic is a choice
    def policy(self, state, mode='deterministic', deepQ=False, w=None):
        if w is None:
            w = self.w

        explore = bool(np.random.choice([1, 0], p=[self.epsilon, 1 - self.epsilon]))

        # print(explore, features, end="")
        if mode == 'deterministic' and not explore:
            if deepQ:
                q = self.qnn.evaluate_all_actions(np.array(state).reshape((1,-1)))
                # print(state, q)
                return np.argmax(q.squeeze())#np.random.choice(np.argwhere(q == np.amax(q)).flatten())
            if not deepQ:
                raise ValueError('Option not defined')
                # q = features.dot(w)
                # return np.random.choice(np.argwhere(q == np.amax(q)).flatten())
        elif explore:
            # print('explore')
            return self.env.action_space.sample()



    def deepq_learning(self, num_iter=1000, max_steps=5000):

        for it in range(num_iter):

            #            episode = []
            prev_state = self.env.reset()
            prev_action = self.policy(prev_state, mode=self.policy_mode, deepQ=True)

            count = 0
            done = False
            while (not done):

                if count > max_steps:
                    self.episode_lengths.append(count)
                    break
                count += 1

                state, reward, done, info = self.env.step(prev_action)
                action = self.policy(state, mode=self.policy_mode, deepQ=True)
                #                episode.append((state, action, reward))

                # evaluation alone, to test a neural network
                if not self.is_a_prime_external:
                    # Q learning
                    self.qnn.train_batch(prev_state.reshape(1,-1), np.array(prev_action).reshape(-1), np.array(reward).reshape(-1), state.reshape(1,-1))
                else:
                    # SARSA (not converging)
                    self.qnn.train_batch(prev_state.reshape(1,-1), np.array(prev_action).reshape(-1), np.array(reward).reshape(-1), state.reshape(1,-1), np.array(action).reshape(-1))

                prev_state = state
                prev_action = action

                if (done):
                    self.episode_lengths.append(count)

                # decrease exploration
                if self.update_epsilon:
                    if self.epsilon > 0.1:
                        self.epsilon -= (self.init_epsilon - self.end_epsilon)*(1./self.exploration_decrease_length)

            if (it + 1) % 1 == 0:
                print("Episode %d" % (it))
                if (done): print("Length %d" % (self.episode_lengths[-1]))

            if (it + 1) % 10 == 0:
                print("exploration ", self.epsilon)
                self.plot_training()

            if (it + 1) % 10 == 0:
                if self.select_env == 'MountainCar-v0':
                    # print('last w', self.w)
                    self.plot_deepQ_policy(mode='deterministic')
                    self.plot_deepQ_function()

            # if (it + 1) % 10 == 0:
            #     self.plot_training()




    def plot_deepQ_function(self):

        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high

        # values to evaluate policy at
        x_range = np.linspace(obs_low[0], obs_high[0], self.plot_resolution)
        v_range = np.linspace(obs_low[1], obs_high[1], self.plot_resolution)
        states = []
        # the second index will change faster when doing np.reshape
        # this fits with the raw-wise change of X in np.meshgrid
        for state2 in v_range:
            for state1 in x_range:
                states.append((state1, state2))

        states = np.array(states)

        deepQ_all = self.qnn.evaluate_all_actions(states)
        print('statesshape', states.shape)
        print('deepQshape', deepQ_all.shape)

        for action in range(self.num_actions):
            print('plotting the evaluated deepQ-function for action {}'.format(action))

            # get values in a grid
            q_func = np.reshape(deepQ_all[:,action], (v_range.shape[0], x_range.shape[0]))

            print("")

            fig = plt.figure()

            ax = fig.add_subplot(111)
            X, Y = np.meshgrid(x_range, v_range)
            # ax.plot_surface(X, Y, q_func, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=True)
            im = ax.pcolormesh(X, Y, q_func)
            fig.colorbar(im)
            ax.set_xlabel("x")
            ax.set_ylabel("v")
            # ax.set_zlabel("negative value")
            plt.show()

        # plotting Q*
        print('plotting the evaluated deepQ-function star (optimal)')

        # get values in a grid
        q_func = np.reshape(np.max(deepQ_all, axis=1), (v_range.shape[0], x_range.shape[0]))

        print("")

        fig = plt.figure()

        ax = fig.add_subplot(111)
        X, Y = np.meshgrid(x_range, v_range)
        # ax.plot_surface(X, Y, q_func, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=True)
        im = ax.pcolormesh(X, Y, q_func)
        fig.colorbar(im)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        # ax.set_zlabel("negative value")
        plt.show()

    def plot_deepQ_policy(self, mode='deterministic'):
        resolution = self.plot_resolution

        # backup of value
        save_epsilon = self.epsilon
        self.epsilon = 0.0  # no exploration

        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high

        # values to evaluate policy at
        x_range = np.linspace(obs_low[0], obs_high[0], resolution)
        v_range = np.linspace(obs_low[1], obs_high[1], resolution)

        # get actions in a grid
        greedy_policy = np.zeros((resolution, resolution))
        for i, x in enumerate(x_range):
            for j, v in enumerate(v_range):
                # print(np.argmax(self.get_features((x,v)).dot(self.theta)), end="")
                greedy_policy[i, j] = self.policy((x, v), mode, deepQ=True)
        print("")

        # plot policy
        fig = plt.figure()
        plt.imshow(greedy_policy,
                   cmap=plt.get_cmap('gray'),
                   interpolation='none',
                   extent=[obs_low[1], obs_high[1], obs_high[0], obs_low[0]],
                   aspect="auto")
        plt.xlabel("velocity")
        plt.ylabel("position")
        plt.show()

        # restore value
        self.epsilon = save_epsilon

    def plot_training(self):

        if any(np.array(self.episode_lengths > 0).flatten()):
            fig = plt.figure()
            plt.plot(self.episode_lengths)
            plt.yscale('log')
            plt.xlabel("episodes")
            plt.ylabel("timesteps")
            plt.show()

    def plot_testing(self):

        fig = plt.figure()
        plt.plot(self.test_lengths)
        plt.yscale('log')
        plt.show()
