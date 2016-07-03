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
                 update_epsilon=True,
                 policy_mode='deterministic',
                 N_0=50.,
                 random_init_theta=True,
                 negative_gradient=False,
                 environment='MountainCar-v0',
                 # environment = 'Acrobot-v0',
                 lambda_=0.5,
                 algorithm='Q-learn',
                 tile_resolution=10,
                 alpha_w=5e-4,
                 forget_rate=0.0,
                 nn_size_hidden = [300,400,400],
                 nn_batch_size = 50,
                 nn_learning_rate = 1e-4,
                 qnn_target = 'q-learning' # 'sarsa'
                 ):

        self.forget_rate = forget_rate

        self.alpha_w = alpha_w

        self.algorithm = algorithm
        self.env = gym.make(environment)
        self.select_env = environment

        self.num_actions = self.env.action_space.n
        self.prob_distrib = np.zeros(self.num_actions)
        self.statedim = self.env.observation_space.shape[0]

        # lengths of all the played episodes
        self.episode_lengths = []

        # running average of the mean value of the episodes' steps
        self.mean_value_fcn = 0.0

        # lengths of all the tested episodes
        self.test_lengths = []

        ## policy parameters initialization

        # tile features
        self.tile_resolution = tile_resolution

        self.overlap = False

        if self.overlap:
            self.num_tile_features = pow(self.tile_resolution, self.statedim) * 2
        else:
            self.num_tile_features = pow(self.tile_resolution, self.statedim)

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
        self.update_epsilon = update_epsilon
        self.total_runs = 0.
        # too long episodes give too much negative reward!!!!
        # self.max_episode_length = 1000000
        # ----> Use gamma!!!!! TODO: slower decrease?
        self.gamma = gamma  # similar to 0.9

        self.N_0 = N_0

        if qnn_target == 'q-learning':
            self.is_a_prime_external = False
        elif qnn_target == 'sarsa':
            self.is_a_prime_external = True
        else:
            throw('ValueError')
        # simultaneous evaluation through neural network
        self.qnn = qnn.qnn(self.statedim, self.num_actions, size_hidden=nn_size_hidden, batch_size=nn_batch_size, learning_rate=nn_learning_rate, is_a_prime_external=self.is_a_prime_external)

        print('N_0', self.N_0)
        print('lambda', self.lambda_)
        print('using environment', environment)
        print('tile resolution', self.tile_resolution)
        print('qnn target', qnn_target, self.is_a_prime_external, self.qnn.is_a_prime_external)


    def get_tile_feature(self, state):

        high = self.env.observation_space.high
        obs_dim = self.env.observation_space.shape[0]  # dimension of observation space
        low = self.env.observation_space.low
        numactions = self.env.action_space.n

        stepsize = (high - low) / self.tile_resolution

        ind = np.floor((state - low) / stepsize).astype(int)

        ind[ind >= self.tile_resolution] = self.tile_resolution - 1  # bound the index so that it doesn't exceed bounds
        ind = tuple(ind)

        grid = np.zeros(np.ones(obs_dim) * self.tile_resolution)
        try:
            grid[ind] = 1
        except IndexError, error:
            print(error)
            print("ind", ind)
            print("state", state)
            print("high", high)
            print("low", low)

            return

        if self.overlap:

            ind_shift = np.floor((state - low + stepsize / 2) / stepsize).astype(int)
            ind_shift[
                ind_shift >= self.tile_resolution] = self.tile_resolution - 1  # bound the index so that it doesn't exceed bounds
            ind_shift = tuple(ind_shift)

            grid_shift = np.zeros(np.ones(obs_dim) * self.tile_resolution)
            grid_shift[ind_shift] = 1

            flatgrid = np.concatenate((grid, grid_shift), axis=0).flatten()

        else:
            flatgrid = grid.flatten()
        return flatgrid

    def get_full_feature(self, state):
        """
        Returns a matrix containing as many rows as actions.
        Each row contains the same state features at different positions (corresponding to the different actions)
        """

        flatgrid = self.get_tile_feature(state)
        length_flatgrid = flatgrid.shape[0]

        full_feature = np.zeros((self.num_actions, length_flatgrid * self.num_actions))
        for action in range(self.num_actions):
            full_feature[action, length_flatgrid * action: (action + 1) * length_flatgrid] = flatgrid

        return full_feature

    # TODO: better solution to over-/underflows? Maybe not needed if proper learning?
    def eval_action_softmax_probabilities(self, full_feature):

        # softmax probability
        activation = full_feature.dot(self.w)
        activation -= np.max(activation)

        # trying to workaround over and underflows
        solved = False
        while (not solved):
            solved = True
            try:
                prob_distrib = np.exp(activation)
                prob_distrib /= np.sum(prob_distrib)
            except FloatingPointError, error:
                solved = False
                # print("potential error in softmax")
                print(error)
                # print(".", end="")

                prob_distrib[np.log(prob_distrib) < -100] = 0  # print("activation ", activation)

        return prob_distrib

    # epsilon-greedy but deterministic or stochastic is a choice
    def policy(self, state, mode='deterministic', deepQ=False, w=None):
        if w is None:
            w = self.w

        explore = bool(np.random.choice([1, 0], p=[self.epsilon, 1 - self.epsilon]))

        features = self.get_full_feature(state)
        # print(explore, features, end="")
        if mode == 'deterministic' and not explore:
            if deepQ:
                q = self.qnn.evaluate_all_actions(np.array(state).reshape((1,-1)))
                # print(state, q)
                return np.argmax(q.squeeze())#np.random.choice(np.argwhere(q == np.amax(q)).flatten())
            if not deepQ:
                q = features.dot(w)
                return np.random.choice(np.argwhere(q == np.amax(q)).flatten())
        elif explore:
            # print('explore')
            return self.env.action_space.sample()

    def run_episode(self, enable_render=False, limit=50000):
        episode = []
        state = self.env.reset()

        count = 0
        done = False
        while (not done):

            if len(episode) > limit:
                return []

            count += 1

            action = self.policy(state, mode=self.policy_mode)
            state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
            if enable_render: self.env.render()
            # if count > self.max_episode_length: break;

        if enable_render: print("This episode took {} steps".format(count))

        return episode





    def train_online(self, num_iter=1000, max_steps=5000, dataname='unnamed_data', save=False):

        for it in range(num_iter):

            #            episode = []
            prev_state = self.env.reset()
            prev_action = self.policy(prev_state, mode=self.policy_mode)
            prev_tile_features = self.get_full_feature(prev_state)[prev_action]
            Qs = prev_tile_features.dot(self.w)

            count = 0
            done = False
            self.eligibiltiy_vector = np.zeros(self.num_tile_features * self.num_actions)
            self.eligibility_vector_theta = np.zeros(self.num_tile_features * self.num_actions)
            while (not done):

                if count > max_steps:
                    self.episode_lengths.append(count)
                    break
                count += 1

                state, reward, done, info = self.env.step(prev_action)
                action = self.policy(state, mode=self.policy_mode)
                #                episode.append((state, action, reward))

                # select the proper line from the full feature matrix (one line per action)
                tile_features = self.get_full_feature(state)[action]

                Qs_prime = tile_features.dot(self.w)

                delta_t = reward + self.gamma * Qs_prime - Qs

                self.eligibiltiy_vector = self.eligibiltiy_vector * self.gamma * self.lambda_ + tile_features

                # self.w += self.alpha_w * delta_t * self.eligibiltiy_vector
                self.w = (1-self.forget_rate)*self.w + self.alpha_w * delta_t * self.eligibiltiy_vector

                # # evaluation alone, to test a neural network
                # self.qnn.train_wait_for_batch(prev_state.reshape(1,-1), np.array(prev_action).reshape(-1), np.array(reward).reshape(-1), state.reshape(1,-1))

                prev_state = state
                prev_action = action
                prev_tile_features = tile_features
                Qs = Qs_prime

                if (done):
                    self.episode_lengths.append(count)

            print("Episode %d" % (it))
            if (done): print("Length %d" % (self.episode_lengths[-1]))

            # print('sum tile features ', tile_features_mat[idx].sum())
            print('max w', self.w.max())
            print('min w', self.w.min())

            # print('last eligibility for theta',self.eligibility_vector_theta)
            print('last td-error', delta_t)

            if (it + 1) % 10 == 0:

                if self.select_env == 'MountainCar-v0':
                    # print('last w', self.w)
                    self.plot_policy(mode='deterministic')
                    self.plot_q_function()
                    # self.plot_deepQ_function()

            if (it + 1) % 10 == 0:

                # do a test run
                save_epsilon = self.epsilon

                self.epsilon = 0.
                limit = 10000
                det_episode = self.run_episode(limit=limit)
                if det_episode == []:
                    len_episode = limit
                else:
                    len_episode = len(det_episode)

                self.test_lengths.append(len_episode)

                # restore epsilon
                self.epsilon = save_epsilon

            if (it + 1) % 10 == 0:
                self.plot_training()
                self.plot_testing()

            # decrease exploration
            if self.update_epsilon:
                self.epsilon = self.N_0 / (self.N_0 + self.total_runs)

            if save:
                self.savedata(dataname=dataname)





    def deepq_evaluate(self, num_iter=1000, max_steps=5000, dataname='unnamed_data', save=False):

        epsilon_backup = self.epsilon
        self.epsilon = 0.0

        for it in range(num_iter):

            #            episode = []
            prev_state = self.env.reset()
            prev_action = self.policy(prev_state, mode=self.policy_mode)

            count = 0
            done = False
            while (not done):

                if count > max_steps:
                    self.episode_lengths.append(count)
                    break
                count += 1

                state, reward, done, info = self.env.step(prev_action)
                action = self.policy(state, mode=self.policy_mode)
                #                episode.append((state, action, reward))

                # evaluation alone, to test a neural network
                if not self.is_a_prime_external:
                    # Q learning
                    self.qnn.train_batch(prev_state.reshape(1,-1), np.array(prev_action).reshape(-1), np.array(reward).reshape(-1), state.reshape(1,-1))
                else:
                    # SARSA
                    self.qnn.train_batch(prev_state.reshape(1,-1), np.array(prev_action).reshape(-1), np.array(reward).reshape(-1), state.reshape(1,-1), np.array(action).reshape(-1))

                prev_state = state
                prev_action = action

                if (done):
                    self.episode_lengths.append(count)

            if (it + 1) % 100 == 0:
                print("Episode %d" % (it))
                if (done): print("Length %d" % (self.episode_lengths[-1]))

            if (it + 1) % 100 == 0:
                if self.select_env == 'MountainCar-v0':
                    # print('last w', self.w)
                    self.plot_deepQ_policy(mode='deterministic')
                    self.plot_deepQ_function()

            # if (it + 1) % 10 == 0:
            #     self.plot_training()

        self.epsilon = epsilon_backup


    def plot_eligibility(self):

        if self.overlap:
            e_vector = self.eligibiltiy_vector[0:self.num_tile_features / 2]  # just visualize first half
        else:
            e_vector = np.array(self.eligibiltiy_vector)

        print('plotting the eligibility traces')

        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high

        # values to evaluate policy at
        x_range = np.linspace(obs_low[0], obs_high[0], self.tile_resolution)
        v_range = np.linspace(obs_low[1], obs_high[1], self.tile_resolution)

        # get actions in a grid
        print(e_vector.shape)
        e_mat = e_vector.reshape((self.tile_resolution, self.tile_resolution))

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_range, v_range)
        ax.plot_wireframe(X, Y, e_mat)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_zlabel("eligibility")
        plt.show()

    def plot_q_function(self, w=None):
        if w is None:
            w = self.w

        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high

        # values to evaluate policy at
        x_range = np.linspace(obs_low[0], obs_high[0], self.tile_resolution * 3 + 1)
        v_range = np.linspace(obs_low[1], obs_high[1], self.tile_resolution * 3)

        # get actions in a grid
        q_func = np.zeros((v_range.shape[0], x_range.shape[0], self.num_actions))
        q_opt = np.zeros((v_range.shape[0], x_range.shape[0]))
        for i, state1 in enumerate(x_range):
            for j, state2 in enumerate(v_range):
                # print(np.argmax(self.get_features((x,v)).dot(self.theta)), end="")
                q_func[j, i, :] = w.dot(self.get_full_feature((state1, state2)).T)
        q_opt = np.max(q_func, axis=2)

        for action in range(self.num_actions):
            print('plotting the q-function for action {}'.format(action))

            print("")

            fig = plt.figure()

            # ax = fig.add_subplot(111, projection='3d')
            ax = fig.add_subplot(111)
            X, Y = np.meshgrid(x_range, v_range)
            # ax.plot_surface(X, Y, q_func, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=True)
            im = ax.pcolormesh(X, Y, q_func[:,:,action])
            fig.colorbar(im)
            ax.set_xlabel("x")
            ax.set_ylabel("v")
            # ax.set_zlabel("negative value")
            plt.show()

        print('plotting the q^star-function (optimal)')

        print("")

        fig = plt.figure()

        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111)
        X, Y = np.meshgrid(x_range, v_range)
        # ax.plot_surface(X, Y, q_func, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=True)
        im = ax.pcolormesh(X, Y, q_opt)
        fig.colorbar(im)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        # ax.set_zlabel("negative value")
        plt.show()

    def plot_deepQ_function(self):

        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high

        # values to evaluate policy at
        x_range = np.linspace(obs_low[0], obs_high[0], self.tile_resolution +1)
        v_range = np.linspace(obs_low[1], obs_high[1], self.tile_resolution +1)
        states = []
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

    def savedata(self, dataname):

        output = open(dataname, 'wb')
        cPickle.dump(self.theta, output)
        cPickle.dump(self.episode_lengths, output)

        cPickle.dump(self.test_lengths, output)
        cPickle.dump(self.v, output)
        cPickle.dump(self.w, output)
        output.close()

    def loaddata(self, dataname):

        pkl_file = open(dataname, 'rb')
        self.theta = cPickle.load(pkl_file)
        self.episode_lengths = cPickle.load(pkl_file)
        self.test_lengths = cPickle.load(pkl_file)
        self.v = cPickle.load(pkl_file)
        self.w = cPickle.load(pkl_file)

        print(self.theta)
        print(self.episode_lengths)
        print(self.episode_lengths)

        pkl_file.close()

    def plot_policy(self, resolution=100, mode='deterministic', w=None):
        if w is None:
            w = self.w

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
                greedy_policy[i, j] = self.policy((x, v), mode, w=w)
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

    def plot_deepQ_policy(self, resolution=50, mode='deterministic'):

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
