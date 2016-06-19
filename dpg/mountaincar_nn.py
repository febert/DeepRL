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

from nn import nn

class mountaincar_nn():

    def __init__(self,
                 gamma=0.99,
                 N_0=50.,
                 random_init_theta=False,
                 environment = 'MountainCarContinuous-v0',
                 algorithm = 'dpg1',
                 ):

        self.algorithm = algorithm
        self.env = gym.make(environment)
        self.select_env = environment

        self.action_limits = (self.env.action_space.low, self.env.action_space.high)
        print('action limits', self.action_limits)
        actionmean = (self.action_limits[0]+ self.action_limits[1])/2

#        self.num_actions = self.env.action_space.n
#        self.prob_distrib = np.zeros(self.num_actions)
        self.statedim = self.env.observation_space.shape[0]


        # lengths of all the played episodes
        self.episode_lengths = []
        # lengths of episodes run with target policy
        self.test_lengths = []

        #tile parameteres
        self.tile_resolution = 10.0
        self.overlap = False
        if self.overlap:
            self.num_tile_features = int(pow(self.tile_resolution,self.statedim)*2)
        else:
            self.num_tile_features = int(pow(self.tile_resolution,self.statedim))


        self.gamma = gamma  #similar to 0.9

        self.N_0 = N_0

        ######################### dpg params ###############


        if random_init_theta:
            # random initialization
            self.theta = np.ones(self.num_tile_features)*actionmean + np.random.randn(self.num_tile_features)*0.1
        else:
            # initialization with "no" actions (corresponds to action = 1)
            self.theta = np.ones(self.num_tile_features)*actionmean

        # for the value function estimation:
        self.v = np.zeros(self.num_tile_features)  # weights for value function estimator

        self.w = np.zeros(self.num_tile_features)  # weights for q-function estimator

        self.sigma_b = 1   #standard deviation for behavior policy

        self.alpha_theta = 1e-3
        self.alpha_w = 1e-2
        self.alpha_v = 1e-2

        print('N_0',self.N_0)
        print('using environment',environment)
        print('tile resolution',self.tile_resolution)
        print('gamma',self.gamma)

        # create neural network:
        self.nn1 = nn()
        self.nn1.main()


    def beta(self,state):
        # behavior policy
        beta_out = np.random.randn(1)*self.sigma_b + self.mu(state)

        return beta_out

    def mu(self, state):
        # target policy

        try:
            m_out = self.theta.dot(self.get_tile_feature(state))
        except TypeError:
            print('theta', self.theta)
            print('getting tile for state: ', state)
            print('self.get_tile_feature(state)', self.get_tile_feature(state))

        return m_out

    def V(self,state):
        return self.v.dot(self.get_tile_feature(state))

    def nabla_mu(self,state):
        return self.get_tile_feature(state)

    def Qw(self,state,action):
        # calc Qfunction
        return (action- self.mu(state))*self.nabla_mu(state).dot(self.w)  + self.V(state)

    def get_tile_feature(self, state):

        high = np.asarray(self.env.observation_space.high)
        obs_dim = self.env.observation_space.shape[0]       #dimension of observation space
        low = np.asarray(self.env.observation_space.low)

        stepsize = (high - low)/self.tile_resolution

        ind = np.floor((state-low)/stepsize).astype(int)


        ind[ind>=self.tile_resolution]=self.tile_resolution-1  #bound the index so that it doesn't exceed bounds
        ind = tuple(ind)

        grid = np.zeros(np.ones(obs_dim)*self.tile_resolution)
        try:
            grid[ind] = 1
        except IndexError, error:
            print(error)
            print('stepsize', stepsize)
            print('size stepsize', stepsize.shape)
            print("ind", ind)
            print("state", state)
            print("state size", state.shape)
            print("high", high)
            print("low", low)

            return


        if self.overlap:

            ind_shift = np.floor((state-low+stepsize/2)/stepsize).astype(int)
            ind_shift[ind_shift>=self.tile_resolution]=self.tile_resolution-1  #bound the index so that it doesn't exceed bounds
            ind_shift = tuple(ind_shift)

            grid_shift = np.zeros(np.ones(obs_dim)*self.tile_resolution)
            grid_shift[ind_shift] = 1

            flatgrid = np.concatenate((grid,grid_shift), axis= 0).flatten()

        else:
            flatgrid = grid.flatten()

        return flatgrid


    def apply_limits(self,action):

        if action < self.action_limits[0]:
            action = self.action_limits[0]

        if action > self.action_limits[1]:
            action = self.action_limits[1]

        return action


    def run_episode(self, enable_render=False, limit=20000):

        episode = []
        state = self.env.reset()

        count = 0
        done = False

        while ( not done ):

            if len(episode)>limit:
                return episode

            count += 1

            action = self.beta(state)
            action = self.apply_limits(action)

            state_prime, reward, done, info = self.env.step(action)
            state_prime = np.squeeze(state_prime)

            delta_t = reward + self.gamma * self.Qw(state_prime, self.mu(state_prime)) - self.Qw(state,action)

            self.theta += self.alpha_theta * self.nabla_mu(state)*self.nabla_mu(state).dot(self.w)

            self.w += self.alpha_w *delta_t*(action - self.mu(state))*self.nabla_mu(state)

            self.v += self.alpha_v *delta_t* self.get_tile_feature(state)

            state = state_prime

            episode.append((state, action, reward))

            # save mu to batch and train neural network
            # self.nn1.add_to_batch(state, self.mu(state))

            if enable_render:
                self.env.render()
                # print("step no. {}".format(count))

        return episode

    def run_target_episode(self, enable_render=False, limit=5000):

        episode = []
        state = self.env.reset()

        count = 0
        done = False

        while ( not done ):

            if len(episode)>limit:
                return episode

            count += 1
            state = np.squeeze(state)  # convert (2,1) array in to (2,)
            action = self.mu(state)
            action = self.apply_limits(action)
            # print('action',action)

            state, reward, done, info = self.env.step(action)

            episode.append((state, action, reward))

            if enable_render:
                self.env.render()

        return episode


    def start_training(self, max_episodes=100, dataname ='unnamed_data', save = False, max_episode_length = 20000):
        # fig = plt.figure()

        for it in range(max_episodes):

            # run episode
            episode = self.run_episode(enable_render=False, limit= max_episode_length)

            self.episode_lengths.append(len(episode))

            # perform a test run with the target policy:
            self.test_lengths.append(len(self.run_target_episode(enable_render=False)))

            print("Finished run #{}".format(it + 1))
            print("lasted {0} steps".format(len(episode)))

            if (it+1)%1 == 0:

                if self.select_env == 'MountainCarContinuous-v0':
                    #print("theta")
                    #print(self.theta)
                    #print('last v', self.v)

                    #print("beta: ")
                    #self.plot_policy(mode= 'stochastic')
                    print("mu: ")
                    self.plot_policy(mode= 'deterministic')

                    self.nn1.plot_learned_function()
                    #self.plot_value_function()


            # print('sum tile features ', tile_features_mat[idx].sum())
            print('max theta', self.theta.max())
            print('min theta', self.theta.min())


            if (it+1)%10 == 0:
                self.plot_training()
                self.plot_testing()

            # # decrease exploration
            # if self.update_epsilon:
            #     self.epsilon = self.N_0 / (self.N_0 + self.total_runs)
            # # decrease alpha
            # if not self.constant_alpha:
            #     self.alpha = self.init_alpha / np.sqrt(self.total_runs)

            if save:
                self.savedata(dataname=dataname)

        return self.theta


    def plot_value_function(self):

        print('plotting the value function')

        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high

        # values to evaluate policy at
        x_range = np.linspace(obs_low[0], obs_high[0]-0.01, self.tile_resolution*3)
        v_range = np.linspace(obs_low[1], obs_high[1]-0.01, self.tile_resolution*3)

        # get actions in a grid
        value_func = np.zeros((x_range.shape[0], v_range.shape[0]))
        for i, state1 in enumerate(x_range):
            for j, state2 in enumerate(v_range):
                value_func[j,i] = -self.V((state1,state2))
        print("")

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_range, v_range)
        ax.plot_surface(X, Y, value_func, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=True)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_zlabel("negative value")
        plt.show()


    def savedata(self, dataname):

        output = open(dataname, 'wb')
        cPickle.dump(self.theta, output)
        cPickle.dump(self.episode_lengths, output)

        cPickle.dump(self.test_lengths, output)
        cPickle.dump(self.v,output)
        cPickle.dump(self.w,output)
        output.close()

    def loaddata(self, dataname):

        pkl_file = open(dataname, 'rb')
        self.theta = cPickle.load(pkl_file)
        self.episode_lengths = cPickle.load(pkl_file)
        self.test_lengths = cPickle.load(pkl_file)
        self.v = cPickle.load(pkl_file)
        self.w = cPickle.load(pkl_file)

        print( self.theta)
        print( self.episode_lengths)
        print( self.episode_lengths)


        pkl_file.close()


    def plot_policy(self, mode= 'stochastic'):

        resolution = self.tile_resolution*2

        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high

        # values to evaluate policy at
        x_range = np.linspace(obs_low[0], obs_high[0], resolution)
        v_range = np.linspace(obs_low[1], obs_high[1], resolution)

        # get actions in a grid
        policy_vals = np.zeros((resolution, resolution))
        for i, x in enumerate(x_range):
            for j, v in enumerate(v_range):
                if mode == 'stochastic':
                    policy_vals[j,i] = self.beta((x,v))
                elif mode == 'deterministic':
                    policy_vals[j,i] = self.mu((x,v))

        # print("policy values:", policy_vals)

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_range, v_range)
        ax.plot_surface(X, Y, policy_vals, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=True)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_zlabel("action")
        plt.show()


    def plot_training(self):

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

        plt.xlabel("episodes")
        plt.ylabel("timesteps")

        plt.show()


if __name__ == '__main__':

    car1 = mountaincar_nn()
    car1.start_training()