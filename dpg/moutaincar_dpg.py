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

class mountaincar_dpg():

    def __init__(self,
                 gamma=0.99,
                 N_0=50.,
                 random_init_theta=False,
                 environment = 'MountainCarContinuous-v0',
                 algorithm = 'dpg1'
                 ):

        self.epsilon = 0.1

        self.algorithm = algorithm
        self.env = gym.make(environment)
        self.select_env = environment

#        self.num_actions = self.env.action_space.n
#        self.prob_distrib = np.zeros(self.num_actions)
        self.statedim = self.env.observation_space.shape[0]


        # lengths of all the played episodes
        self.episode_lengths = []

        # running average of the mean value of the episodes' steps
        self.mean_value_fcn = 0.0

        # lengths of all the tested episodes TODO
        self.test_lengths = []

        ## policy parameters initialization

        #tile features
        self.tile_resolution = 10.0

        self.overlap = True

        if self.overlap:
            self.num_tile_features = int(pow(self.tile_resolution,self.statedim)*2)
        else:
            self.num_tile_features = int(pow(self.tile_resolution,self.statedim))


        self.gamma = gamma  #similar to 0.9

        self.N_0 = N_0

        ######################### dpg params ###############

        if random_init_theta:
            # random initialization
            self.theta = np.random.randn(self.num_tile_features)*0.001
        else:
            # zero initialization
            self.theta = np.zeros(self.num_tile_features)

        # for the value function estimation:
        self.v = np.zeros(self.num_tile_features)  # weights for value function estimator

        self.w = np.zeros(self.num_tile_features)  # weights for q-function estimator

        self.gamma = 0.9

        self.sigma_b = 1e-1   #standard deviation for behavior policy

        self.alpha_theta = 1e-3
        self.alpha_w = 1e-2
        self.alpha_v = 1e-2


        print('N_0',self.N_0)
        print('using environment',environment)
        print('tile resolution',self.tile_resolution)

    def beta(self,state):
        # behavior policy
         return np.random.randn(1)*self.sigma_b + self.mu(state)

    def mu(self, state):
        # target policy
          return self.theta.dot(self.get_tile_feature(state))

    def V(self,state):
        return self.v.dot(self.get_tile_feature(state))

    def nabla_mu(self,state):
        return self.get_tile_feature(state)

    def Qw(self,state,action):
        # calc Qfunction
        return (action- self.mu(state))*self.nabla_mu(state).dot(self.w) + self.V(state)



    def get_tile_feature(self, state):

        high = self.env.observation_space.high
        obs_dim = self.env.observation_space.shape[0]   #dimension of observation space
        low = np.asarray(self.env.observation_space.low)
#        numactions = self.env.action_space.n


        stepsize = (high - low)/self.tile_resolution

        ind = np.floor((state-low)/stepsize).astype(int)


        ind[ind>=self.tile_resolution]=self.tile_resolution-1  #bound the index so that it doesn't exceed bounds
        ind = tuple(ind)

        grid = np.zeros(np.ones(obs_dim)*self.tile_resolution)
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

            ind_shift = np.floor((state-low+stepsize/2)/stepsize).astype(int)
            ind_shift[ind_shift>=self.tile_resolution]=self.tile_resolution-1  #bound the index so that it doesn't exceed bounds
            ind_shift = tuple(ind_shift)

            grid_shift = np.zeros(np.ones(obs_dim)*self.tile_resolution)
            grid_shift[ind_shift] = 1

            flatgrid = np.concatenate((grid,grid_shift), axis= 0).flatten()

        else:
            flatgrid = grid.flatten()


        return flatgrid



    def run_episode(self, enable_render=False, limit=5000):
        episode = []
        state = self.env.reset()
        print('size state 0', state.shape)

        count = 0
        done = False
        #while ( not done ):

        if len(episode)>limit:
            return []

        count += 1

        action = self.beta(state)

        state_prime, reward, done, info = self.env.step(action)
        state_prime = np.squeeze(state_prime)

        delta_t = reward + self.gamma * self.Qw(state_prime, self.mu(state_prime)) - self.Qw(state,action)

        self.theta += self.alpha_theta * self.nabla_mu(state)*self.nabla_mu(state).dot(self.w)


        self.w += self.alpha_w *delta_t*(action - self.mu(state))*self.nabla_mu(state)

        self.v += self.alpha_v *delta_t* self.get_tile_feature(state)

        state = state_prime

        episode.append((state, action, reward))
        if enable_render: self.env.render()
        # if count > self.max_episode_length: break;

        if enable_render: print("This episode took {} steps".format(count))

        return episode



    def start_training(self, max_episodes=1000, dataname ='unnamed_data', save = False):
        # fig = plt.figure()

        for it in range(max_episodes):

            # run episode
            episode = self.run_episode(enable_render=False)

            self.episode_lengths.append(len(episode))

            if (it+1)%1 == 0:
                # output training info
                print("EPISODE #{}".format(it))
                print("with a exploration of {}%".format(self.epsilon*100))
#                print("and learning rate of {}".format(self.alpha))
                print("lasted {0} steps".format(len(episode)))

                # do a test run
                # save_policy_mode = self.policy_mode
                save_epsilon = self.epsilon
                # print(self.policy_mode)

                # self.policy_mode = "deterministic"
                self.epsilon = 0.
                limit = 10000
                det_episode = self.run_episode(limit=limit)
                if det_episode == []:
                    len_episode = limit
                else:
                    len_episode = len(det_episode)

                self.test_lengths.append(len_episode)
                # self.policy_mode = save_policy_mode
                self.epsilon = save_epsilon



            # print('sum tile features ', tile_features_mat[idx].sum())
            print('max theta', self.theta.max())
            print('min theta', self.theta.min())

            if (it+1)%1 == 0:

                if self.select_env == 'MountainCar-v0':

                    print("theta")
                    print(self.theta)
                    print('last v', self.v)
                    self.plot_policy(mode= 'stochastic')
                    self.plot_policy(mode= 'deterministic')
                    self.plot_value_function()

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

    def plot_eligibility(self):

        if self.overlap:
            e_vector = self.eligibiltiy_vector[0:self.num_tile_features/2]  #just visualize first half
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
        e_mat = e_vector.reshape((self.tile_resolution,self.tile_resolution))

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_range, v_range)
        ax.plot_wireframe(X,Y, e_mat)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_zlabel("eligibility")
        plt.show()

    def plot_q_function(self):

        for action in range(self.num_actions):
            print('plotting the q-function for action {}'.format(action))

            obs_low = self.env.observation_space.low
            obs_high = self.env.observation_space.high

            # values to evaluate policy at
            x_range = np.linspace(obs_low[0], obs_high[0]-0.01, self.tile_resolution*3)
            v_range = np.linspace(obs_low[1], obs_high[1]-0.01, self.tile_resolution*3)

            # get actions in a grid
            q_func = np.zeros((x_range.shape[0], v_range.shape[0]))
            for i, state1 in enumerate(x_range):
                for j, state2 in enumerate(v_range):
                    # print(np.argmax(self.get_features((x,v)).dot(self.theta)), end="")
                    q_func[i,j] = -self.w.dot(self.get_full_feature((state1,state2))[action])
            print("")

            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(x_range, v_range)
            ax.plot_surface(X, Y, q_func, rstride=1, cstride=1, cmap=cm.jet,linewidth=0.1, antialiased=True)
            ax.set_xlabel("x")
            ax.set_ylabel("v")
            ax.set_zlabel("negative value")
            plt.show()

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
                value_func[i,j] = -self.V((state1,state2))
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


    def plot_policy(self, resolution=100, mode= 'stochastic'):

        # backup of value
        save_epsilon = self.epsilon
        self.epsilon = 0.0      # no exploration

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
                if mode == 'stocastic':
                    greedy_policy[i,j] = self.mu((x,v))
                elif mode == 'deterministic':
                    greedy_policy[i, j] = self.beta((x, v))

        print("")

        # plot policy
        fig = plt.figure()
        plt.imshow(greedy_policy,
                   cmap=plt.get_cmap('gray'),
                   interpolation='none',
                   extent=[obs_low[1],obs_high[1],obs_high[0],obs_low[0]],
                   aspect="auto")
        plt.xlabel("velocity")
        plt.ylabel("position")
        plt.show()

        # restore value
        self.epsilon = save_epsilon


    def plot_training(self):

        fig = plt.figure()
        plt.plot(self.episode_lengths)
        plt.yscale('log')
        plt.show()
        plt.xlabel("episodes")
        plt.ylabel("timesteps")



    def plot_testing(self):

        fig = plt.figure()
        plt.plot(self.test_lengths)
        plt.yscale('log')
        plt.show()
