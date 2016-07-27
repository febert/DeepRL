from __future__ import print_function
from __future__ import division
import numpy as np

np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt
from matplotlib import cm
# from mpl_toolkits.mplot3d import axes3d
from matplotlib.colors import LogNorm
# import time
import math
# import cPickle
import gym as gym
from PIL import Image
from PIL import ImageOps
from collections import deque
import copy

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
                 descent_method = 'grad',
                 dropout_keep_prob = 1.0,
                 ema_decay_rate = 0.999,
                 init_weights = None,
                 num_steps_until_train_step = None,
                 train_frequency = 1.0,
                 from_pixels = False,
                 repeat_action_times = 2
                 ):
        self.from_pixels = from_pixels
        self.repeat_action_times = repeat_action_times
        self.frame_downscaling = 6

        if num_steps_until_train_step is None:
            num_steps_until_train_step = nn_batch_size

        self.env = gym.make(environment)
        self.env_name = environment

        self.num_actions = self.env.action_space.n
        self.prob_distrib = np.zeros(self.num_actions)
        self.statedim = self.env.observation_space.shape[0]

        # lengths of all the played episodes
        self.episode_lengths = []
        self.total_train_episodes = 0

        # lengths of all the tested episodes
        if self.env_name=='MountainCar-v0':
            self.max_test_length = 1000
        elif self.env_name=='CartPole-v0':
            self.max_test_length = 10000
        else:
            self.max_test_length = 1000
        self.test_lengths = []
        self.test_lengths_std = []
        self.test_its = []
        self.test_runs_to_average = 5


        self.plot_resolution = plot_resolution


        self.lambda_ = lambda_

        ## stochastic or deterministic softmax-based actions
        self.policy_mode = policy_mode

        normalization_mean = None
        normalization_var = None
        if not self.from_pixels:
            # STATE NORMALIZATION
            print('Calculating normalization by random action sampling...')
            states = []

            while len(states) < 1e5:
                self.env.reset()
                done = False
                while not done:
                    state, _, done, _ = self.env.step(self.env.action_space.sample())
                    states.append(state)

            normalization_mean = np.mean(states, axis=(0)).astype(np.float32)
            normalization_var = np.var(states, axis=(0)).astype(np.float32)

        # if self.env_name == 'CartPole-v0':
        #     normalization_mean = np.zeros_like(normalization_mean)
        #     normalization_var = np.ones_like(normalization_var)

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

        # DEPRECATED
        if qnn_target == 'q-learning':
            self.is_a_prime_external = False
        elif qnn_target == 'sarsa':
            self.is_a_prime_external = True
        else:
            throw('ValueError')

        # set pixel state parameters
        if self.from_pixels or True:
            self.env.render()
            self.img_height = self.env.viewer.height
            self.img_width = self.env.viewer.width
            self.reduced_height = 84#self.img_height//self.frame_downscaling
            self.reduced_width = 84#self.img_width//self.frame_downscaling

        # simultaneous evaluation through neural network
        self.qnn = qnn.qnn(self.statedim,
                           self.num_actions,
                           discount=self.gamma,
                           size_hidden=nn_size_hidden,
                           batch_size=nn_batch_size,
                           learning_rate=nn_learning_rate,
                           is_a_prime_external=self.is_a_prime_external,
                           replay_memory_size=replay_memory_size,
                           descent_method=descent_method,
                           keep_prob_val=dropout_keep_prob,
                           ema_decay_rate=ema_decay_rate,
                           normalization_mean=normalization_mean,
                           normalization_var=normalization_var,
                           env_name=environment,
                           init_weights=init_weights,
                           from_pixels=self.from_pixels,
                           input_width=self.reduced_width,
                           input_height=self.reduced_height,
                           input_channels=self.repeat_action_times
                           )
        self.learning_rate = nn_learning_rate
        self.train_frequency = train_frequency

        print('using environment', environment)
        print('qnn target', qnn_target, self.is_a_prime_external, self.qnn.is_a_prime_external)



    # epsilon-greedy but deterministic or stochastic is a choice
    def policy(self, state, mode='deterministic', deepQ=False):

        explore = bool(np.random.choice([1, 0], p=[self.epsilon, 1 - self.epsilon]))

        # print(explore, features, end="")
        if mode == 'deterministic' and not explore:
            if deepQ:
                q = self.qnn.evaluate_all_actions(state)
                # print(state, q)
                return np.argmax(q.squeeze())#np.random.choice(np.argwhere(q == np.amax(q)).flatten())
            if not deepQ:
                raise ValueError('Option not defined')
                # q = features.dot(w)
                # return np.random.choice(np.argwhere(q == np.amax(q)).flatten())
        elif explore:
            # print('explore')
            return self.env.action_space.sample()

    def get_render(self):
        return np.asarray(\
                    ImageOps.flip(\
                        self.env.render('rgb_array')\
                          .convert('L')\
                              .resize((self.reduced_width, self.reduced_height), \
                                      Image.BILINEAR)))
    def get_cnn_input_tensor_from_deque(self, pixel_state_deque):
        return np.swapaxes(\
                    np.swapaxes(\
                        np.array(pixel_state_deque, ndmin=4),1,2),2,3)


    def deepq_learning(self, num_iter=1000, max_steps=5000, max_learning_steps=np.inf, learning_rate=None, reset_replay_memory=False):
        if learning_rate is None:
            learning_rate = self.learning_rate

        if reset_replay_memory:
            self.qnn.replay_memory.clear()

        # Show initial state, since algorithm is highly biased by the initial conditions
        if self.statedim == 2:
            # print('last w', self.w)
            self.plot_deepQ_policy(mode='deterministic')
            self.plot_deepQ_function()

        prev_writeout = self.qnn.samples_count
        prev_writeout_1 = self.qnn.samples_count
        ref_learning_steps = self.qnn.training_steps_count
        is_first = True
        for it in range(num_iter):
            if (self.qnn.training_steps_count - ref_learning_steps) > max_learning_steps:
                print('done training, I\'m tired. I started by', ref_learning_steps)
                break
            self.total_train_episodes += 1

            #            episode = []
            prev_state = self.env.reset()

            count = 0
            done = False
            # running list of the last pixel states
            pixel_state = deque(maxlen=self.repeat_action_times)
            # fill initially
            for _ in range(self.repeat_action_times):
                pixel_state.append(self.get_render())
            state_tensor = self.get_cnn_input_tensor_from_deque(pixel_state)

            # choose first action
            if not self.from_pixels:
                prev_action = self.policy(np.array(prev_state).reshape((1,-1)),
                                          mode=self.policy_mode,
                                          deepQ=True)
            if self.from_pixels:
                prev_action = self.policy(state_tensor,
                                          mode=self.policy_mode,
                                          deepQ=True)
            # run episode
            while (not done):
                state_prev_tensor = state_tensor
                pixel_state_prev = copy.copy(pixel_state) # shallow copy
                if count > max_steps:
                    self.episode_lengths.append(count)
                    break

                for _ in range(self.repeat_action_times):
                    count += 1
                    state, reward, done, info = self.env.step(prev_action)
                    if self.from_pixels:
                        pixel_state.append(self.get_render())
                    if done: break

                state_tensor = self.get_cnn_input_tensor_from_deque(pixel_state)

                if not self.from_pixels:
                    action = self.policy(np.array(state).reshape((1,-1)), mode=self.policy_mode, deepQ=True)
                if self.from_pixels:
                    action = self.policy(state_tensor,
                                         mode=self.policy_mode,
                                         deepQ=True)
                    # action = self.policy(state, mode=self.policy_mode, deepQ=True)
                #                episode.append((state, action, reward))

                # evaluation alone, to test a neural network
                if not self.is_a_prime_external:
                    # Q learning
                    if not self.from_pixels:
                        self.qnn.train_batch(prev_state.reshape(1,-1),
                                             np.array(prev_action).reshape(-1),
                                             np.array(reward).reshape(-1),
                                             state.reshape(1,-1),
                                             done,
                                             learning_rate=learning_rate,
                                             train_frequency=self.train_frequency)
                    if self.from_pixels:
                        self.qnn.train_batch(state_prev_tensor,
                                             np.array(prev_action).reshape(-1),
                                             np.array(reward).reshape(-1),
                                             state_tensor,
                                             done,
                                             learning_rate=learning_rate,
                                             train_frequency=self.train_frequency)
                        # self.qnn.train_batch()
                else:
                    # SARSA (not converging)
                    raise ValueError('Option not defined')
                    # self.qnn.train_batch(prev_state.reshape(1,-1), np.array(prev_action).reshape(-1), np.array(reward).reshape(-1), state.reshape(1,-1), np.array(action).reshape(-1))

                prev_state = state
                prev_action = action

                if (done):
                    self.episode_lengths.append(count)

                # decrease exploration
                if self.update_epsilon:
                    if self.epsilon > 0.1:
                        self.epsilon -= (self.init_epsilon - self.end_epsilon)*(1./self.exploration_decrease_length)

            # if (it + 1) % 5 == 0:
            if self.qnn.training_steps_count > 0 or is_first:
                if (self.qnn.samples_count - prev_writeout) > 1e4/self.train_frequency:
                    prev_writeout = self.qnn.samples_count
                    print("Episode %d" % (it), "total samples", self.qnn.samples_count, "train steps", self.qnn.training_steps_count)
                    if (done): print("Length %d" % (self.episode_lengths[-1]))

                # if (it + 1) % 100 == 0:
                if (self.qnn.samples_count - prev_writeout_1) > 1e5/self.train_frequency:
                    prev_writeout_1 = self.qnn.samples_count
                    print("exploration ", self.epsilon)
                    self.plot_training()

                    test_runs = [self.run_test_episode(limit=self.max_test_length) for _ in range(self.test_runs_to_average)]
                    self.test_lengths.append(np.mean(test_runs))
                    self.test_lengths_std.append(np.std(test_runs))
                    self.test_its.append(self.total_train_episodes)
                    self.plot_testing()

                    if self.statedim == 2:
                        # print('last w', self.w)
                        self.plot_deepQ_policy(mode='deterministic')
                        self.plot_deepQ_function()

                is_first = False

        self.plot_replay_memory_2d_state_histogramm()



    def run_test_episode(self, enable_render=False, limit=5000):
        save_epsilon = self.epsilon
        self.epsilon = 0.

        episode_length = 0.
        state = self.env.reset()

        # running list of the last pixel states
        pixel_state = deque(maxlen=self.repeat_action_times)
        # fill initially
        for _ in range(self.repeat_action_times):
            pixel_state.append(self.get_render())

        done = False
        while (not done):

            if episode_length > limit:
                self.epsilon = save_epsilon
                return episode_length



            if not self.from_pixels:
                action = self.policy(np.array(state).reshape((1,-1)), mode=self.policy_mode, deepQ=True)
            if self.from_pixels:
                action = self.policy(self.get_cnn_input_tensor_from_deque(pixel_state),
                                     mode=self.policy_mode,
                                     deepQ=True)

            for _ in range(self.repeat_action_times):
                episode_length += 1
                state, _, done, _ = self.env.step(action)
                if self.from_pixels:
                    pixel_state.append(self.get_render())
                if done: break

            if enable_render: self.env.render()
            # if count > self.max_episode_length: break;

        if enable_render: print("This episode took {} steps".format(count))

        self.epsilon = save_epsilon
        return episode_length



    def plot_deepQ_function(self):
        if self.from_pixels:
            print("plot test. May burn!!")

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
                if not self.from_pixels:
                    states.append((state1, state2))
                if self.from_pixels:
                    states.append(self.get_approx_pixel_state_from_state((state1,state2)).squeeze())

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

    def get_approx_pixel_state_from_state(self, state):
        """
        state should be a 1-D array
        """
        np_state = np.array(state)
        self.env.reset()
        self.env.state = np_state
        # print('---------------------------')
        # print(self.env.state)

        pixel_state = []
        pixel_state.append(self.get_render())
        for _ in range(self.repeat_action_times -1):
            state, reward, done, info = self.env.step(1)
            # print(state)
            pixel_state.append(self.get_render())
        # print('---------------------------')
        return self.get_cnn_input_tensor_from_deque(pixel_state)

    def plot_deepQ_policy(self, mode='deterministic'):
        if self.from_pixels:
            print("plot experiment. Watch out!")
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
                if not self.from_pixels:
                    greedy_policy[i, j] = self.policy(np.array((x, v)).reshape((1,-1)), mode, deepQ=True)
                if self.from_pixels:
                    greedy_policy[i, j] = self.policy(self.get_approx_pixel_state_from_state((x,v)), mode, deepQ=True)
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
            if len(self.episode_lengths) > 1000:
                plt.plot(np.arange(len(self.episode_lengths))[range(0,len(self.episode_lengths),10)],
                         np.array(self.episode_lengths)[range(0,len(self.episode_lengths),10)],
                         '.', linewidth=0)
            else:
                plt.plot(self.episode_lengths, '.', linewidth=0)
            plt.yscale('log')
            plt.xlabel("episodes")
            plt.ylabel("timesteps")
            plt.show()

    def plot_testing(self):

        if any(np.array(self.test_lengths > 0).flatten()):
            fig = plt.figure()
            if len(self.test_lengths) > 1000:
                # plt.plot(np.convolve(self.test_lengths, np.ones(10)/10, mode='same'), '.', linewidth=0)
                plt.plot(np.convolve(self.test_lengths, np.ones(10)/10, mode='same'), '.', linewidth=0)
            else:
                # plt.plot(self.test_its, self.test_lengths, '.', linewidth=0)
                plt.errorbar(self.test_its, self.test_lengths, yerr=self.test_lengths_std, fmt='.')#, linewidth=0)
            plt.yscale('log')
            plt.xlabel("test episodes")
            plt.ylabel("timesteps")
            plt.show()

    def plot_replay_memory_2d_state_histogramm(self):
        if self.from_pixels:
            print("plot not available. Move on.")
            return
        if self.statedim == 2:
            rm=np.array(self.qnn.replay_memory)
            states, _,_,_,_,_ = zip(*rm)
            states_np = np.array(states)
            states_np = np.squeeze(states_np)

            x,v = zip(*states_np)
            plt.hist2d(x, v, bins=40, norm=LogNorm())
            plt.xlabel("position")
            plt.ylabel("velocity")
            plt.colorbar()
            plt.show()
