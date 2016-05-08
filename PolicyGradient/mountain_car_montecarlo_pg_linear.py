from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time

import gym as gym

class mountain_car():

    def __init__(self,
                 alpha=0.0001,
                 num_features=2,
                 init_epsilon=1.0,
                 update_epsilon=True,
                 policy_mode='stochastic',
                 N_0=50.):
        self.env = gym.make('MountainCar-v0')
        self.num_actions = 3
        self.num_features = num_features
        # learning rate
        self.alpha = alpha

        # lengths of all the played episodes
        self.episode_lengths = []

        # lengths of all the tested episodes TODO
        self.test_lengths = []

        # policy parameters
            # random initialization
        # self.theta = np.random.randn(self.num_features, self.num_actions)*0.01
            # zero initialization
        self.theta = np.zeros((self.num_features, self.num_actions))
            # stochastic or deterministic softmax-based actions
        self.policy_mode = policy_mode
            # exploration parameters
            # too much exploration is wrong!!!
        self.epsilon = init_epsilon    # explore probability
        self.update_epsilon = update_epsilon
        self.total_runs = 0
        # too long episodes give too much negative reward!!!!
        # self.max_episode_length = 1000000
        # ----> Use gamma!!!!! TODO: slower decrease?
        self.gamma = 0.99  #similar to 0.9

        self.N_0 = N_0

        np.seterr(all='raise')

    def get_features(self, state):
        normalizer = np.abs(self.env.observation_space.high - self.env.observation_space.low)
        if self.num_features == 2:
            return np.array([state[0]/normalizer[0],
                             state[1]/normalizer[1]])
        if self.num_features == 3:
            return np.array([1,
                             state[0]/normalizer[0],
                             state[1]/normalizer[1]])
        if self.num_features == 4:
            return np.array([state[0]/normalizer[0],
                             state[1]/normalizer[1],
                             (state[0]/normalizer[0])**2,
                             (state[1]/normalizer[1])**2])
        if self.num_features == 5:
            return np.array([1,
                             state[0]/normalizer[0],
                             state[1]/normalizer[1],
                             (state[0]/normalizer[0])**2,
                             (state[1]/normalizer[1])**2])

    # TODO: better solution to over-/underflows? Maybe not needed if proper learning?
    def eval_action_softmax_probabilities(self, state):
        features = self.get_features(state)
        #softmax probability
        activation = features.dot(self.theta)

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
                print(".", end="")

                prob_distrib[np.log(prob_distrib)<-100] =0        # print("activation ", activation)

                # while np.min(activation) < 200:
                #     activation += 10
                # while np.max(activation) > 200:
                #     activation -= 10
                # while np.max(activation) - 60 > np.min(activation):
                #     activation[activation!=np.max(activation)] += 10
                # while np.max(activation) - np.min(activation) > 200:
                #     activation[np.argmin(activation)] += 10
                #     activation[np.argmax(activation)] -= 10

        # prob_distrib[np.log(prob_distrib)<-100] =0        # print("activation ", activation)

        return prob_distrib

    #epsilon-greedy but deterministic or stochastic is a choice
    def policy(self, state, mode='deterministic'):
        explore = bool(np.random.choice([1,0],p=[self.epsilon, 1-self.epsilon]))

        features = self.get_features(state)
        # print(explore, features, end="")
        if mode=='deterministic' and not explore:
            # print('deterministic')
            return np.argmax(features.dot(self.theta))
        elif mode=='stochastic' and not explore:
            prob_distrib = self.eval_action_softmax_probabilities(state)
            # print('stochastic', prob_distrib)
            return np.random.choice(np.arange(self.num_actions),p=prob_distrib)
        elif explore:
            # print('explore')
            return self.env.action_space.sample()

    def run_episode(self, enable_render=False):
        episode = []
        state = self.env.reset()

        count = 0
        done = False
        while ( not done ):

            if self.policy_mode== "deterministic":
                if len(episode)>1000000: break

            count += 1

            action = self.policy(state, mode=self.policy_mode)
            state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
            if enable_render: self.env.render()
            # if count > self.max_episode_length: break;

        if enable_render: print("This episode took {} steps".format(count))

        return episode


    def score_function(self, state, action):
        features = self.get_features(state)

        prob_distrib = self.eval_action_softmax_probabilities(state)

        try:
            score = -features[:,None]*prob_distrib[None,:]
        except FloatingPointError, error:
            print(error)
            print("features", features)
            print("prob_distrib", prob_distrib)

        score[:,action] += features

        return score


    def train(self, iter=1000):
        # fig = plt.figure()

        for it in range(iter):
            # run episode
            episode = self.run_episode()
            # keep track of training episode lengths
            self.episode_lengths.append(len(episode))

            if (it+1)%10 == 0:
                # output training info
                print("EPISODE #{}".format(it+1))
                print("with a exploration of {}%".format(self.epsilon*100))
                print("lasted {0} steps".format(len(episode)))

                # do a test run
                save_policy_mode = self.policy_mode
                print(self.policy_mode)

                self.policy_mode = "deterministic"
                det_episode = self.run_episode()

                self.test_lengths.append(len(det_episode))
                self.policy_mode = save_policy_mode

            # theta += alpha*calculate_gradient()
            # Version reversed (equivalent if only updated at the end)
            theta_update = 0
            value_fcn = 0
            for idx in range(len(episode),0,-1):
                (state, action, reward) = episode[idx-1]
                value_fcn = reward + self.gamma*value_fcn
                theta_update += self.score_function(state,action)*value_fcn
                # TODO: check sign in this line. HAS IT TO BE -. WHY??
            self.theta += self.alpha*theta_update
            ## Original version
            # value_fcn = np.zeros(len(episode))
            # value_fcn[-1] = episode[-1][2]
            # for i in range(len(episode)-1,0,-1):
            #     value_fcn[i-1] = self.gamma*value_fcn[i] + episode[i-1][2]         #value from next plus reward from step
            # # print(value_fcn)
            # theta_update = 0
            # for idx, (state, action, reward) in enumerate(episode):
            #     ## debug
            #     # if idx%1000 == 0:
            #     #     print("THETA update", idx)
            #     #     print("state", state)
            #     #     print("theta", self.theta)
            #     #     print("score ", self.score_function(state,action))
            #     #     print("value fcn ", value_fcn[idx])
            #     #     features = self.get_features(state)
            #     #     prob_distrib = np.exp(features.dot(self.theta))
            #     #     prob_distrib /= np.sum(prob_distrib)
            #     #     print("prob_distrib", prob_distrib)
            #     ## end debug
            #     try:
            #         theta_update += self.alpha*self.score_function(state,action)*value_fcn[idx]
            #         # self.theta += self.alpha*self.score_function(state,action)*value_fcn[idx]
            #     except FloatingPointError, error:
            #         print(error)
            #         print(self.alpha)
            #         print(self.score_function(state,action))
            #         print(value_fcn[idx])
            # self.theta += theta_update
            # print("max min theta", np.max(self.theta), np.min(self.theta))
            if (it+1)%10 == 0:
                print("theta")
                print(self.theta)
                self.plot_policy()
            if (it+1)%100 == 0:
                self.plot_training()
                self.plot_testing()
                # self.test_episode
            # if i % 10 == 0:
            #     plt.plot(episode_lengths)
            #     plt.show()

            # decrease exploration
            self.total_runs += 1
            if self.update_epsilon:
                self.epsilon = self.N_0 / (self.N_0 + self.total_runs)

        return self.theta


    def plot_policy(self, resolution=100):

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
                greedy_policy[i,j] = self.policy((x,v),'deterministic')
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
        plt.show()


    def plot_testing(self):

        fig = plt.figure()
        plt.plot(self.test_lengths)
        plt.show()
