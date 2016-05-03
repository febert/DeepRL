import numpy as np
import matplotlib.pyplot as plt
import time

import gym as gym

class mountain_car():

    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.num_actions = 3
        self.num_features = 3
        self.theta = np.random.randn(self.num_features, self.num_actions)*0.01
        self.alpha = 0.000001
        self.episode_lengths = []
        self.epsilon = 0.05     # explore probability

    def get_features(self, state):
        return np.array([1, state[0], state[1]])

    #epsilon-greedy but deterministic or stochastic is a choice
    def policy(self, state, mode='deterministic'):
        explore = bool(np.random.choice([1,0],p=[self.epsilon, 1-self.epsilon]))

        features = self.get_features(state)
        if mode=='deterministic' and not explore:
            return np.argmax(features.dot(self.theta))
        elif mode=='stochastic' and not explore:
            prob_distrib = np.exp(features.dot(self.theta))
            prob_distrib /= np.sum(prob_distrib)
            return np.random.choice(np.arange(self.num_actions),p=prob_distrib)
        elif explore:
            return self.env.action_space.sample()

    def run_episode(self, enable_render=False):
        print("NEW EPISODE")
        episode = []
        state = self.env.reset()

        done = False
        while ( not done ):
            action = self.policy(state, mode='stochastic')
            state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
            if enable_render: self.env.render()

        return episode

    def score_function(self, state, action):
        features = self.get_features(state)
        #softmax probability
        prob_distrib = np.exp(features.dot(self.theta))
        prob_distrib /= np.sum(prob_distrib)

        score = -features[:,None]*prob_distrib[None,:]
        score[:,action] += features

        return score


    def train(self, iter=1000):
        # fig = plt.figure()
        for i in range(iter):
            print(i)
            episode = self.run_episode()
            self.episode_lengths.append(len(episode))
            # theta += alpha*calculate_gradient()
            value_fcn = np.zeros(len(episode))
            value_fcn[-1] = episode[-1][2]
            for i in range(len(episode)-1,0,-1):
                value_fcn[i-1] = value_fcn[i] + episode[i-1][2]         #value from next plus reward from step
            for idx, (state, action, reward) in enumerate(episode):
                if idx%1000 == 0:
                    print("THETA update", idx)
                    print("score ", self.score_function(state,action))
                    print("value fcn ", value_fcn[idx])
                    print("theta", self.theta)
                    features = self.get_features(state)
                    prob_distrib = np.exp(features.dot(self.theta))
                    prob_distrib /= np.sum(prob_distrib)
                    print("prob_distrib", prob_distrib)
                self.theta += self.alpha*self.score_function(state,action)*value_fcn[idx]
                if idx%1000 == 0:
                    print("THETA update", idx)
                    print("score ", self.score_function(state,action))
                    print("value fcn ", value_fcn[idx])
                    print("theta", self.theta)
                    features = self.get_features(state)
                    prob_distrib = np.exp(features.dot(self.theta))
                    prob_distrib /= np.sum(prob_distrib)
                    print("prob_distrib", prob_distrib)
            print("theta", self.theta)
            # if i % 10 == 0:
            #     plt.plot(episode_lengths)
            #     plt.show()
        return self.theta
