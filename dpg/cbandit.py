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


class cbandit():


    def __init__(self,):

        self.n = 100 #number of parameters, in this case equal number of actions m

        self.theta = np.zeros(self.n)

        self.C = np.diag(np.random.rand(self.n)*0.9 + 0.1)

        self.sigma_b = 0.1 #behavior policy variance

        self.w = np.zeros(self.n)

        self.b = 0

        self.gamma = 1

        self.alpha_theta = 1e-3
        self.alpha_w = 1e-2
        self.alpha_v = 1e-2

        self.rewards = []

    def beta(self):
        # behavior policy
         return np.random.randn(self.n)*self.sigma_b + self.theta

    def mu(self):
        # target policy
        return self.theta

    def Qw(self,action):
        # calc Qfunction
        return self.w.dot(action - self.mu())  + self.b

    def rollout(self,action):

        astar = np.ones(self.n)*4
        return -(action - astar).dot(self.C).dot(action - astar)


    def train(self, nsteps=400):

        print('C',self.C)

        for i in range(nsteps):

            if (i+1)%1000 == 0:
                print('reward',r_t)
                print('theta  ', self.theta)
                print('w: ',self.w)
                print('bias :', self.b)

            #choose action from behavior policy:
            a_t= self.beta()

            #apply the action of behavior policy:
            r_t = self.rollout(a_t)
            self.rewards.append(r_t)


            delta_t = r_t - self.gamma * self.Qw(self.mu()) - self.Qw(a_t)

            self.theta += self.alpha_theta * self.w

            self.w += self.alpha_w * delta_t * (a_t - self.theta)

            self.b += self.alpha_v * delta_t


    def plot_training(self):

        fig = plt.figure()
        plt.plot(-np.asarray(self.rewards))
        plt.yscale('log')
        plt.xlabel("episodes")
        plt.ylabel("timesteps")
        plt.show()



np.meshgrid()
c1 = cbandit()

c1.train(nsteps=20000)


c1.plot_training()

