import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

class ornstein_uhlenbeck():

    def __init__(self, ndim, theta, sigma, delta_t):
        self.x = np.zeros(ndim)
        self.theta = theta
        self.delta_t = delta_t
        self.sigma = sigma
        self.ndim = ndim

    def reset(self):
        self.x = np.zeros_like(self.x)

    def ou_step(self):

        epsilon = np.random.randn(self.ndim)
        # print('epsilon:', epsilon)
        # print('x:', self.x)
        self.x += -self.theta*self.x*self.delta_t + self.sigma*np.sqrt(self.delta_t)*epsilon

        return  self.x


if __name__ == '__main__':

        ou1 = ornstein_uhlenbeck(ndim= 1, theta= 0.15, sigma= .3, delta_t= 1)

        nsample = 10000
        ou1vals = np.zeros(nsample)

        for i in range(nsample):
            ou1vals[i]= ou1.ou_step()

        fig = plt.figure()
        plt.plot(ou1vals)
        plt.show()

