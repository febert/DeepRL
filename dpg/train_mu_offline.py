from __future__ import print_function
import moutaincar_dpg

import numpy as np

np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import time
import math

import cPickle


from nn import nn

class mu_offline_training():

    def __init__(self):

        self.num_sgd_updates = 100000

        # create neural network:
        self.nn1 = nn()
        self.nn1.main()

        self.car1 = moutaincar_dpg.mountaincar_dpg()
        self.car1.loaddata('dpg_mountain_car_iter250')
        self.car1.plot_policy(mode= 'deterministic')

        self.theta = self.car1.theta

    def start_training(self):
        obs_low = self.car1.env.observation_space.low
        obs_high = self.car1.env.observation_space.high

        for i in range(self.num_sgd_updates*self.nn1.batchsize):
            xs = np.array([ np.random.uniform(obs_low[0], obs_high[0]),
                            np.random.uniform(obs_low[1], obs_high[1]) ]).squeeze()

            self.nn1.add_to_batch(state= xs, mu= self.car1.mu(xs))

if __name__ == '__main__':

    t1 = mu_offline_training()
    t1.start_training()