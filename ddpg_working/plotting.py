import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

def plot_replay_memory_2d_state_histogramm(states):
    x,v = zip(*states)
    plt.hist2d(x, v, bins=40, norm=LogNorm())
    plt.xlabel("position")
    plt.ylabel("velocity")
    plt.colorbar()
    plt.show()


def plot_learned_mu(eval_mu, env):

    print('plotting the mu() policy learned by NN')

    obs_low = env.observation_space.low
    obs_high = env.observation_space.high

    resolution = 20
    x_range = np.linspace(obs_low[0], obs_high[0], resolution)
    v_range = np.linspace(obs_low[1], obs_high[1], resolution)

    # get actions in a grid
    vals = np.zeros((resolution, resolution))
    for i, x in enumerate(x_range):
        for j, v in enumerate(v_range):
            x_ = np.array([x,v],dtype=np.float32).reshape((1,2))
            vals[j,i]= eval_mu(x_)[0]


    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x_range, v_range)
    ax.plot_surface(X, Y, vals, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_zlabel("action")
    plt.show()

def plot_episode_lengths(data):

        fig = plt.figure()

        plt.plot(data)

        plt.yscale('log')

        plt.xlabel("episodes")
        plt.ylabel("timesteps")

        plt.show()