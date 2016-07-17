import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from matplotlib.colors import LogNorm
import tensorflow as tf


def plot_episode_lengths( lengths):
    fig = plt.figure()
    plt.plot(lengths)
    plt.yscale('log')
    plt.xlabel("episodes")
    plt.ylabel("timesteps")
    plt.show()


def plot_replay_memory_2d_state_histogramm(replay_memory):
    rm=np.array(replay_memory)
    states, _,_,_,_ = zip(*rm)
    states_np = np.array(states)
    states_np = np.squeeze(states_np)

    x,v = zip(*states_np)
    plt.hist2d(x, v, bins=40, norm=LogNorm())
    plt.xlabel("position")
    plt.ylabel("velocity")
    plt.colorbar()
    plt.show()

def plot_q_func(q_func, env):
    print('plotting the Qfunction')
    obs_low = env.observation_space.low
    obs_high = env.observation_space.high
    action_limits = [env.action_space.low, env.action_space.high]

    for action in np.linspace(action_limits[0],action_limits[1],num=5):
        print('Qfunction for action {}'.format(action))
        resolution = 20
        x_range = np.linspace(obs_low[0], obs_high[0], resolution)
        v_range = np.linspace(obs_low[1], obs_high[1], resolution)

        # get actions in a grid
        vals = np.zeros((resolution, resolution))
        for i, x in enumerate(x_range):
            for j, v in enumerate(v_range):
                x_ = np.array([x,v],dtype=np.float32).reshape((1,2))
                action = action.reshape((1,1))
                vals[j,i]= q_func(x_, action)[0]

        fig = plt.figure()

        ax = fig.add_subplot(111)
        X, Y = np.meshgrid(x_range, v_range)
        im = ax.pcolormesh(X, Y, vals)
        fig.colorbar(im)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        plt.show()

def plot_learned_func(eval_func, env):
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
            x_ = np.array([x,v]).reshape((1,2))
            vals[j,i]= eval_func(x_)[0]

    # print('muvals', vals)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x_range, v_range)
    ax.plot_surface(X, Y, vals, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_zlabel("action")
    plt.show()


def grad_histograms(grads_and_vars):
    s = []
    for grad, var in grads_and_vars:
        s.append(tf.histogram_summary(var.op.name + '', var))
        s.append(tf.histogram_summary(var.op.name + '/gradients', grad))
    return tf.merge_summary(s)

def hist_summaries(*args):
    return tf.merge_summary([tf.histogram_summary(t.name, t) for t in args])

