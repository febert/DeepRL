from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS

import numpy as np

np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import time
import math
import cPickle
import gym as gym

from collections import deque
from ornstein_uhlenbeck import ornstein_uhlenbeck

from matplotlib.colors import LogNorm


class ddpg():

    def __init__(self,
                 learning_rates= (5e-5, 5e-4),
                 # environment = 'MountainCarContinuous-v0',
                 # environment = 'Reacher-v1',
                 environment = 'InvertedPendulum-v1',
                 noise_scale = 1,
                 enable_plotting = True,
                 ql2 = 0.01,
                 tensorboard_logs = True,
                 maxstep = 1e4,
                 warmup = 5e4
                 ):
        self.maxstep = maxstep
        self.ql2 = ql2
        # lr_actor = 5e-5
        # lr_critic = 5e-4   #according to Silver dpg-paper the critic should be faster !!
        self.lr_actor = learning_rates[0]
        self.lr_critic = learning_rates[1]

        self.gamma = 0.99

        self.enable_plotting = enable_plotting

        # start tf session
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(
            inter_op_parallelism_threads=4,
            log_device_placement=False,
            allow_soft_placement=True))

        self.l1 = 400  #neurons layer 1
        self.l2 = 300  #neurons layer 2

        self.step = 0

        self.summaries_dir = './logging/ddpg'

        replay_memory_size = 5e5 #number of transitions to be stored in replay buffer
        self.warmup = warmup

        self.train_lengths = []
        self.test_lengths = []

        self.replay_memory = deque(maxlen=replay_memory_size)

        # environment specific:
        self.env = gym.make(environment)
        self.select_env = environment


        self.num_outputs = 1
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

        print('state dim', self.state_dim)
        print('action dim', self.action_dim)
        self.ou_process = ornstein_uhlenbeck(ndim= self.action_dim, theta= 0.15, sigma= .2, delta_t= 1)

        self.noise_scale = noise_scale

        self.batch_size = 32

        self.samples_count = 0

        self.obs_low = self.env.observation_space.low
        self.obs_high = self.env.observation_space.high
        self.tensorboard_logs = tensorboard_logs

    def __enter__(self):
        return self

    def __del__(self):
        print( "deling", self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('closing the interactive session')
        tf.InteractiveSession.close(self.sess)
        tf.reset_default_graph()

    def rescale_action(self, action):
        actionmean = (self.env.action_space.low + self.env.action_space.high)/2
        ascale = (self.env.action_space.high- self.env.action_space.low)/2

        return action*ascale + actionmean

    def run_episode(self, enable_render=False, limit=5000, test_run = True):

        state = self.env.reset()
        self.ou_process.reset()

        accum_reward = 0
        done = False
        step_count = 0

        while ( not done ):

            if step_count>limit:
                return accum_reward
            step_count += 1


            state = np.squeeze(state).reshape((1,self.state_dim))  # convert (2,1) array in to (2,1)

            if test_run:
                action_raw = self.eval_mu(state)
            else:
                action_raw = self.eval_mu(state) + self.ou_process.ou_step()*self.noise_scale

            action = self.rescale_action(action_raw)
            action = self.apply_limits(action)
            action = action.squeeze()

            state_prime, reward, done, _ = self.env.step(action)

            accum_reward += reward

            if not test_run:
                self.replay_memory.append((state, action_raw, reward, state_prime, done))
                self.samples_count += 1

                if (len(self.replay_memory) > self.warmup) and (self.samples_count % (self.batch_size/2) == 0):
                    self.train_networks()

            state = state_prime

            if enable_render:
                self.env.render()


        return accum_reward


    def start_training(self, dataname ='unnamed_data', save = False):  #maxstep 5e4

        it = 0
        while self.step < self.maxstep:
            # if it % 1000 == 0:
            #     print('replay size',len(self.replay_memory))

            # run episode
            episode_length = self.run_episode(test_run= False, enable_render=False, limit= 10000)

            self.train_lengths.append(episode_length)

            # if it% 5== 0:
            #     self.plot_learned_mu()
            #     self.plot_replay_memory_2d_state_histogramm()
            if self.select_env == 'Reacher-v1':
                test_freq = 15
                plot_freq = 1

            if self.select_env == 'MountainCarContinuous-v0' or 'AcrobotContinuous-v0':
                test_freq = 5
                plot_freq = 10

            if self.select_env == 'InvertedPendulum-v1':
                test_freq = 100
                plot_freq = 1000

            if (it+1) % test_freq == 0:
                # perform a test run with the target policy:
                self.test_lengths.append(self.run_episode(test_run= True, enable_render=False))

            if (it+1) % plot_freq == 0:
                self.plot_episode_lengths(self.train_lengths)
                self.plot_episode_lengths(self.test_lengths)

            it+=1


        self.test_lengths.sort()
        print('sorted test lengths', self.test_lengths)
        best_10_percent = self.test_lengths[-int(len(self.test_lengths)*0.1):]
        mean_over_best_10_percent = np.mean(best_10_percent)

        return mean_over_best_10_percent


    def hist_summaries(self,*args):
        return tf.merge_summary([tf.histogram_summary(t.name, t) for t in args])

    def fanin_init(self, shape, fanin=None):
        if fanin != None:
            return tf.constant(0.,shape= shape)

        fanin = fanin or shape[0]
        v = 1 / np.sqrt(fanin)  # * np.sqrt(2.)

        # tf.random_uniform(shape, minval=-v, maxval=v)
        return tf.truncated_normal(shape, v)

    def create_theta_p(self,dimO, dimA):
        with tf.variable_scope("theta_p"):
            return [tf.Variable(self.fanin_init([dimO, self.l1]), name='1w'),
                    tf.Variable(self.fanin_init([self.l1], dimO), name='1b'),
                    tf.Variable(self.fanin_init([self.l1, self.l2]), name='2w'),
                    tf.Variable(self.fanin_init([self.l2], self.l1), name='2b'),
                    tf.Variable(tf.random_uniform([self.l2, dimA], 0, 0), name='3w'),
                    # tf.Variable(tf.random_uniform([self.l2, dimA], -3e-3, 3e-3), name='3w'),
                    # tf.Variable(tf.random_uniform([dimA], -3e-3, 3e-3), name='3b'),
                    tf.Variable(tf.random_uniform([dimA], 0, 0), name='3b'),
                    ]

    def mu_net(self, obs, theta, name='policy'):
        with tf.variable_op_scope([obs], name, name):
            h0 = tf.identity(obs, name='h0-obs')
            h1 = tf.nn.relu(tf.matmul(h0, theta[0]) + theta[1], name='h1')
            h2 = tf.nn.relu(tf.matmul(h1, theta[2]) + theta[3], name='h2')
            h3 = tf.identity(tf.matmul(h2, theta[4]) + theta[5], name='h3')
            action = tf.nn.tanh(h3, name='h4-action')

            summary = self.hist_summaries(h0, h1, h2, h3, action)
            return action, summary


    def create_theta_q(self,dimO, dimA):
        with tf.variable_scope("theta_q"):
            return [tf.Variable(self.fanin_init([dimO, self.l1]), name='1w'),
                    tf.Variable(self.fanin_init([self.l1], dimO), name='1b'),
                    tf.Variable(self.fanin_init([self.l1 + dimA, self.l2]), name='2w'),
                    tf.Variable(self.fanin_init([self.l2], self.l1 + dimA), name='2b'),
                    # tf.Variable(tf.random_uniform([self.l2, 1], -3e-4, 3e-4), name='3w'),
                    tf.Variable(tf.random_uniform([self.l2, 1], 0, 0), name='3w'),
                    # tf.Variable(tf.random_uniform([1], -3e-4, 3e-4), name='3b'),
                    tf.Variable(tf.random_uniform([1], 0, 0), name='3b')]


    def q_net(self,obs, act, theta, name="qfunction"):
        with tf.variable_op_scope([obs, act], name, name):
            h0 = tf.identity(obs, name='h0-obs')
            h1 = tf.nn.relu(tf.matmul(h0, theta[0]) + theta[1], name='h1')
            h1a = tf.concat(1, [h1, act])
            h2 = tf.nn.relu(tf.matmul(h1a, theta[2]) + theta[3], name='h2')
            qs = tf.matmul(h2, theta[4]) + theta[5]
            q = tf.squeeze(qs, [1], name='h3-q')

            summary = self.hist_summaries(h0, h1, h2, q)
            return q, summary

    def exponential_moving_averages(self, theta, tau=0.001):
        ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
        update = ema.apply(theta)  # also creates shadow vars
        averages = [ema.average(x) for x in theta]
        return averages, update

    def initialize_training(self, sess):

        self.theta_mu = self.create_theta_p(self.state_dim, self.action_dim)
        self.theta_q = self.create_theta_q(self.state_dim, self.action_dim)
        self.theta_mu_prime, update_mu_averages = self.exponential_moving_averages(self.theta_mu, 0.001)
        self.theta_q_prime, update_q_averages = self.exponential_moving_averages(self.theta_q, 0.001)

        self.state_raw = tf.placeholder(tf.float32, [None, self.state_dim], name='x-states')

        if self.select_env == 'MountainCarContinuous-v0':
            # mean = tf.constant(self.normalization_mean, name='batch_mean')
            # variance = tf.constant(self.normalization_var, name='batch_variance')
            mean = tf.constant([ -5.16865671e-01 ,  1.22739366e-05], name='batch_mean')
            variance = tf.constant([ 0.1141372,   0.00058848], name='batch_variance')
            # mean = tf.constant([-0.3, 0], name='batch_mean')
            # variance = tf.constant([0.81, 0.0049], name='batch_variance')


        if self.select_env == 'InvertedPendulum-v1':
            mean = tf.constant([0., 0., 0., 0.], name='batch_mean')
            variance = tf.constant([4.22701007e-03,  1.98672228e-02,   9.07489777e-01,   4.62239027e+00], name='batch_variance')
            # self.state = self.state_raw

        if self.select_env == 'Reacher-v1':
            mean = tf.constant([-0.005007  , -0.21291223, -0.00599161, -0.00261705,  0.13582967,
                    -0.02070659,  0.16570283, -0.00088109, -0.13647044,  0.02026401,  0.        ])
            variance = tf.constant([  4.98794347e-01,   5.26965082e-01,   5.01144707e-01, 4.27696466e-01,
                          5.27200673e-27,   2.31298216e-27, 1.22266579e+02,   5.15871811e+01,   8.68974719e-03,
                         8.72577634e-03,   0.00000000e+00])

        if self.select_env == 'AcrobotContinuous-v0':
            mean = tf.constant([-0.00020907, -0.00273588,  0.00071914, -0.01408309], name='batch_mean')
            variance = tf.constant([  0.79269701,   2.74056292,   3.55256128,  10.79969215], name='batch_variance')


        self.state = tf.nn.batch_normalization(self.state_raw, mean, variance, None, None, 1e-8)

        self.mu, sum_p = self.mu_net(self.state, self.theta_mu)

        self.learning_rate_actor = tf.placeholder(tf.float32, shape= [], name='actor_learning_rate')
        self.learning_rate_critic = tf.placeholder(tf.float32, shape= [], name='critic_learning_rate')

        # test
        self.q, sum_q = self.q_net(self.state, self.mu, self.theta_q, name='q_mu_of_s')
        # training
        # policy loss
        meanq = tf.reduce_mean(self.q)
        # wd_p = tf.add_n([pl2 * tf.nn.l2_loss(var) for var in self.theta_mu])  # weight decay
        self.mu_loss = -meanq #+ wd_p
        # policy optimization
        # optim_p = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_actor)
        optim_p = tf.train.AdamOptimizer(learning_rate=self.learning_rate_actor)
        grads_and_vars_p = optim_p.compute_gradients(self.mu_loss, var_list=self.theta_mu)
        optimize_p = optim_p.apply_gradients(grads_and_vars_p)
        with tf.control_dependencies([optimize_p]):
            self.mu_train_step = tf.group(update_mu_averages)

        # q optimization
        self.act_train = tf.placeholder(tf.float32, [None, self.action_dim], name='actions')
        self.rew = tf.placeholder(tf.float32, [None, 1], name='rewards')
        self.state_prime_raw = tf.placeholder(tf.float32, [None, self.state_dim], name='states_prime')
        if self.select_env == "MountainCarContinuous-v0":
            state_prime = tf.nn.batch_normalization(self.state_prime_raw, mean, variance, None, None, 1e-8)
        else:
            state_prime = self.state_prime_raw
        self.term2 = tf.placeholder(tf.bool, [None, 1], name='term2')

        # q
        self.q_train, sum_qq = self.q_net(self.state, self.act_train, self.theta_q, name= 'qs_a')

        # q targets
        act2, sum_p2 = self.mu_net(state_prime, theta=self.theta_mu_prime)
        self.q2, sum_q2 = self.q_net(state_prime, act2, theta=self.theta_q_prime, name= 'qsprime_aprime')
        q_target = tf.stop_gradient(tf.select(self.term2, self.rew, self.rew + self.gamma * tf.reshape(self.q2, [self.batch_size,1]) ))

        # q loss
        td_error = self.q_train - q_target
        ms_td_error = tf.reduce_mean(tf.square(td_error))
        wd_q = tf.add_n([self.ql2 * tf.nn.l2_loss(var) for var in self.theta_q])  # weight decay
        self.q_loss = ms_td_error   + wd_q
        # q optimization
        # optim_q = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_critic)
        optim_q = tf.train.AdamOptimizer(learning_rate=self.learning_rate_critic)
        grads_and_vars_q = optim_q.compute_gradients(self.q_loss, var_list=self.theta_q)
        optimize_q = optim_q.apply_gradients(grads_and_vars_q)
        with tf.control_dependencies([optimize_q]):
            self.q_train_step = tf.group(update_q_averages)


        # logging
        log_obs = [] if self.state_dim > 20 else [tf.histogram_summary("obs/" + str(i), self.state_raw[:, i]) for i in range(self.state_dim)]
        log_act = [] if self.action_dim > 20 else [tf.histogram_summary("act/inf" + str(i), self.mu[:, i]) for i in
                                           range(self.action_dim)]
        log_act2 = [] if self.action_dim > 20 else [tf.histogram_summary("act/train" + str(i), self.act_train[:, i]) for i in
                                            range(self.action_dim)]
        log_misc = [sum_p, sum_qq, tf.histogram_summary("td_error", td_error)]
        log_grad = [self.grad_histograms(grads_and_vars_p), self.grad_histograms(grads_and_vars_q)]
        log_train = log_obs + log_act + log_act2 + log_misc + log_grad

        tf.scalar_summary('mean squared tderror',ms_td_error )
        tf.scalar_summary('qloss',self.mu_loss )


        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        self.merged = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(self.summaries_dir, sess.graph)
        tf.initialize_all_variables().run()

    def get_train_batch(self):

        #selecting transitions randomly from the replay memory:
        indices =  np.random.randint(0, len(self.replay_memory), [self.batch_size])
        transition_batch = [self.replay_memory[i] for i in indices]

        states = np.asarray([transition_batch[i][0].squeeze() for i in range(self.batch_size)])
        actions = np.asarray([transition_batch[i][1] for i in range(self.batch_size)])
        rewards = np.asarray([transition_batch[i][2] for i in range(self.batch_size)])
        states_prime = np.asarray([transition_batch[i][3].squeeze() for i in range(self.batch_size)])
        term2 = np.asarray([transition_batch[i][4] for i in range(self.batch_size)])

        return states, actions, rewards, states_prime, term2

    def train_networks(self):


        def feed_dict():

            states, action, reward, states_prime, term2 = self.get_train_batch()

            return {self.state_raw: states.squeeze().reshape(self.batch_size, self.state_dim),
                    self.act_train: action.squeeze().reshape(self.batch_size,self.action_dim),
                    self.rew: reward.squeeze().reshape(self.batch_size,1),
                    self.state_prime_raw: states_prime.squeeze().reshape(self.batch_size, self.state_dim),
                    self.term2: term2.squeeze().reshape(self.batch_size,1),
                    self.learning_rate_actor: self.lr_actor,
                    self.learning_rate_critic: self.lr_critic,
                    }

        dict_ = feed_dict()

        if self.tensorboard_logs:
            summary, _, _, mse_val = self.sess.run([self.merged, self.q_train_step, self.mu_train_step, self.q_loss], feed_dict= dict_)
            if self.step % 10 == 0:
                self.train_writer.add_summary(summary, self.step)
        else:
            _, _, mse_val = self.sess.run([self.q_train_step, self.mu_train_step, self.q_loss], feed_dict= dict_)


        if self.step % 1000 == 0:
            print('result after minibatch no. {} : mean squared error: {}'.format(self.step, mse_val))
            # print('batch train data states', dict_[self.x_states])
            # print('batch train data actions', dict_[self.x_action])
            if self.select_env == 'MountainCarContinuous-v0':
                self.plot_learned_mu()
                self.plot_replay_memory_2d_state_histogramm()
                self.plot_q_func()

            # print('qs: ', self.q.eval(feed_dict = {self.x_states: dict_[self.x_states], self.x_action: dict_[self.x_action]}))

            # print('batch train data targets', dict_[self.q_targets])

        self.step += 1

    def eval_mu(self, state):
        return self.mu.eval(feed_dict= {self.state_raw : state})

    def eval_q(self, state, action):
        return self.q_train.eval(feed_dict= {self.state_raw : state, self.act_train : action})


    def apply_limits(self,action):
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)


    def plot_episode_lengths(self, lengths):
        if self.enable_plotting:
            fig = plt.figure()

            plt.plot(np.abs(np.asarray(lengths)))

            plt.yscale('log')

            plt.xlabel("episodes")
            plt.ylabel("timesteps")

            plt.show()

    def main(self):
        if self.tensorboard_logs:
            if tf.gfile.Exists(self.summaries_dir):
                tf.gfile.DeleteRecursively(self.summaries_dir)
            tf.gfile.MakeDirs(self.summaries_dir)

        self.initialize_training(self.sess)
        return self.start_training()

    def plot_replay_memory_2d_state_histogramm(self):

        if self.enable_plotting:
            if self.state_dim == 2:
                rm=np.array(self.replay_memory)
                states, _,_,_,_ = zip(*rm)
                states_np = np.array(states)
                states_np = np.squeeze(states_np)

                x,v = zip(*states_np)
                plt.hist2d(x, v, bins=40, norm=LogNorm())
                plt.xlabel("position")
                plt.ylabel("velocity")
                plt.colorbar()
                plt.show()

    def plot_q_func(self):

        if self.enable_plotting:
            print('plotting the Qfunction')
            for action in np.linspace(self.env.action_space.low[0],self.env.action_space.high[0],num=5):
                print('action {}'.format(action))
                resolution = 20
                x_range = np.linspace(self.obs_low[0], self.obs_high[0], resolution)
                v_range = np.linspace(self.obs_low[1], self.obs_high[1], resolution)

                # get actions in a grid
                vals = np.zeros((resolution, resolution))
                for i, x in enumerate(x_range):
                    for j, v in enumerate(v_range):
                        x_ = np.array([x,v],dtype=np.float32).reshape((1,2))
                        action = action.reshape((1,1))
                        vals[j,i]= self.eval_q(x_, action)[0]

                fig = plt.figure()

                ax = fig.add_subplot(111)
                X, Y = np.meshgrid(x_range, v_range)
                im = ax.pcolormesh(X, Y, vals)
                fig.colorbar(im)
                ax.set_xlabel("x")
                ax.set_ylabel("v")
                plt.show()

    def plot_learned_mu(self):
        if self.enable_plotting:

            print('plotting the mu() policy learned by NN')

            resolution = 20
            x_range = np.linspace(self.obs_low[0], self.obs_high[0], resolution)
            v_range = np.linspace(self.obs_low[1], self.obs_high[1], resolution)

            # get actions in a grid
            vals = np.zeros((resolution, resolution))
            for i, x in enumerate(x_range):
                for j, v in enumerate(v_range):
                    x_ = np.array([x,v],dtype=np.float32).reshape((1,2))
                    vals[j,i]= self.eval_mu(x_)[0]

            # print('muvals', vals)

            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(x_range, v_range)
            ax.plot_surface(X, Y, vals, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=True)
            ax.set_xlabel("x")
            ax.set_ylabel("v")
            ax.set_zlabel("action")
            plt.show()

    def grad_histograms(self,grads_and_vars):
        s = []
        for grad, var in grads_and_vars:
            s.append(tf.histogram_summary(var.op.name + '', var))
            s.append(tf.histogram_summary(var.op.name + '/gradients', grad))
        return tf.merge_summary(s)

if __name__ == '__main__':

    car = ddpg(warmup= 1, maxstep= 10)
    car.main()
