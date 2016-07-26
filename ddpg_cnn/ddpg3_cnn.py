from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import time

flags = tf.app.flags
FLAGS = flags.FLAGS

import numpy as np

np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt
from matplotlib import cm
import gym as gym

from collections import deque
from matplotlib.colors import LogNorm

from networks_cnn import *
from utils.ornstein_uhlenbeck import *

from PIL import Image
from PIL import ImageOps

import cProfile

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
                 maxstep = 1e5,
                 warmup = 10 #5e4
                 ):

        self.maxstep = maxstep
        self.ql2 = ql2

        self.lr_actor = learning_rates[0]
        self.lr_critic = learning_rates[1]

        self.gamma = 0.99

        self.enable_plotting = enable_plotting

        # start tf session
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(
            inter_op_parallelism_threads=4,
            log_device_placement=False,
            allow_soft_placement=True))

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
        self.dim_actions = self.env.action_space.shape[0]

        self.env.reset()
        print(self.env.render('rgb_array').shape)
        self.dimO = [64, 64]

        print('observation dim', self.dimO)
        print('action dim', self.dim_actions)

        self.ou_process = ornstein_uhlenbeck(ndim= self.dim_actions, theta= 0.15, sigma= .2, delta_t= 1)

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

    def getframe(self):
        if self.select_env == 'InvertedPendulum-v1':
            im = Image.fromarray(self.env.render('rgb_array'), 'RGB')
            return np.asarray(im.convert('L').resize((self.dimO[0], self.dimO[1]), Image.BILINEAR))/255.

        return np.asarray(ImageOps.flip(self.env.render('rgb_array').convert('L').resize((self.dimO[0], self.dimO[1]), Image.BILINEAR)))/255.

    def run_episode(self, enable_render=False, limit=5000, test_run = True):

        self.env.reset()
        state = self.getframe()
        print('state size :', state.shape)
        print(state)
        print(np.min(state), np.max(state))
        plt.imshow(state,cmap='gray', interpolation='none')
        plt.show()

        state = np.zeros((self.dimO[0], self.dimO[1], 3))
        state_prime = np.zeros((self.dimO[0], self.dimO[1], 3))
        for t in range(3):
            self.env.step(self.env.action_space.sample())
            state[...,t] = self.getframe()

        self.ou_process.reset()

        accum_reward = 0
        done = False
        step_count = 0

        while ( not done ):

            if step_count>limit:
                return accum_reward
            step_count += 1


            state = np.squeeze(state).reshape([1] +self.dimO + [3])  # convert (2,1) array in to (2,1)

            if test_run:
                action_raw = self.eval_mu(state)
            else:
                action_raw = self.eval_mu(state) + self.ou_process.ou_step()*self.noise_scale

            action = self.rescale_action(action_raw)
            action = self.apply_limits(action)
            action = action.squeeze()

            t_1 = time.clock()
            asdf = []
            for t in range(3):
                _ , reward, done, _ = self.env.step(action)
                # state_prime[...,t] = self.getframe()
                asdf.append(self.getframe())
            print('time to retrieve 3 pictures:', time.clock() - t_1)

            accum_reward += reward

            if not test_run:
                self.replay_memory.append((state, action_raw, reward, state_prime, done))
                self.samples_count += 1

                if (len(self.replay_memory) > self.warmup) and (self.samples_count % (self.batch_size/2) == 0):
                    t_2 = time.clock()
                    cProfile.runctx('self.train_networks()',globals(),locals())
                    print('time of sgd step:', time.clock() - t_2)


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

            t_start_episode = time.clock()
            episode_length = self.run_episode(test_run= False, enable_render=False, limit= 10000)
            print('episode no. {} lastet {} seconds'.format(it, time.clock()- t_start_episode))

            self.train_lengths.append(episode_length)

            # if it% 5== 0:
            #     self.plot_learned_mu()
            #     self.plot_replay_memory_2d_state_histogramm()
            if self.select_env == 'Reacher-v1':
                test_freq = 3
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


    def initialize_training(self, sess):

        cnn_conf = cnn_config(self.dimO)

        self.theta_mu = theta_mu(self.dim_actions, cnn_conf)
        self.theta_q = theta_q(self.dimO, self.dim_actions, cnn_conf)
        self.theta_mu_prime, update_mu_averages = exponential_moving_averages(self.theta_mu, 0.001)
        self.theta_q_prime, update_q_averages = exponential_moving_averages(self.theta_q, 0.001)

        self.obs = tf.placeholder(tf.float32, [None, self.dimO[0], self.dimO[1], 3], name='x-states')

        self.mu, sum_p = mu_net(self.obs, self.theta_mu, cnn_conf)

        self.learning_rate_actor = tf.placeholder(tf.float32, shape= [], name='actor_learning_rate')
        self.learning_rate_critic = tf.placeholder(tf.float32, shape= [], name='critic_learning_rate')

        # test
        self.q, sum_q = q_net(self.obs, self.mu, self.theta_q, cnn_conf, name='q_mu_of_s')
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
        self.act_train = tf.placeholder(tf.float32, [None, self.dim_actions], name='actions')
        self.rew = tf.placeholder(tf.float32, [None, 1], name='rewards')
        self.obs2 = tf.placeholder(tf.float32, [None, self.dimO[0], self.dimO[1], 3], name='states_prime')

        self.term2 = tf.placeholder(tf.bool, [None, 1], name='term2')

        # q
        self.q_train, sum_qq = q_net(self.obs, self.act_train, self.theta_q,cnn_conf, name='qs_a')

        # q targets
        act2, sum_p2 = mu_net(self.obs2, self.theta_mu_prime, cnn_conf,name='mu_of_sprime')
        self.q2, sum_q2 = q_net(self.obs2, act2, self.theta_q_prime, cnn_conf, name= 'qsprime_aprime')
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
        log_obs = [tf.histogram_summary("observations/" , self.obs)]
        log_act = [] if self.dim_actions > 20 else [tf.histogram_summary("act/inf" + str(i), self.mu[:, i]) for i in
                                                    range(self.dim_actions)]
        log_act2 = [] if self.dim_actions > 20 else [tf.histogram_summary("act/train" + str(i), self.act_train[:, i]) for i in
                                                     range(self.dim_actions)]
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

            return {self.obs: states.squeeze().reshape(self.batch_size, self.dimO[0], self.dimO[1],3),
                    self.act_train: action.squeeze().reshape(self.batch_size, self.dim_actions),
                    self.rew: reward.squeeze().reshape(self.batch_size,1),
                    self.obs2: states_prime.squeeze().reshape(self.batch_size, self.dimO[0], self.dimO[1],3),
                    self.term2: term2.squeeze().reshape(self.batch_size,1),
                    self.learning_rate_actor: self.lr_actor,
                    self.learning_rate_critic: self.lr_critic,
                    }

        dict_ = feed_dict()


        t = time.clock()
        if self.tensorboard_logs:
            summary, _, _, mse_val = self.sess.run([self.merged, self.q_train_step, self.mu_train_step, self.q_loss], feed_dict= dict_)
            if self.step % 10 == 0:
                self.train_writer.add_summary(summary, self.step)
        else:
            _, _, mse_val = self.sess.run([self.q_train_step, self.mu_train_step, self.q_loss], feed_dict= dict_)

        print('sgd step only:', time.clock() - t)


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
        return self.mu.eval(feed_dict= {self.obs : state})

    def eval_q(self, state, action):
        return self.q_train.eval(feed_dict= {self.obs : state, self.act_train : action})


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
            if self.dimO == 2:
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

    car = ddpg()
    car.main()
