from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time

import cPickle

import gym as gym

class mountain_car():

    def __init__(self,
                 gamma=0.99,
                 init_alpha=1e-2,
                 constant_alpha=False,
                 num_features=2,
                 init_epsilon=1.0,
                 update_epsilon=True,
                 policy_mode='stochastic',
                 N_0=50.,
                 random_init_theta=False,
                 negative_gradient = False):


        if negative_gradient:
            self.baseline_enable = 0
            self.gradient_sign = -1
        else:
            self.baseline_enable = 1
            self.gradient_sign = 1


        self.env = gym.make('MountainCar-v0')
        self.num_actions = 3
        self.num_features = num_features
        ## learning rate
        self.init_alpha = init_alpha
        self.alpha = init_alpha
        self.constant_alpha = constant_alpha

        # lengths of all the played episodes
        self.episode_lengths = []

        # running average of the mean value of the episodes' steps
        self.mean_value_fcn = 0.0

        # lengths of all the tested episodes TODO
        self.test_lengths = []

        ## policy parameters initialization
        if random_init_theta:
            # random initialization
            self.theta = np.random.randn(self.num_features, self.num_actions)
        else:
            # zero initialization
            self.theta = np.zeros((self.num_features, self.num_actions))

        ## stochastic or deterministic softmax-based actions
        self.policy_mode = policy_mode

        ## exploration parameters
        # too much exploration is wrong!!!
        self.epsilon = init_epsilon    # explore probability
        self.update_epsilon = update_epsilon
        self.total_runs = 0.
        # too long episodes give too much negative reward!!!!
        # self.max_episode_length = 1000000
        # ----> Use gamma!!!!! TODO: slower decrease?
        self.gamma = gamma  #similar to 0.9

        self.N_0 = N_0

        np.seterr(all='raise')

    # def get_features(self, state):
    #     # normalizer = np.abs(self.env.observation_space.high - self.env.observation_space.low)
    #     if self.num_features == 2:
    #         return np.array([state[0],
    #                          state[1]])
    #     if self.num_features == 3:
    #         return np.array([1,
    #                          state[0],
    #                          state[1]])
    #     if self.num_features == 4:
    #         return np.array([state[0],
    #                          state[1],
    #                          state[0],
    #                          state[1]])
    #     if self.num_features == 5:
    #         return np.array([1,
    #                          state[0],
    #                          state[1],
    #                          state[0],
    #                          state[1]])
    def get_features(self, state, normalize=True):
        normalizer = np.abs(self.env.observation_space.high - self.env.observation_space.low)
        if not normalize:
            normalizer = np.ones_like(normalizer)
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
                print(error)
                # print(".", end="")

                prob_distrib[np.log(prob_distrib)<-100] =0        # print("activation ", activation)


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

    def run_episode(self, enable_render=False, limit=1000000):
        episode = []
        state = self.env.reset()

        count = 0
        done = False
        while ( not done ):

            if len(episode)>limit: break

            count += 1

            action = self.policy(state, mode=self.policy_mode)
            state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
            if enable_render: self.env.render()
            # if count > self.max_episode_length: break;

        if enable_render: print("This episode took {} steps".format(count))

        return episode


    # gradient of the policy function
    def score_function(self, state, action):
        features = self.get_features(state)

        prob_distrib = self.eval_action_softmax_probabilities(state)
        weight = -prob_distrib
        weight[action] += 1.

        try:
            score = features[:,None]*weight[None,:]
            # print("score neg")
            # print(score)
        except FloatingPointError, error:
            print(error)
            print("features", features)
            print("prob_distrib", prob_distrib)

        return score

    def numerical_score_function(self, state, action, delta=1e-8):
        # backup value
        save_theta = np.array(self.theta)

        # base value
        log_prob = np.log(self.eval_action_softmax_probabilities(state))[action]

        score = np.zeros_like(self.theta)

        # apply delta to every component of theta
        for index, th in np.ndenumerate(self.theta):
            self.theta[index] += delta
            score[index] = ( np.log(self.eval_action_softmax_probabilities(state))[action] - log_prob ) / delta

            # restore value
            self.theta[index] = save_theta[index]

        return score

    def train(self, iter=1000, dataname = 'dataname', save = False):
        # fig = plt.figure()

        for it in range(iter):
            # run episode
            episode = self.run_episode()
            self.total_runs += 1.0
            # keep track of training episode lengths
            self.episode_lengths.append(len(episode))

            if (it+1)%10 == 0:
                # output training info
                print("EPISODE #{}".format(self.total_runs))
                print("with a exploration of {}%".format(self.epsilon*100))
                print("and learning rate of {}".format(self.alpha))
                print("lasted {0} steps".format(len(episode)))

                # do a test run
                save_policy_mode = self.policy_mode
                save_epsilon = self.epsilon
                # print(self.policy_mode)

                self.policy_mode = "deterministic"
                self.epsilon = 0.
                det_episode = self.run_episode(limit=10000)

                self.test_lengths.append(len(det_episode))
                self.policy_mode = save_policy_mode
                self.epsilon = save_epsilon

            # theta += alpha*calculate_gradient()
            # Version reversed (equivalent if only updated at the end)
            theta_update = 0.
            value_fcn = 0.
            mean_value = 0.
            # if it is the first run
            if self.total_runs < 1.1:
                # get a first estimation of the average mean
                for state, action, reward in episode:
                    value_fcn = reward + self.gamma*value_fcn
                    mean_value += value_fcn / len(episode)
                self.mean_value_fcn += (mean_value - self.mean_value_fcn) / self.total_runs

            # backwards, for every step in the episode
            for idx in range(len(episode),0,-1):
                (state, action, reward) = episode[idx-1]
                value_fcn = reward + self.gamma*value_fcn
                # incrementally calculate theta update
                theta_update += self.score_function(state,action)*(value_fcn - self.mean_value_fcn*self.baseline_enable)
                # mean value function
                mean_value += value_fcn / len(episode)
                # TODO: check sign in this line. HAS IT TO BE -. WHY??
            # update theta
            self.theta += self.alpha*theta_update*self.gradient_sign
            # update mean value function
                # if it is not the first run
            if self.total_runs > 1.1:
                self.mean_value_fcn += (mean_value - self.mean_value_fcn) / self.total_runs


            #print('theta_update', theta_update)

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
                self.plot_policy(mode= 'deterministic')
            if (it+1)%100 == 0:
                self.plot_training()
                self.plot_testing()
                # self.test_episode
            # if i % 10 == 0:
            #     plt.plot(episode_lengths)
            #     plt.show()

            # decrease exploration
            if self.update_epsilon:
                self.epsilon = self.N_0 / (self.N_0 + self.total_runs)
            # decrease alpha
            if not self.constant_alpha:
                self.alpha = self.init_alpha / np.sqrt(self.total_runs)


            if save:
                self.savedata(dataname=dataname)

        return self.theta



    def savedata(self, dataname):

        output = open(dataname, 'wb')
        cPickle.dump(self.theta, output)
        cPickle.dump(self.episode_lengths, output)

        cPickle.dump(self.test_lengths, output)
        output.close()

    def loaddata(self, dataname):

        pkl_file = open(dataname, 'rb')
        self.theta = cPickle.load(pkl_file)
        self.episode_lengths = cPickle.load(pkl_file)
        self.test_lengths = cPickle.load(pkl_file)

        print( self.theta)
        print( self.episode_lengths)
        print( self.episode_lengths)


        pkl_file.close()


    def plot_policy(self, resolution=100, mode= 'stochastic'):

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
                greedy_policy[i,j] = self.policy((x,v),mode)
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
        plt.yscale('log')
        plt.show()


    def plot_testing(self):

        fig = plt.figure()
        plt.plot(self.test_lengths)
        plt.yscale('log')
        plt.show()


    def compare_gradients(self,):

        self.theta = np.array([[-0.01338339, -0.01746333 , 0.03084672],[-0.38015887 , 0.00830843,  0.37185044]])

        numepisodes = 200
        self.epsilon = 0  #the exploration has to be set to zero!

        self.policy_mode = 'deterministic'  # could work with stocastic as well

        self.compute_with_policy_gradient_theorem(numepisodes)
        self.compute_numeric_gradient(numepisodes)




    def compute_with_policy_gradient_theorem(self, numepisodes):

        accum_values = np.zeros_like(self.theta)
        for epi_number in range(numepisodes):

            episode = self.run_episode()


            #print(self.policy_mode)
            value_fcn = 0
            for idx in range(len(episode),0,-1):
                (state, action, reward) = episode[idx-1]
                value_fcn = reward + self.gamma*value_fcn
                accum_values += self.score_function(state,action)*value_fcn


            #print(len(episode))

        average_gradient = accum_values/numepisodes

        print('average gradient computed with policy gradient theorem')
        print('result after %d episodes:'%(numepisodes))
        print(average_gradient)


    def compute_numeric_gradient(self,numepisodes):

        average_numeric_gradient = np.zeros_like(self.theta)
        numeric_gradient_accum = np.zeros_like(self.theta)

        theta_saved = np.array(self.theta)

        before = 0.0
        for epi_number in range(numepisodes):
            before -= len(self.run_episode())  # compute length of episode before applying perturbations
        before=float(before)/numepisodes

        print('average episode length before perturbation',before)

        for i in range(self.theta.shape[0]):
            for j in range(self.theta.shape[1]):

                for epi_number in range(numepisodes):
                    perturbation = 1e-5
                    # perturb the selected parameter
                    self.theta = np.array(theta_saved)
                    self.theta[i][j]+= perturbation

                    after = -len(self.run_episode())
                    #print('episode length after perturbing element %d,%d : %d' % (i,j,after) )

                    numeric_gradient_accum[i][j]+= float(after-before)/perturbation

                print(i, j)
                print( numeric_gradient_accum)

                average_numeric_gradient[i][j] = numeric_gradient_accum[i][j] /numepisodes

        print('average gradient computed numerically')
        print('result after %d episodes:'% (numepisodes))
        print(average_numeric_gradient)

        self.theta = np.array(theta_saved)
