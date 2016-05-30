from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import time
import math

import cPickle

import gym as gym

class mountain_car():

    def __init__(self,
                 gamma=0.99,
                 init_alpha=1e-6,
                 constant_alpha=True,
                 value_fcn_learning_rate=1e-3,
                 type_features=0,
                 init_epsilon=1.0,
                 update_epsilon=True,
                 policy_mode='stochastic',
                 N_0=50.,
                 random_init_theta=True,
                 negative_gradient = False,
                 environment = 'MountainCar-v0',
                #  environment = 'Acrobot-v0',
                 lambda_ = 0.5,
                 tile_resolution = 20
                 ):
        self.limit_steps_per_episode = 5000

        if negative_gradient:
            self.baseline_enable = 0
            self.gradient_sign = -1
        else:
            self.baseline_enable = 1
            self.gradient_sign = 1


        self.env = gym.make(environment)
        self.num_actions = self.env.action_space.n
        self.statedim = self.env.observation_space.shape[0]

        self.type_features = type_features
        ## learning rate
        self.init_alpha = init_alpha
        self.alpha = init_alpha
        self.constant_alpha = constant_alpha

        self.value_fcn_learning_rate = value_fcn_learning_rate

        # lengths of all the played episodes
        self.episode_lengths = []

        # running average of the mean value of the episodes' steps
        self.mean_value_fcn = 0.0

        # lengths of all the tested episodes TODO
        self.test_lengths = []

        ## policy parameters initialization

        if self.type_features == 0:
            self.num_features = self.env.observation_space.shape[0]
        if self.type_features == 1:
            self.num_features = 1
        if self.type_features == 2:
            self.num_features = 2
        if self.type_features == 3:
            self.num_features = 3
        if self.type_features == 4:
            self.num_features = 4
        if self.type_features == 5:
            self.num_features = 5
        if self.type_features == "poly4":
            self.num_features = 4* self.env.observation_space.shape[0]

        #tile features:
        self.tile_resolution = tile_resolution
        self.overlap = False
        if self.overlap:
            self.num_tile_features = pow(self.tile_resolution,self.statedim)*2
        else:
            self.num_tile_features = pow(self.tile_resolution,self.statedim)

        #for the value function estimation:
        self.v = np.zeros(self.num_tile_features)
        self.eligibility_vector = np.zeros(self.num_tile_features)
        self.lambda_ = lambda_

        if random_init_theta:
            # random initialization
            self.theta = np.random.randn(self.num_features * self.num_actions)
        else:
            # zero initialization
            self.theta = np.zeros(self.num_features * self.num_actions)

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

        self.DEBUG = False

        print('N_0',self.N_0)
        print('init alpha',self.init_alpha)
        print('Constant Alpha', constant_alpha)


    def get_features(self, state, normalize=True):
        mean = (self.env.observation_space.high + self.env.observation_space.low)/2.
        normalizer = np.abs(self.env.observation_space.high - self.env.observation_space.low)

        if not normalize:
            normalizer = np.ones_like(normalizer)
            mean = np.zeros_like(mean)
        if self.type_features == 0:
            features = np.array(state)
        if self.type_features == 1:
            features = np.array([state[1]/normalizer[1]])
        if self.type_features == 2:
            features = np.array([state[0]/normalizer[0],
                             state[1]/normalizer[1]])
        if self.type_features == 3:
            features = np.array([1,
                             state[0]/normalizer[0],
                             state[1]/normalizer[1]])
        if self.type_features == 4:
            features = np.array([state[0]/normalizer[0],
                             state[1]/normalizer[1],
                             (state[0]/normalizer[0])**2,
                             (state[1]/normalizer[1])**2])
        if self.type_features == 5:
            features = np.array([1,
                             state[0]/normalizer[0],
                             state[1]/normalizer[1],
                             (state[0]/normalizer[0])**2,
                             (state[1]/normalizer[1])**2])
        if self.type_features == "poly4":
            state_0 = state - mean
            features = np.concatenate((state_0/normalizer, (state_0/normalizer)**2, (state_0/normalizer)**3, (state_0/normalizer)**4))
        # if self.type_features ==

        full_features = np.zeros((self.num_actions,self.num_features*self.num_actions))

        for action in range(self.num_actions):

            # print(full_features, full_features.shape)
            # print(features, features.shape)
            # print(full_features[action,action*self.num_features:(action+1)*self.num_features].shape)
            full_features[action,action*self.num_features:(action+1)*self.num_features] = features
            if self.DEBUG: print(features.shape, end="")

        return  full_features


    def get_tile_feature(self, state):

        high = self.env.observation_space.high
        obs_dim = self.env.observation_space.shape[0]   #dimension of observation space
        low = self.env.observation_space.low
        numactions = self.env.action_space.n

        stepsize = (high - low)/self.tile_resolution

        ind = np.floor((state-low)/stepsize).astype(int)
        ind[ind>=self.tile_resolution] = self.tile_resolution -1
        ind = tuple(ind)

        grid = np.zeros((np.ones(obs_dim)*self.tile_resolution).astype(int))

        try:
            grid[ind] = 1
        except IndexError, error:
            print(error)
            print("ind", ind)
            print("state", state)
            print("high", high)
            print("low", low)

        if self.overlap:
            ind_shift = np.floor((state-low+stepsize/2)/stepsize).astype(int)
            ind_shift[ind_shift>=self.tile_resolution] = self.tile_resolution -1
            ind_shift = tuple(ind_shift)

            grid_shift = np.zeros((np.ones(obs_dim)*self.tile_resolution).astype(int))
            grid_shift[ind_shift] = 1

            flatgrid = np.concatenate((grid,grid_shift), axis= 0).flatten()
        else:
            flatgrid = grid.flatten()


        #used for when approximating
        #full_feature = np.zeros(self.num_actions,length_flatgrid*numactions)
        #for action in range(self.num_actions):
        #    full_feature[action,length_flatgrid*action: (action+1)*length_flatgrid] = flatgrid

        #return  full_feature
        return  flatgrid


    # TODO: better solution to over-/underflows? Maybe not needed if proper learning?
    def eval_action_softmax_probabilities(self, state):

        #softmax probability
        activation = self.get_features(state).dot(self.theta)
        activation-= np.max(activation)

        prob_distrib = np.exp(activation)
        prob_distrib /= np.sum(prob_distrib)

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

    def run_episode(self, enable_render=False, limit=150000):
        limit = np.min((limit, self.limit_steps_per_episode))

        episode = []
        state = self.env.reset()

        count = 0
        done = False
        while ( not done ):

            if len(episode)>limit:
                return []

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
        #print('features', features)

        prob_distrib = self.eval_action_softmax_probabilities(state)

        try:
            score = features[action] - prob_distrib.dot(features)
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

    def train(self, iter=1000, dataname = 'unnamed_data', save = False):
        gradient_sign_count = 0.0
        # fig = plt.figure()

        for it in range(iter):
            # run episode
            episode = self.run_episode(enable_render=False)
            self.total_runs += 1.0
            # keep track of training episode lengths
            self.episode_lengths.append(len(episode))

            if (it+1)%10 == 0:
                # output training info
                print("----------------------------------------")
                print("EPISODE #{}".format(int(self.total_runs)))
                print("with a exploration of {}%".format(self.epsilon*100))
                print("and learning rate of {}".format(self.alpha))
                print("lasted {0} steps".format(len(episode)))
                print("----------------------------------------")

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

            # calculate all the tile features once, to use below
            tile_features_mat = np.zeros((len(episode), self.num_tile_features))
            for idx in range(len(episode)):
                tile_features_mat[idx,:] = self.get_tile_feature(episode[idx][0])




            #offline td-lambda for estimating the value function
            if not(len(episode)==0):
                (state, action, reward) = episode[0]
                previous_state = state #initialize with the start state
            self.eligibiltiy_vector = np.zeros(self.num_tile_features)
            for idx in range(1,len(episode)):
                (state, action, reward) = episode[idx]
                Vs = tile_features_mat[idx-1].dot(self.v)
                Vs_prime = tile_features_mat[idx].dot(self.v)

                delta_t = reward + self.gamma*Vs_prime - Vs

                self.eligibiltiy_vector = self.eligibiltiy_vector*self.gamma*self.lambda_ + tile_features_mat[idx]
                # print('eligibiltiy_vector', eligibiltiy_vector)
                # print('delta_t', delta_t)
                # print('Vs',Vs)
                # print('Vs_prime',Vs_prime)
                # print('v',self.v)
                self.v += self.value_fcn_learning_rate*delta_t*self.eligibiltiy_vector
                previous_state = state




            #monte carlo update

            # theta += alpha*calculate_gradient()
            # Version reversed (equivalent if only updated at the end)
            theta_update = np.zeros_like(self.theta)
            value_fcn = 0.
            later_Vest = 0.
            mean_value = 0.

            Vest_episode = tile_features_mat.dot(self.v)

            # backwards, for every step in the episode
            for idx in range(len(episode),0,-1):
                (state, action, reward) = episode[idx-1]
                value_fcn = reward + self.gamma*value_fcn

                # incrementally calculate theta update
                Vest = Vest_episode[idx-1]
                # if idx%1000==0:
                    # print(Vest, end=" ")
                # Vest = -100

                delta = value_fcn - Vest * self.baseline_enable
                theta_update += self.score_function(state,action)*delta
                # self.theta += self.alpha*self.score_function(state,action)*delta*self.gradient_sign

                later_Vest = Vest

                if delta > 0:
                    gradient_sign_count += 1
                if delta < 0:
                    gradient_sign_count -= 1

            self.theta += self.alpha*theta_update*self.gradient_sign

            print("gradient sign sum", gradient_sign_count, end=" ")


            # decrease exploration
            if self.update_epsilon:
                self.epsilon = self.N_0 / (self.N_0 + self.total_runs)
            # decrease alpha
            if not self.constant_alpha:
                self.alpha = self.init_alpha / np.sqrt(self.total_runs)

            if save:
                self.savedata(dataname=dataname)



            # PLOTTING --------------------------------
            if (it+1)%10 == 0:
                print("theta")
                print(self.theta)
                if self.env.spec.id == 'MountainCar-v0':
                    self.plot_policy(mode= 'stochastic')
                    self.plot_policy(mode= 'deterministic')
                    self.plot_eligibility()
                    self.plot_value_function()

            if (it+1)%10 == 0:
                self.plot_training()
                self.plot_testing()
                # self.plot_value_function()


        return self.theta

    def plot_eligibility(self):
        if self.overlap:
            e_vector = self.eligibiltiy_vector[0:self.num_tile_features/2]  #just visualize first half
        else:
            e_vector = np.array(self.eligibiltiy_vector)  #just visualize first half

        print('plotting the eligibility traces')

        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high

        # values to evaluate policy at
        x_range = np.linspace(obs_low[0], obs_high[0], self.tile_resolution)
        v_range = np.linspace(obs_low[1], obs_high[1], self.tile_resolution)

        # get actions in a grid
        e_mat = e_vector.reshape((self.tile_resolution,self.tile_resolution))

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_range, v_range)
        ax.plot_wireframe(X,Y, e_mat)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_zlabel("eligibility")
        plt.show()

    def plot_value_function(self):

        print('plotting the value function')

        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high

        # values to evaluate policy at
        x_range = np.linspace(obs_low[0], obs_high[0]-0.01, self.tile_resolution*3)
        v_range = np.linspace(obs_low[1], obs_high[1]-0.01, self.tile_resolution*3)

        # get actions in a grid
        value_func = np.zeros((x_range.shape[0], v_range.shape[0]))
        for i, state1 in enumerate(x_range):
            for j, state2 in enumerate(v_range):
                # print(np.argmax(self.get_features((x,v)).dot(self.theta)), end="")
                value_func[i,j] = -self.v.dot(self.get_tile_feature((state1,state2)))
        print("")


        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x_range, v_range)
        ax.plot_wireframe(X,Y, value_func)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_zlabel("negative value")
        plt.show()


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
                greedy_policy[i,j] = self.policy(np.array((x,v)),mode)
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
        self.episode_lengths[self.episode_lengths==0] = 1
        plt.plot(self.episode_lengths)
        plt.yscale('log')
        plt.show()


    def plot_testing(self):

        fig = plt.figure()
        self.test_lengths[self.test_lengths==0] = 1
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




#car1 = mountain_car(init_alpha=1e-3)


#state = (1,1)
#action = 0

#print('grad theorm score:',car1.score_function(state,action))
#print('numeric score:',car1.numerical_score_function(state,action))

#car1.train(iter=100)
