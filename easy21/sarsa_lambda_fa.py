import cPickle
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt
from mpl_toolkits.mplot3d import axes3d
import os
os.environ["FONTCONFIG_PATH"]="/etc/fonts"


#
# import time
#
# def procedure():
#     time.sleep(2.5)

#
# # measure process time
# t0 = time.clock()
# procedure()
# print time.clock() - t0, "seconds process time"

from easy21 import *



class sarsa_fa:

    def __init__(self,epsilon=0.05):

        self.epsilon =  epsilon
        print "using epsilon ", self.epsilon
        self.alpha = 0.01


        pkl_file = open('Qtable_monte_carlo_1e6.pkl', 'rb')
        self.Q_table_mc = cPickle.load(pkl_file)

        pkl_file.close()

        self.error_lists = []

        self.mse_1000 = []


    def features(self,state,action):

        phi = np.zeros(36)

        if state[0] < 1 or state[0] > 21:
            raise ValueError('boeses Faul!')
        if state[1] < 1 or state[1] > 11:
            raise ValueError('boeses Faul!')

        dealer_int = [(1,4),(4,7),(7,10)]
        player_int = [(1,6),(4,9),(7,12),(10,15),(13,18),(16,21)]

        dealer_idx = []
        player_idx = []

        for int_idx in range(len(dealer_int)):
            if dealer_int[int_idx][0] <= state[1] and state[1] <= dealer_int[int_idx][1]:  #check dealer interval
                dealer_idx.append(int_idx)

        for int_idx in range(len(player_int)):
            if player_int[int_idx][0] <= state[0] and state[0] <= player_int[int_idx][1]:  #check player interval
                player_idx.append(int_idx)

        for d in dealer_idx:
            for p in player_idx:
                phi[d*12 + p*2 + action] = 1

        return phi




    def compare_qtables(self,Qtable,Q_mc):
        return  sqrt(mean_squared_error(Qtable.flatten(), Q_mc.flatten()))


    def runepisode(self,lambda_):
        #print "new episode"
        Esa = np.zeros(36)

        #initialize the state and acion randomly at beginning of episode
        state = np.random.randint(low = 1, high=10, size=None),np.random.randint(low = 1, high=10, size=None) #player, dealer
        A = self.policy(state)

        terminated = False
        while( not terminated):

            reward, successor, terminated = step(state[0],state[1],A)
            #print("successor" , successor)
            if not terminated:
                A_prime = self.policy(successor)
                Qsprime_aprime = self.features(successor,A_prime).dot(self.theta)
            else:
                Qsprime_aprime = 0

            Qsa = self.features(state,A).dot(self.theta)
            delta = reward + Qsprime_aprime - Qsa

            #counting state visits
            Esa = lambda_*Esa + self.features(state,A)

            #
            # print "state", state,"action ", A , "successor", successor
            # if not terminated:
            #     print "A prime", A_prime
            # print "Esa", Esa
            # print "theta", self.theta
            # print "Qsa", Qsa
            # print "Qsa prime ", Qsprime_aprime

            self.theta += self.alpha*delta*Esa

            if not terminated:
                A = A_prime
                state = successor   # the if is preventing illegal terminal states


    def policy(self,state):

        explore = np.random.choice([1,0],p=[self.epsilon, 1-self.epsilon])
        if not explore:
            return np.argmax((self.features(state,0).dot(self.theta),self.features(state,1).dot(self.theta)))
        else:
            return np.random.choice([1,0])


    def calcQtable(self):

        Qtable = np.zeros((21,10,2))
        for i in range(1,11):
            for j in range(1,22):
                for a in range(2):

                    Qtable[j-1,i-1,a] = self.features((j,i),a).dot(self.theta)

        return Qtable


    def loop_over_lambda(self, numiter=10001, n_lambda=3):


        print "numiter", numiter, "n_lambda", n_lambda
        self.n_lambda = n_lambda
        for lambda_ in np.linspace(0,1,n_lambda):

            self.theta = np.random.randn(36)

            mse = []

            print lambda_

            # policy = np.argmax(Qtable[state])
            for i in range(numiter):


                #print i
                self.runepisode(lambda_)

                #compare q tables
                if i%1000 == 0:
                    mse.append(self.compare_qtables(self.calcQtable(),self.Q_table_mc))       # mean squared error every 1000 iterations

                if i == 1000:
                    self.mse_1000.append(self.compare_qtables(self.calcQtable(),self.Q_table_mc))             #mean squared error at iteration 1000

            self.error_lists.append(mse)         #lists of error lists

        self.opt_Valuefunction = np.max(self.calcQtable(),2)



    def visualize(self):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(range(1,11), range(1,22))
        # print(X.shape,Y.shape)
        ax.plot_wireframe(X,Y, self.opt_Valuefunction)
        ax.set_xlabel("dealer")
        ax.set_ylabel("player")
        ax.set_zlabel("value")

        fig = plt.figure()
        opt_policy = np.argmax(self.calcQtable(),2)
        plt.imshow(opt_policy,cmap=plt.get_cmap('gray'),interpolation='none')

        fig = plt.figure()
        plt.plot(self.mse_1000)
        plt.xlabel("lambda")
        plt.ylabel("mean squared errror from 1e6 monte carlo")


        fig = plt.figure()
        for i in range(11):
            line1, = plt.plot(range(len(self.error_lists[i])), self.error_lists[i], label="lambda ="+str(np.linspace(0,1,3)[i]))
            plt.legend(handles=[line1], loc=1)

        plt.xlabel("1000 episodes")
        plt.ylabel("Mean squared error to Monte Carlo 1e6 ")

        plt.show()

        plt.close('all')




# sarsa_test = sarsa_fa()
#
# sarsa_test.loop_over_lambda(100000)
#
# sarsa_test.visualize()