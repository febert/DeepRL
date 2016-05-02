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

pkl_file = open('Qtable_monte_carlo_1e6.pkl', 'rb')
Q_table_mc = cPickle.load(pkl_file)

pkl_file.close()





def compare_qtables(Qtable,Q_table_mc):
    #np.sum(((Qtable-Q_table_mc)**2).flatten())
    return  sqrt(mean_squared_error(Qtable.flatten(), Q_table_mc.flatten()))



def runepisode():

    #initialize the state and acion randomly at beginning of episode
    state = np.random.randint(low = 1, high=10, size=None),np.random.randint(low = 1, high=10, size=None) #player, dealer
    A = policy(state)

    terminated = False
    while( not terminated):

        reward, successor, terminated = step(state[0],state[1],A)
        #print("successor" , successor)
        if not terminated:
            A_prime = policy(successor)
            Qsprime_aprime = Qtable[successor[0]-1,successor[1]-1,A_prime]
        else:
            Qsprime_aprime = 0

        delta = reward + Qsprime_aprime - Qtable[state[0]-1,state[1]-1,A]
        episode.append((state,A,reward))

        #counting state visits
        Nsa[state[0]-1,state[1]-1,A] += 1
        Esa[state[0]-1,state[1]-1,A] += 1

        for s, a, reward in episode:
            alpha = 1/Nsa[s[0]-1,s[1]-1,a]
            Qtable[s[0]-1,s[1]-1,a] += alpha*delta*Esa[s[0]-1,s[1]-1,a]
            Esa[s[0]-1,s[1]-1,a]*= lambda_



        if not terminated:
            A = A_prime
            state = successor


def policy(state):
    # print(Nsa)
    # print Nsa.shape
    Ns = Nsa[state[0]-1,state[1]-1,0] + Nsa[state[0]-1,state[1]-1,1]
    N_0 = 100
    epsilon = N_0/(N_0 + Ns)

    explore = np.random.choice([1,0],p=[epsilon, 1-epsilon])
    if not explore:
        return np.argmax(Qtable[state[0]-1,state[1]-1,:])
    else:
        return np.random.choice([1,0])

numiter = 10 #00000


error_lists = []

mse_1000 = []

for lambda_ in np.linspace(0,1,11):


    mse = []


    Qtable = np.zeros((21,10,2))
    Nsa = np.zeros((21,10,2))

    print lambda_

    # policy = np.argmax(Qtable[state])
    for i in range(numiter):
        episode = []  #just one episode
        Esa = np.zeros((21,10,2))
        #print i
        runepisode()

        #compare q tables
        if i%1000 == 0:
            mse.append(compare_qtables(Qtable,Q_table_mc))

    mse_1000.append(compare_qtables(Qtable,Q_table_mc))

    error_lists.append(mse)

opt_Valuefunction = np.max(Qtable,2)
print(opt_Valuefunction.shape)


## save to file
save = False
if save:
    output = open('Qtable_monte_carlo_1e6.pkl', 'wb')
    cPickle.dump(opt_Valuefunction, output)
    output.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(range(1,11), range(1,22))
# print(X.shape,Y.shape)
ax.plot_wireframe(X,Y, opt_Valuefunction)
ax.set_xlabel("dealer")
ax.set_ylabel("player")
ax.set_zlabel("value")


fig = plt.figure()
opt_policy = np.argmax(Qtable,2)
plt.imshow(opt_policy,cmap=plt.get_cmap('gray'),interpolation='none')
plt.xlabel("dealer")
plt.ylabel("player")

fig = plt.figure()
plt.plot(mse_1000)
plt.xlabel("lambda")
plt.ylabel("mean squared errror from 1e6 monte carlo")


fig = plt.figure()
for i in range(11):
    line1, = plt.plot(range(len(error_lists[i])), error_lists[i], label="lambda ="+str(np.linspace(0,1,11)[i]))
    plt.legend(handles=[line1], loc=1)

plt.xlabel("1000 episodes")
plt.ylabel("Mean squared error to Monte Carlo 1e6 ")

plt.show()




