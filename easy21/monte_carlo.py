import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import  os
os.environ["FONTCONFIG_PATH"]="/etc/fonts"



from easy21 import *


Qtable = np.zeros((21,10,2))
Nsa = np.zeros((21,10,2))
# policy = np.zeros(21,10)

def runepisode():

    #initialize the state randomly
    state = np.random.randint(low = 1, high=10, size=None),np.random.randint(low = 1, high=10, size=None) #player, dealer

    terminated = False
    while( not terminated):
        # action = policy(Qtable[state])
        action = policy(state)
        reward, successor, terminated = step(state[0],state[1],action)
        episode.append((state,action,reward))

        #counting state visits
        Nsa[state[0]-1,state[1]-1,action] += 1

        state = successor



def policy(state):

    Ns = Nsa[state[0]-1,state[1]-1,0] + Nsa[state[0]-1,state[1]-1,1]
    N_0 = 100
    epsilon = N_0/(N_0 + Ns)

    explore = np.random.choice([1,0],p=[epsilon, 1-epsilon])
    if not explore:
        return np.argmax(Qtable[state])
    else:
        return np.random.choice([1,0])

numiter = 10

# policy = np.argmax(Qtable[state])
for i in range(numiter):
    episode = []  #just one episode
    runepisode()
    Gt = episode[-1][2]

    for state, action, reward in episode:
        Qtable[state[0]-1,state[1]-1,action] += (1/Nsa[state[0]-1,state[1]-1,action])*(Gt - Qtable[state[0]-1,state[1]-1,action])



opt_Valuefunction = np.max(Qtable,2)
print(opt_Valuefunction.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(range(1,11), range(1,22))
print(X.shape,Y.shape)
ax.plot_wireframe(X,Y, opt_Valuefunction)
