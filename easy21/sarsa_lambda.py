import cPickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os
os.environ["FONTCONFIG_PATH"]="/etc/fonts"



from easy21 import *


Qtable = np.zeros((21,10,2))
Nsa = np.zeros((21,10,2))
lambda_ = 0.9

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

numiter = 1000000

# policy = np.argmax(Qtable[state])
for i in range(numiter):
    episode = []  #just one episode
    Esa = np.zeros((21,10,2))
    print i
    runepisode()


opt_Valuefunction = np.max(Qtable,2)
print(opt_Valuefunction.shape)


## save to file
save = False
if save:
    output = open('opt_valuefcn.pkl', 'wb')
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
plt.show()




pkl_file = open('opt_valuefcn.pkl', 'rb')
data1 = cPickle.load(pkl_file)
pkl_file.close()
