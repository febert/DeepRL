import numpy as np

class easy21:

    def __init__(self):

        self.dealer = []
        self.player = []

        #1 is black
        #-1 is red
        self.dealer.append((1,np.random.randint(low = 1, high=10, size=1)))   # (color,value)
        self.player.append((1,np.random.randint(low = 1, high=10, size=1)))   # (color,value)
        self.dealerscore = self.dealer[0][1]
        self.playerscore = self.player[0][1]

        self.gameover = False

    def stick(self):


        #the dealer plays until the end

        while self.dealerscore < 17:
            #the deals hits !

            self.dealer.append((np.random.choice([-1,1,1]),np.random.randint(low = 1, high=10, size=1)))
            self.dealerscore+= self.dealer[-1][0]*self.dealer[-1][1]

        #game is over evaluate

        if self.dealerscore > self.playerscore:


        self.dealerscore =

        if self.dealer()


    def hit:
        #draw a card


    def
        np.random.randint(low, high=None, size=None)






