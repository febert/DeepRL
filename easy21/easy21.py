import numpy as np

# class easy21:
#
#     def __init__(self):
#
#         self.dealer = []
#         self.player = []
#
#         #1 is black
#         #-1 is red
#         self.dealer.append((1,np.random.randint(low = 1, high=10, size=None)))   # (color,value)
#         self.player.append((1,np.random.randint(low = 1, high=10, size=None)))   # (color,value)
#         self.dealerscore = self.dealer[0][1]
#         self.playerscore = self.player[0][1]
#
#         self.gameover = False
#
#     def stick(self):
#
#
#         #the dealer plays until the end
#
#         while self.dealerscore < 17:
#             #the deals hits !
#
#             self.dealer.append((np.random.choice([-1,1,1]),np.random.randint(low = 1, high=10, size=None)))
#             self.dealerscore+= self.dealer[-1][0]*self.dealer[-1][1]
#             if self.dealerscore > 21 or self.dealerscore < 1:
#                 self.gameover = True
#                 break
#
#
#         #game is over evaluate
#         self.gameover = True
#
#
#     def getscore(self):
#
#         if not self.gameover:
#             raise ValueError('A very specific bad thing happened (game is not over)')
#
#         if self.playerscore > 21 or self.playerscore < 1:
#             return -1
#         if self.dealerscore > 21 or self.dealerscore < 1:
#             return 1
#         if self.dealerscore > self.playerscore:
#             return -1
#         if self.dealerscore == self.playerscore:
#             return 0
#         if self.dealerscore < self.playerscore:
#             return 1
#
#     def hit(self):
#         if self.gameover:
#             print(self.getscore())
#             raise ValueError('(game is over)')
#
#         #draw a card
#         self.player.append((np.random.choice([-1,1,1]),np.random.randint(low = 1, high=10, size=None)))
#         self.playerscore+= self.player[-1][0]*self.player[-1][1]
#
#         if self.playerscore > 21 or self.playerscore < 1:
#             self.gameover = True




def step(player_sum, dealer_sum, action): #action; 1 hit, 0 stick
    reward = 0
    terminated = False
    print("action", action)
    if action: # when hitting
        player_sum+= np.random.choice([-1,1,1])*np.random.randint(low = 1, high=10, size=None)
        successor_state = (player_sum, dealer_sum)
        if player_sum > 21 or player_sum < 1:
            reward = -1
            terminated = True

    else:  # when sticking
        while dealer_sum < 17:
        #the deals hits !
            dealer_sum+= np.random.choice([-1,1,1])*np.random.randint(low = 1, high=10, size=None)
            successor_state = (player_sum, dealer_sum)
            # print("dealer_sum", dealer_sum)

            if dealer_sum > 21 or dealer_sum < 1:
                reward = 1
                terminated = True
                return reward, successor_state, terminated


        if dealer_sum > player_sum:
            reward = -1
        if dealer_sum == player_sum:
            reward = 0
        if dealer_sum < player_sum:
            reward = 1

        terminated = True

    return reward, successor_state, terminated


# sum = 0
# num = 10000.0
# for i in range(int(num)):
#     dealer_sum = np.random.randint(low = 1, high=10, size=None)
#     player_sum = np.random.randint(low = 1, high=10, size=None)
#     print("dealer_sum", dealer_sum)
#     print("player_sum", player_sum)
#     reward = 0
#     while player_sum < 17:
#         reward, (player_sum, dealer_sum), terminated = step(player_sum, dealer_sum,1)  #hit
#         print("player_sum", player_sum)
#         if reward == -1:
#             print("You lost haha")
#             break
#
#     if not reward == -1:
#         reward, (player_sum, dealer_sum), terminated = step(player_sum, dealer_sum,0)
#     print("dealer_sum", dealer_sum)
#
#
#     sum += reward
# print(sum/num)
# #
# #
#
#
# #testrun
# sum = 0
# num = 10000.
# for i in range(int(num)):
#     testgame = easy21()
#
#     while testgame.playerscore < 17:
#         testgame.hit()
#         # print('player cards',testgame.player)
#         # print('player score ',testgame.playerscore)
#         #
#         if testgame.gameover:
#             # print(testgame.getscore())
#             break
#
#     testgame.stick()
#
#     # print('dealer cards',testgame.dealer)
#     # print('dealer score ',testgame.dealerscore)
#     # print(testgame.getscore())
#     sum += testgame.getscore()
#
# print(sum/num)

