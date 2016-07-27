import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('Acrobot-v0')
print env.action_space
# print env.action_space.low
# print env.action_space.high


print env.spec.reward_threshold


ep_lengths = []
for i in range(10000):
    env.reset()
    done = False

    sum_rewards = 0
    while not done:
        # action = np.asarray((.001,.0005))
        action = env.action_space.sample()
        # state, reward, done, _ = env.step(env.action_space.sample())
        state, reward, done, _ = env.step(action)
        sum_rewards+= reward
        rgbarray= env.render('rgb_array')

        print('state',state)
        print('done', done)
        print('reward',reward)
        print('rgb array size', rgbarray.shape)

        plt.imshow(rgbarray)
        plt.show()

    # print(sum_rewards)
    ep_lengths.append(sum_rewards)

print(np.mean(ep_lengths))