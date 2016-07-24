import gym
import numpy as np

env = gym.make('Reacher-v1')
print env.action_space
print env.action_space.low
print env.action_space.high


print env.spec.reward_threshold


ep_lengths = []
for i in range(10000):
    env.reset()
    done = False

    sum_rewards = 0
    while not done:
        action = np.asarray((.001,.0005))
        # action = env.action_space.sample()
        # state, reward, done, _ = env.step(env.action_space.sample())
        state, reward, done, _ = env.step(action)
        sum_rewards+= reward
        env.render()
        print(state)
        print(done)
        print('reward',reward)

    # print(sum_rewards)
    ep_lengths.append(sum_rewards)

print(np.mean(ep_lengths))