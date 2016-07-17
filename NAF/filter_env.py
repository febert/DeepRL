import numpy as np
import gym


def makeFilteredEnv(env):
    """ crate a new environment class with actions and states normalized to [-1,1] """
    acsp = env.action_space
    obsp = env.observation_space
    if not type(acsp) == gym.spaces.box.Box:
        raise RuntimeError('Environment with continous action space (i.e. Box) required.')
    if not type(obsp) == gym.spaces.box.Box:
        raise RuntimeError('Environment with continous observation space (i.e. Box) required.')

    env_type = type(env)
    # print 'timestep limit', env.timestep_limit

    class FilteredEnv(env_type):
        def __init__(self):
            self.__dict__.update(env.__dict__)  # transfer properties

            # Observation space
            if np.any(obsp.high < 1e10):
                h = obsp.high
                l = obsp.low
                sc = h - l
                self.o_c = (h + l) / 2.
                self.o_sc = sc / 2.
            else:
                self.o_c = np.zeros_like(obsp.high)
                self.o_sc = np.ones_like(obsp.high)

            # Action space
            h = acsp.high
            l = acsp.low
            sc = (h - l)
            self.a_c = (h + l) / 2.
            self.a_sc = sc / 2.

            # Check and assign transformed spaces
            self.observation_space = gym.spaces.Box(self.filter_observation(obsp.low),
                                                    self.filter_observation(obsp.high))
            self.action_space = gym.spaces.Box(-np.ones_like(acsp.high), np.ones_like(acsp.high))

            def assertEqual(a, b):
                assert np.all(a == b), "{} != {}".format(a, b)

            assertEqual(self.filter_action(self.action_space.low), acsp.low)
            assertEqual(self.filter_action(self.action_space.high), acsp.high)

        def filter_observation(self, obs):
            return (obs - self.o_c) / self.o_sc

        def filter_action(self, action):
            return self.a_sc * action + self.a_c

        def step(self, action):

            # ac_f = np.clip(self.filter_action(action), self.action_space.low, self.action_space.high)
            ac_clipped = np.clip(action, self.action_space.low, self.action_space.high)
            ac_f = self.filter_action(ac_clipped)

            # print(ac_f)
            obs, reward, term, info = env_type.step(self, ac_f)  # super function
            obs = obs.squeeze()

            obs_f = self.filter_observation(obs)

            return obs_f, reward, term, info


    fenv = FilteredEnv()

    print('True action space: ' + str(acsp.low) + ', ' + str(acsp.high))
    print('True state space: ' + str(obsp.low) + ', ' + str(obsp.high))
    print('Filtered action space: ' + str(fenv.action_space.low) + ', ' + str(fenv.action_space.high))
    print('Filtered state space: ' + str(fenv.observation_space.low) + ', ' + str(fenv.observation_space.high))

    return fenv
