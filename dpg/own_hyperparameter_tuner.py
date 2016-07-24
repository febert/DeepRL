from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ddpg3
import numpy as np


np.set_printoptions(threshold=np.inf)


class hyper_parameter_tuner:

    def __init__(self, param_dict, num_exp, dataname
                 ):

        self.num_exp = num_exp
        self.param_dict = param_dict
        self.dataname = dataname

    def sampler(self):

        val = {}
        for key in self.param_dict.keys():
            # print(self.param_dict[key])
            ind = np.random.random_integers(0,len(self.param_dict[key])-1)
            val[key] = self.param_dict[key][ind]

        return val


    def run_experiments(self):

        print('starting experiments...')

        score_params_list = []

        for n_exp in range(self.num_exp):
            print('##################################################')
            print('starting run ', n_exp)

            value_dict = self.sampler()
            print('using values')
            print(value_dict)
            score = self.run_trial(value_dict=value_dict)
            print('score: ', score)
            score_params_list.append((score, value_dict))
            self.savedata(self.dataname, score_params_list)

    def savedata(self, dataname, data):
        import cPickle
        output = open(dataname, 'wb')
        cPickle.dump(data, output)
        output.close()

    def run_trial(self, value_dict):

        with ddpg3.ddpg(environment=value_dict['env'],
                        #environment='MountainCarContinuous-v0',
                        learning_rates= value_dict['lr'],
                        noise_scale= value_dict['noise_level'],
                        enable_plotting= False,
                        ql2= value_dict['weight_decay'],
                        tensorboard_logs= False,
                        ) as ddpg3_instance:

            score = ddpg3_instance.main()

        return score

if __name__ == '__main__':


    lr_list = np.linspace(1e-5, 1e-2, num=10)
    lr_list = zip(lr_list * 0.1, lr_list)  # the order is actor_lr critic_lr, the critic needs to be faster
    print('Using the following parameter sets: ')
    print('lr_list: ', lr_list)

    noise_levels = np.linspace(start= 0.5, stop=1, num = 5)
    print('noise levels: ', noise_levels)

    ql2_factor = np.linspace(start= 0, stop=0.01, num = 2)
    print('weight decay ql2:', ql2_factor)

    paramdict = {'env':['InvertedPendulum-v1'],
                'lr': lr_list,
                'noise_level': noise_levels,
                'weight_decay': ql2_factor
    }

    hpt = hyper_parameter_tuner(param_dict= paramdict, num_exp= 10, dataname= 'hypertest')
    hpt.run_experiments()

