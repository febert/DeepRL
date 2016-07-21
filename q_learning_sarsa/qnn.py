from __future__ import print_function

import tensorflow as tf
import numpy as np
from collections import deque

class qnn:

    def __init__(self,
                 size_input,
                 size_output,
                 discount=0.99,
                 size_hidden = [300,400,400],
                #  descent_method = 'adam',
                 descent_method = 'grad',
                #  descent_method = 'rmsprop',
                 learning_rate = 1e-4,
                 batch_size = 50,
                 replay_memory = False,
                 keep_prob_val = 1.0,
                 is_a_prime_external = False,
                 replay_memory_size = 1e6,
                 ema_decay_rate = 0.999,
                 normalization_mean = None,
                 normalization_var = None,
                 env_name = None,
                 do_train_every_sample = False,
                 init_weights = None
                 ):
        print("normaliztion mean", normalization_mean)
        print("normalization var", normalization_var)

        self.do_train_every_sample = do_train_every_sample
        # if env_name == 'CartPole-v0':
        #     self.do_train_every_sample = True
        self.is_a_prime_external = is_a_prime_external # This means SARSA instead of Q-learning
        self.keep_prob_val = keep_prob_val
        self.batch_size = batch_size

        if init_weights is None:
            init_weights = 0.1
        self.init_weights = init_weights
        # print("init_weights", self.init_weights)

        def weight_variable(shape, fan_in=True, act=None):
            """
            Create a weight variable with appropriate initialization.
            Random to break symmetries.
            """
            if fan_in:
                if act is None:
                    std=1.0/np.sqrt(shape[0])
                elif act is tf.nn.relu:
                    print('relu!!!!')
                    std=np.sqrt(2.0)/np.sqrt(shape[0])
            else:
                std=self.init_weights
            print('weights std =', std)

            initial = tf.truncated_normal(shape, stddev=std)
            return tf.Variable(initial)

        def bias_variable(shape, fan_in_size=0, act=None):
            """
            Create a bias variable with appropriate initialization.
            Larger than zero to avoid dead ReLUs
            """
            # if fan_in_size > 0:
            #     if act is None:
            #         std=1.0/np.sqrt(fan_in_size)
            #     elif act is tf.nn.relu:
            #         print('relu!!!!')
            #         std=np.sqrt(2.0)/np.sqrt(fan_in_size)
            # elif fan_in_size == 0:
            #     std=0.0
            std=0.0
            print('bias std =', std)
            # initial = tf.constant(0.0, shape=shape)
            initial = tf.truncated_normal(shape, stddev=std)
            return tf.Variable(initial)

        def variable_summaries(var,name):
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.scalar_summary('mean/' + name, mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
                tf.scalar_summary('sttdev/' + name, stddev)
                tf.scalar_summary('max/' + name, tf.reduce_max(var))
                tf.scalar_summary('min/' + name, tf.reduce_min(var))
                tf.histogram_summary(name, var)

        def batch_norm(x,shape_x, scope):
            """
            Batch normalization on convolutional maps.
            Args:
                x:           Tensor, 4D BHWD input maps
                n_out:       integer, depth of input maps
                scope:       string, variable scope
            Return:
                normed:      batch-normalized maps
            """
            with tf.variable_scope(scope):
                beta = tf.Variable(tf.constant(0.0, shape=[shape_x]),
                                             name='beta', trainable=True)
                gamma = tf.Variable(tf.constant(1.0, shape=[shape_x]),
                                              name='gamma', trainable=True)
                # tf.histogram_summary('gamma', gamma)
                batch_mean, batch_var = tf.nn.moments(x, axes= [0], name='moments')
                # tf.scalar_summary('batch_mean', batch_mean)
                # tf.scalar_summary('batch_var', batch_var)
                ema = tf.train.ExponentialMovingAverage(decay=0.9)

                def mean_var_with_update():
                    ema_apply_op = ema.apply([batch_mean, batch_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(batch_mean), tf.identity(batch_var)

                # summary_batch_mean = ema.average(batch_mean)
                # summary_var_mean = ema.average(batch_var)
                # variable_summaries(summary_batch_mean, scope + '/batch_mean')
                # variable_summaries(summary_var_mean, scope + '/batch_mean')

                mean, var = tf.cond(self.phase_train,
                                    mean_var_with_update,
                                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
                names = tf.constant([scope + '/mean_x', scope + '/mean_v'])
                tf.scalar_summary(names, mean)
                # tf.scalar_summary('mean', mean)
                # tf.scalar_summary('var', var)
                normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

            return normed, (mean, var, beta, gamma)

        def dual_batch_norm(x,shape_x, scope):
            """
            Batch normalization on convolutional maps.
            Args:
                x:           Tensor, 4D BHWD input maps
                n_out:       integer, depth of input maps
                scope:       string, variable scope
            Return:
                normed:      batch-normalized maps
            """
            with tf.variable_scope(scope):
                beta = tf.Variable(tf.constant(0.0, shape=[shape_x]),
                                             name='beta', trainable=True)
                gamma = tf.Variable(tf.constant(1.0, shape=[shape_x]),
                                              name='gamma', trainable=True)
                # tf.histogram_summary('gamma', gamma)
                batch_mean, batch_var = tf.nn.moments(x, axes= [0], name='moments')
                # tf.scalar_summary('batch_mean', batch_mean)
                # tf.scalar_summary('batch_var', batch_var)
                ema = tf.train.ExponentialMovingAverage(decay=0.9)

                def mean_var_with_update():
                    ema_apply_op = ema.apply([batch_mean, batch_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(batch_mean), tf.identity(batch_var)

                # summary_batch_mean = ema.average(batch_mean)
                # summary_var_mean = ema.average(batch_var)
                # variable_summaries(summary_batch_mean, scope + '/batch_mean')
                # variable_summaries(summary_var_mean, scope + '/batch_mean')

                mean, var = tf.cond(self.phase_train,
                                    mean_var_with_update,
                                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
                names = tf.constant([scope + '/mean_x', scope + '/mean_v'])
                tf.scalar_summary(names, mean)
                # tf.scalar_summary('mean', mean)
                # tf.scalar_summary('var', var)
                normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

            return normed, (mean, var, beta, gamma)

        # batch norm parameter
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')

        def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
            """Reusable code for making a simple neural net layer.

            It does a matrix multiply, bias add, and then uses relu to nonlinearize.
            It also sets up name scoping so that the resultant graph is easy to read,
            and adds a number of summary ops.
            """
            # Adding a name scope ensures logical grouping of the layers in the graph.
            with tf.name_scope(layer_name):
                # This Variable will hold the state of the weights for the layer
                with tf.name_scope('weights'):
                    weights = weight_variable([input_dim, output_dim],act=act)
                    variable_summaries(weights, layer_name + '/weights')
                with tf.name_scope('biases'):
                    # biases = bias_variable([output_dim])
                    biases = bias_variable([output_dim], fan_in_size=input_dim, act=act)
                    variable_summaries(biases, layer_name + '/biases')
                # if it is not the last layer
                if act is not None:
                    with tf.name_scope('Wx_plus_b'):
                        preactivate = tf.matmul(input_tensor, weights) + biases
                        tf.histogram_summary(layer_name + '/pre_activations', preactivate)
                    activations = act(preactivate, 'activation')
                    tf.histogram_summary(layer_name + '/activations', activations)
                    return activations
                # if it is the last layer
                else:
                    with tf.name_scope('Wx_plus_b'):
                        activations = tf.matmul(input_tensor, weights) + biases
                        tf.histogram_summary(layer_name + '/activation', activations)
                    return activations

        def nn_dual_layer(input_tensor_list, input_dim, output_dim, layer_name, act=tf.nn.relu):
            """Create a layer with two different inputs and two different outputs
               calculated with the same weights. Thought for Q_learning training.

            It does a matrix multiply, bias add, and then uses relu to nonlinearize.
            It also sets up name scoping so that the resultant graph is easy to read,
            and adds a number of summary ops.
            """
            # Adding a name scope ensures logical grouping of the layers in the graph.
            with tf.name_scope(layer_name):
                # This Variable will hold the state of the weights for the layer
                with tf.name_scope('weights'):
                    weights = weight_variable([input_dim, output_dim],act=act)
                    variable_summaries(weights, layer_name + '/weights')
                with tf.name_scope('biases'):
                    # biases = bias_variable([output_dim])
                    biases = bias_variable([output_dim], fan_in_size=input_dim, act=act)
                    variable_summaries(biases, layer_name + '/biases')
                # if it is not the last layer
                if act is not None:
                    with tf.name_scope('Wx_plus_b'):
                        preactivate_s = tf.matmul(input_tensor_list[0], weights) + biases
                        tf.histogram_summary(layer_name + '/pre_activations_s', preactivate_s)
                        preactivate_s_prime = tf.matmul(input_tensor_list[1], weights) + biases
                        tf.histogram_summary(layer_name + '/pre_activations_s_prime', preactivate_s_prime)
                    activations_s = act(preactivate_s, 'activation_s')
                    tf.histogram_summary(layer_name + '/activations_s', activations_s)
                    activations_s_prime = act(preactivate_s_prime, 'activation_s_prime')
                    tf.histogram_summary(layer_name + '/activations_s_prime', activations_s_prime)
                    return (activations_s, activations_s_prime)
                # if it is the last layer
                else:
                    with tf.name_scope('Wx_plus_b'):
                        activations_s = tf.matmul(input_tensor_list[0], weights) + biases
                        tf.histogram_summary(layer_name + '/activations_s', activations_s)
                        activations_s_prime = tf.matmul(input_tensor_list[1], weights) + biases
                        tf.histogram_summary(layer_name + '/activations_s_prime', activations_s_prime)
                    return (activations_s, activations_s_prime)

        def nn_dual_layer_with_decay(input_tensor_list, input_dim, output_dim, layer_name, ema_ops_list, act=tf.nn.relu, decay=0.999):
            """Create a layer with two different inputs and two different outputs
               calculated with the same weights. Thought for Q_learning training.

            It does a matrix multiply, bias add, and then uses relu to nonlinearize.
            It also sets up name scoping so that the resultant graph is easy to read,
            and adds a number of summary ops.
            """
            # Adding a name scope ensures logical grouping of the layers in the graph.
            with tf.name_scope(layer_name):
                # This Variable will hold the state of the weights for the layer
                with tf.name_scope('weights'):
                    weights = weight_variable([input_dim, output_dim],act=act)
                    variable_summaries(weights, layer_name + '/weights')
                with tf.name_scope('biases'):
                    # biases = bias_variable([output_dim])
                    biases = bias_variable([output_dim], fan_in_size=input_dim, act=act)
                    variable_summaries(biases, layer_name + '/biases')
                # prime network uses a moving average of the main network
                ema = tf.train.ExponentialMovingAverage(decay=decay)
                ema_ops_list.append(ema.apply([weights,biases]))
                # if it is not the last layer
                if act is not None:
                    with tf.name_scope('Wx_plus_b'):
                        preactivate_s = tf.matmul(input_tensor_list[0], weights) + biases
                        tf.histogram_summary(layer_name + '/pre_activations_s', preactivate_s)
                        preactivate_s_prime = tf.matmul(input_tensor_list[1], ema.average(weights)) + ema.average(biases)
                        tf.histogram_summary(layer_name + '/pre_activations_s_prime', preactivate_s_prime)
                    activations_s = act(preactivate_s, 'activation_s')
                    tf.histogram_summary(layer_name + '/activations_s', activations_s)
                    activations_s_prime = act(preactivate_s_prime, 'activation_s_prime')
                    tf.histogram_summary(layer_name + '/activations_s_prime', activations_s_prime)
                    return (activations_s, activations_s_prime)
                # if it is the last layer
                else:
                    with tf.name_scope('Wx_plus_b'):
                        activations_s = tf.matmul(input_tensor_list[0], weights) + biases
                        tf.histogram_summary(layer_name + '/activations_s', activations_s)
                        activations_s_prime = tf.matmul(input_tensor_list[1], ema.average(weights)) + ema.average(biases)
                        tf.histogram_summary(layer_name + '/activations_s_prime', activations_s_prime)
                    return (activations_s, activations_s_prime)


        # NORMALIZE INPUT
        with tf.name_scope('input'):
            mean = tf.constant(normalization_mean, name='batch_mean')
            variance = tf.constant(normalization_var, name='batch_variance')
            # mean = tf.constant([0., 0.], name='batch_mean')
            # variance = tf.constant([1.0, 1.0], name='batch_variance')
            # need to be class variables to be able to make a feed_dict from other class functions
            # state s and next state s_prime
            self.s = tf.placeholder(tf.float32, shape = [None,size_input], name='s-input')
            # mean, variance = tf.nn.moments(self.s, [0])
            # self.s_norm = tf.nn.batch_normalization(self.s, mean, variance, None, None, 1e-3)
            self.s_norm = tf.nn.batch_normalization(self.s, mean, variance, None, None, 1e-8)

            self.s_prime = tf.placeholder(tf.float32, shape = [None,size_input], name='s_prime-input')
            # mean, variance = tf.nn.moments(self.s_prime, [0])
            # self.s_prime_norm = tf.nn.batch_normalization(self.s_prime, mean, variance, None, None, 1e-3)
            self.s_prime_norm = tf.nn.batch_normalization(self.s_prime, mean, variance, None, None, 1e-8)

            # self.mean = tf.Variable(tf.constant(0.0, shape=[size_input]), trainable=False)
            # variable_summaries(self.mean, layer_name + '/batch_mean')
            # self.variance = tf.Variabl(tf.constant(1.0, shape=[size_input]), trainable=False)
            # variable_summaries(self.variance, layer_name + '/batch_variance')

            # self.s = tf.placeholder(tf.float32, shape = [None,size_input], name='s-input')
            # self.s_norm, _ = batch_norm(self.s, size_input, 'batch_norm')
            # self.s_prime = tf.placeholder(tf.float32, shape = [None,size_input], name='s_prime-input')
            # self.s_prime_norm, _ = batch_norm(self.s_prime, size_input, 'batch_norm_prime')

            # self.s = tf.placeholder(tf.float32, shape = [None,size_input], name='s-input')
            # self.s_prime = tf.placeholder(tf.float32, shape = [None,size_input], name='s_prime-input')
            # self.s_norm, self.s_prime_norm = dual_batch_norm((self.s, self.s_prime), size_input, 'batch_norm')

            # reward
            self.r = tf.placeholder(tf.float32, shape = [None], name='reward')
            # action from state s to state s_prime
            self.a = tf.placeholder(tf.int32, shape = [None], name='action')
            a_one_hot = tf.one_hot(self.a, size_output, 1.0, 0.0)
            if self.is_a_prime_external:
                self.a_prime = tf.placeholder(tf.int32, shape = [None], name='action_prime')
                a_prime_one_hot = tf.one_hot(self.a_prime, size_output, 1.0, 0.0)

        # DROPOUT PROBABILITY
        self.keep_prob = tf.placeholder(tf.float32)

        # EXPONENTIAL MOVING AVERAGE FOR WEIGHTS
        ema_ops_list = []

        # HIDDEN LAYERS
        hidden_prev = nn_dual_layer_with_decay((self.s_norm, self.s_prime_norm), size_input, size_hidden[0], 'layer'+str(1), ema_ops_list, decay=ema_decay_rate)
        # with tf.name_scope('dropout'):
        #     hidden_prev = tf.nn.dropout(hidden_prev[0], self.keep_prob), tf.nn.dropout(hidden_prev[1], self.keep_prob)
        for idx in range(1, len(size_hidden) ):
            hidden = nn_dual_layer_with_decay(hidden_prev, size_hidden[idx-1], size_hidden[idx], 'layer'+str(idx+1), ema_ops_list, decay=ema_decay_rate)
            # with tf.name_scope('dropout'):
            #     dropped = tf.nn.dropout(hidden[0], self.keep_prob), tf.nn.dropout(hidden[1], self.keep_prob)
            # hidden_prev = dropped
            hidden_prev = hidden

        # self.is_a_prime_external = tf.placeholder(tf.bool)
        # q_all contains all the q values for all the actions

        # OUTPUT LAYERS
        with tf.name_scope('output'):
            self.is_not_end_state = tf.placeholder(tf.float32, shape=[None], name='is_end_state_prime')
            self.q_all, q_all_prime = nn_dual_layer_with_decay(hidden_prev, size_hidden[-1], size_output, 'layer'+str(len(size_hidden)+1), ema_ops_list, act=None, decay=ema_decay_rate)
            if self.is_a_prime_external:
                # SARSA
                q_prime = tf.reduce_sum(tf.mul(q_all_prime, a_prime_one_hot), reduction_indices=[1])
            else:
                # Q-learning
                print('Q-learning configured in nn graph')
                q_prime = tf.stop_gradient(tf.reduce_max(q_all_prime, reduction_indices=[1]))
            q_target = discount*q_prime*self.is_not_end_state + self.r
            tf.histogram_summary('q_target', q_target)
            q_a = tf.reduce_sum(tf.mul(self.q_all, a_one_hot), reduction_indices=[1])

        with tf.name_scope('mean_squares_error'):
            mean_squares_error = tf.squeeze(tf.reduce_sum((q_target-q_a)**2) / tf.to_float(tf.shape(q_target)[0]))
            tf.scalar_summary('log_mean_squares_error', tf.log(mean_squares_error))

        with tf.name_scope('train'):
            self.learning_rate_value = learning_rate
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
            # self.learning_rate = tf.constant(learning_rate)
            print('descending with ', descent_method)
            if descent_method == 'adam':
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(mean_squares_error)
            elif descent_method == 'grad':
                self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(mean_squares_error)
            elif descent_method == 'rmsprop':
                self.train_step = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.95, momentum=0.95, epsilon=1e-2).minimize(mean_squares_error)


        self.summary_op = tf.merge_all_summaries()

        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        self.sess.run(tf.initialize_all_variables())
        self.summary_writer = tf.train.SummaryWriter('/tmp/tfdata/'+env_name, self.sess.graph)

        self.training_steps_count = 0
        self.samples_count = 0

        # self.simple_batch_list = []
        self.replay_memory = deque(maxlen=replay_memory_size)






    # def train_single_example(self, s, a, r, s_prime):
    #     feed_dict = {self.s: s, self.a: a.astype(np.int32), self.r: r, self.s_prime: s_prime}
    #
    #     # every 10th iteration, save a summary
    #     if self.training_steps_count % 10 == 0:
    #         # summary, _ = self.sess.run([self.summary_op, self.train_step], feed_dict=feed_dict)
    #         summary, _ = self.sess.run([self.summary_op, self.train_step], feed_dict=feed_dict)
    #         self.summary_writer.add_summary(summary, self.training_steps_count)
    #     # run train only, without summary
    #     else:
    #         self.sess.run(self.train_step, feed_dict=feed_dict)
    #
    #     self.training_steps_count += 1
    #
    #
    #
    #
    #
    # def train_wait_for_batch(self, s, a, r, s_prime):
    #
    #     if len(self.simple_batch_list) == self.batch_size:
    #         states, actions, rewards, states_prime = zip(*self.simple_batch_list)
    #         states = np.squeeze(np.array(states))
    #         actions = np.squeeze(np.array(actions))
    #         rewards = np.squeeze(np.array(rewards))
    #         states_prime = np.squeeze(np.array(states_prime))
    #
    #         feed_dict = {self.s: states, self.a: actions, self.r: rewards, self.s_prime: states_prime}
    #         # train and write summary
    #         summary, _ = self.sess.run([self.summary_op, self.train_step], feed_dict=feed_dict)
    #         self.summary_writer.add_summary(summary, self.training_steps_count)
    #
    #         # reset batch list
    #         self.simple_batch_list = []
    #     else:
    #         # fill list
    #         self.simple_batch_list.append((s,a,r,s_prime))
    #         # print(self.simple_batch_list)
    #     # feed_dict = {self.s: s, self.a: a.astype(np.int32), self.r: r, self.s_prime: s_prime}
    #
    #     self.training_steps_count += 1





    def train_batch(self, s, a, r, s_prime, done, a_prime=None, learning_rate=None, train_frequency = 1.0):
        if learning_rate is None:
            learning_rate = self.learning_rate_value

        if (len(self.replay_memory) >= 5e4) and (self.samples_count % (self.batch_size/train_frequency) == 0 or self.do_train_every_sample):
            # fill list
            self.replay_memory.append((s,a,r,s_prime,a_prime,done))

            # sample random batch
            sample_idx = np.random.randint(0, len(self.replay_memory), [self.batch_size])
            batch = [self.replay_memory[i] for i in sample_idx]

            # train
            states, actions, rewards, states_prime, actions_prime, done = zip(*batch)
            states = np.squeeze(np.array(states))
            actions = np.squeeze(np.array(actions))
            rewards = np.squeeze(np.array(rewards))
            states_prime = np.squeeze(np.array(states_prime))
            actions_prime = np.squeeze(np.array(actions_prime))
            done = np.array(done)

            feed_dict = {self.s: states, self.a: actions, self.r: rewards, self.s_prime: states_prime,
                         self.is_not_end_state: 1.0 - done,
                         self.keep_prob: self.keep_prob_val,
                         self.phase_train: True,
                         self.learning_rate: learning_rate}
            if self.is_a_prime_external:
                feed_dict[self.a_prime] = actions_prime
                # print(feed_dict)

            # train and write summary
            if self.training_steps_count % 100 == 0:
                summary, _ = self.sess.run([self.summary_op, self.train_step], feed_dict=feed_dict)
                self.summary_writer.add_summary(summary, self.training_steps_count)
            else:
                self.sess.run(self.train_step, feed_dict=feed_dict)


            self.training_steps_count += 1
        else:
            # fill list
            self.replay_memory.append((s,a,r,s_prime,a_prime,done))
            # print(self.simple_batch_list)
        # feed_dict = {self.s: s, self.a: a.astype(np.int32), self.r: r, self.s_prime: s_prime}

        self.samples_count += 1






    def evaluate_all_actions(self, states):
        # print(states, states.shape)
        feed_dict = {self.s: states, self.keep_prob: 1.0, self.phase_train: False}
        return self.sess.run(self.q_all, feed_dict=feed_dict)

    # this does not seem to work
    def clear(self):
        self.sess.close()
        tf.reset_default_graph()
