from __future__ import print_function

import tensorflow as tf
import numpy as np

class qnn:

    def __init__(self,
                 size_input,
                 size_output,
                 size_hidden = [300,400,400],
                #  descent_method = 'adam',
                 descent_method = 'grad',
                 learning_rate = 1e-6,
                 batch_size = 50,
                 replay_memory = False):

        self.batch_size = batch_size

        def weight_variable(shape):
            """
            Create a weight variable with appropriate initialization.
            Random to break symmetries.
            """
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            """
            Create a bias variable with appropriate initialization.
            Larger than zero to avoid dead ReLUs
            """
            initial = tf.constant(0.1, shape=shape)
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
                    weights = weight_variable([input_dim, output_dim])
                    variable_summaries(weights, layer_name + '/weights')
                with tf.name_scope('biases'):
                    biases = bias_variable([output_dim])
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
                    weights = weight_variable([input_dim, output_dim])
                    variable_summaries(weights, layer_name + '/weights')
                with tf.name_scope('biases'):
                    biases = bias_variable([output_dim])
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


        with tf.name_scope('input'):
            mean = tf.constant([-0.3, 0], name='batch_mean')
            variance = tf.constant([0.81, 0.0049], name='batch_variance')
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


            # reward
            self.r = tf.placeholder(tf.float32, shape = [None], name='reward')
            # action from state s to state s_prime
            self.a = tf.placeholder(tf.int64, shape = [None], name='action')
            a_one_hot = tf.one_hot(self.a, size_output, 1.0, 0.0)

        self.keep_prob = tf.placeholder(tf.float32)
# dual activation version
        hidden_prev = nn_dual_layer((self.s_norm, self.s_prime_norm), size_input, size_hidden[0], 'layer'+str(1) )
        with tf.name_scope('dropout'):
            hidden_prev = tf.nn.dropout(hidden_prev[0], self.keep_prob), tf.nn.dropout(hidden_prev[1], self.keep_prob)
        for idx in range(1, len(size_hidden) ):
            hidden = nn_dual_layer(hidden_prev, size_hidden[idx-1], size_hidden[idx], 'layer'+str(idx+1))
            with tf.name_scope('dropout'):
                dropped = tf.nn.dropout(hidden[0], self.keep_prob), tf.nn.dropout(hidden[1], self.keep_prob)
            hidden_prev = dropped
            # no dropout for now

        # q_all contains all the q values for all the actions
        with tf.name_scope('output'):
            self.q_all, q_all_prime = nn_dual_layer(hidden, size_hidden[-1], size_output, 'layer'+str(len(size_hidden)+1), act=None)
            q_prime_max = tf.stop_gradient(tf.reduce_max(q_all_prime, reduction_indices=[1]))
            q_target = q_prime_max + self.r
            q_a = tf.reduce_sum(tf.mul(self.q_all, a_one_hot), reduction_indices=[1])

        with tf.name_scope('mean_squares_error'):
            mean_squares_error = tf.squeeze(tf.reduce_sum((q_target-q_a)**2))
            tf.scalar_summary('mean_squares_error', mean_squares_error)

        with tf.name_scope('train'):
            # self.learning_rate = tf.placeholder(tf.float32, shape=[])
            self.learning_rate = tf.constant(learning_rate)
            if descent_method == 'adam':
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(mean_squares_error)
            elif descent_method == 'grad':
                self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(mean_squares_error)

        self.summary_op = tf.merge_all_summaries()

        self.sess = tf.Session()

        self.sess.run(tf.initialize_all_variables())
        self.summary_writer = tf.train.SummaryWriter('./data', self.sess.graph)

        self.training_steps_count = 0

        self.simple_batch_list = []






    def train_single_example(self, s, a, r, s_prime):
        feed_dict = {self.s: s, self.a: a.astype(np.int64), self.r: r, self.s_prime: s_prime}

        # every 10th iteration, save a summary
        if self.training_steps_count % 10 == 0:
            # summary, _ = self.sess.run([self.summary_op, self.train_step], feed_dict=feed_dict)
            summary, _ = self.sess.run([self.summary_op, self.train_step], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, self.training_steps_count)
        # run train only, without summary
        else:
            self.sess.run(self.train_step, feed_dict=feed_dict)

        self.training_steps_count += 1





    def train_wait_for_batch(self, s, a, r, s_prime):

        if len(self.simple_batch_list) == self.batch_size:
            states, actions, rewards, states_prime = zip(*self.simple_batch_list)
            states = np.squeeze(np.array(states))
            actions = np.squeeze(np.array(actions))
            rewards = np.squeeze(np.array(rewards))
            states_prime = np.squeeze(np.array(states_prime))

            feed_dict = {self.s: states, self.a: actions, self.r: rewards, self.s_prime: states_prime}
            # train and write summary
            summary, _ = self.sess.run([self.summary_op, self.train_step], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, self.training_steps_count)

            # reset batch list
            self.simple_batch_list = []
        else:
            # fill list
            self.simple_batch_list.append((s,a,r,s_prime))
            # print(self.simple_batch_list)
        # feed_dict = {self.s: s, self.a: a.astype(np.int64), self.r: r, self.s_prime: s_prime}

        self.training_steps_count += 1





    def train_batch(self, s, a, r, s_prime):

        if (len(self.simple_batch_list) >= 10*self.batch_size) and (len(self.simple_batch_list) % self.batch_size == 0):
            # fill list
            self.simple_batch_list.append((s,a,r,s_prime))

            # sample random batch
            sample_idx = np.random.randint(0, len(self.simple_batch_list), [self.batch_size])
            batch = [self.simple_batch_list[i] for i in sample_idx]

            # train
            states, actions, rewards, states_prime = zip(*batch)
            states = np.squeeze(np.array(states))
            actions = np.squeeze(np.array(actions))
            rewards = np.squeeze(np.array(rewards))
            states_prime = np.squeeze(np.array(states_prime))

            feed_dict = {self.s: states, self.a: actions, self.r: rewards, self.s_prime: states_prime, self.keep_prob: 0.9}
            # train and write summary
            summary, _ = self.sess.run([self.summary_op, self.train_step], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, self.training_steps_count)
        else:
            # fill list
            self.simple_batch_list.append((s,a,r,s_prime))
            # print(self.simple_batch_list)
        # feed_dict = {self.s: s, self.a: a.astype(np.int64), self.r: r, self.s_prime: s_prime}

        self.training_steps_count += 1





    def evaluate_all_actions(self, states):
        # print(states, states.shape)
        feed_dict = {self.s: states, self.keep_prob: 1.0}
        return self.sess.run(self.q_all, feed_dict=feed_dict)

    # this does not seem to work
    def clear(self):
        self.sess.close()
        tf.reset_default_graph()
