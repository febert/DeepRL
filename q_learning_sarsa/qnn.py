from __future__ import print_function

import tensorflow as tf
import numpy as np

class qnn:

    def __init__(self,
                 size_input,
                 size_output,
                 size_hidden = [10,10],
                 descent_method = 'adam',
                 learning_rate = 1e-4,
                 batch_size = 1,
                 replay_memory = False):

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
                        tf.histogram_summary(layer_name + '/pre_activations_s', activations_s)
                        activations_s_prime = tf.matmul(input_tensor_list[1], weights) + biases
                        tf.histogram_summary(layer_name + '/pre_activations_s_prime', activations_s_prime)
                    return (activations_s, activations_s_prime)

        with tf.name_scope('input'):
            # need to be self variables to be able to make a feed_dict from other class functions
            # state s and next state s_prime
            self.s = tf.placeholder(tf.float32, shape = [None,size_input], name='s-input')
            self.s_prime = tf.placeholder(tf.float32, shape = [None,size_input], name='s_prime-input')
            # reward
            self.r = tf.placeholder(tf.float32, shape = [None], name='reward')
            # action from state s to state s_prime
            self.a = tf.placeholder(tf.int64, shape = [None], name='action')
            a_one_hot = tf.one_hot(self.a, size_output, 1.0, 0.0)

# dual activation version
        hidden_prev = nn_dual_layer((self.s, self.s_prime), size_input, size_hidden[0], 'layer'+str(1) )
        for idx in range(1, len(size_hidden) ):
            hidden = nn_dual_layer(hidden_prev, size_hidden[idx-1], size_hidden[idx], 'layer'+str(idx+1))
            hidden_prev = hidden
            # no dropout for now

        # q_all contains all the q values for all the actions
        with tf.name_scope('output'):
            q_all, q_all_prime = nn_dual_layer(hidden, size_hidden[-1], size_output, 'layer'+str(len(size_hidden)+1), act=None)
            q_prime_max = tf.stop_gradient(tf.reduce_max(q_all_prime, reduction_indices=[1]))
            q_target = q_prime_max + self.r
            q_a = tf.matmul(q_all, tf.transpose(a_one_hot))


# single activation version
        # hidden_prev = nn_layer(self.x, size_input, size_hidden[0], 'layer'+str(1) )
        # for idx in range(1, len(size_hidden) ):
        #     hidden = nn_layer(hidden_prev, size_hidden{idx-1}, size_hidden[idx], 'layer'+str(idx+1))
        #     hidden_prev = hidden
        #     # no dropout for now
        #
        # self.q = nn_layer(hidden, size_hidden[-1], size_output, 'layer'+str(len(siz_hidden)+1), act=None)
        # self.q_max = tf.reduce_max(q, reduction_indices=[1])
        # # self.q_max_no_grad = tf.stop_gradient(q_max)

        with tf.name_scope('mean_square_error'):
            mean_square_error = tf.reduce_sum((q_target-q_a)**2, reduction_indices=[0])
            tf.scalar_summary('mean_square_error', mean_square_error)

        with tf.name_scope('train'):
            # self.learning_rate = tf.placeholder(tf.float32, shape=[])
            self.learning_rate = tf.constant(learning_rate)
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(mean_square_error)

        self.summary_op = tf.merge_all_summaries()

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter('./data', self.sess.graph)

    def train_single_example(self, s, a, r, s_prime):
        feed_dict = {self.s: s, self.a: a.astype(np.int64), self.r: r, self.s_prime: s_prime}
        self.sess.run(self.train_step, feed_dict=feed_dict)
