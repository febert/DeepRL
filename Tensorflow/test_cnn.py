import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


sess = tf.InteractiveSession()

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

learning_rate = tf.placeholder(tf.float32, shape=[])

global_step = tf.Variable(0, name='global_step', trainable=False)

# losses
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]), name= 'cross_entropy')

# summaries
tf.scalar_summary(cross_entropy.op.name, cross_entropy)
tf.histogram_summary('W_conv1',W_conv1)
tf.histogram_summary('b_conv1',b_conv1)
tf.histogram_summary('W_conv2',W_conv2)
tf.histogram_summary('b_conv2',b_conv2)
tf.histogram_summary('W_fc1',W_fc1)
tf.histogram_summary('b_fc1',b_fc1)

tf.histogram_summary('h_pool1',h_pool1)
tf.histogram_summary('h_pool2',h_pool2)
tf.histogram_summary('h_fc1',h_fc1)


# training
# train_step = tf.train.GradientDescentOptimizer(1e-6).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#     back_train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(reconstruction_error)

# evaluation
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


########## INIT AND SUMMARY HAVE TO APPEAR HERE AT THE END TO BE PROPERLY DEFINED
# misc
init = tf.initialize_all_variables()
summary_op = tf.merge_all_summaries()


summary_writer = tf.train.SummaryWriter('/tmp/cnn', sess.graph)

t_elapsed = 0.

sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        print('time for one sgd step', t_elapsed)
    #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    t_loopstart = time.clock()
    sumry , _ = sess.run([summary_op, train_step], feed_dict= {x: batch[0], y_: batch[1], keep_prob: 0.5})
    t_elapsed = time.clock() - t_loopstart

    if i%10 ==0:
        summary_writer.add_summary(sumry,i)
        summary_writer.flush()

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))