
"""
Also see the header coments in
/home/rm/Code-GettingStartedWithTF/Chapter 5/convolutional_network.py
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

from layers import conv_layer, max_pool_2x2, full_layer

DATA_DIR = '/home/rm/tmp/data'
MINIBATCH_SIZE = 50
STEPS = 5000

#For the mnist object in the following statement see
#"/home/rm/anaconda3/envs/tensorflow/lib/python3.6/site-packages/tensorflow/" + \
#"contrib/learn/python/learn/datasets/mnist.py"

mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_actual = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])
conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_detected = full_layer(full1_drop, 10)

cost = \
    tf.reduce_mean(\
           tf.nn.softmax_cross_entropy_with_logits_v2(\
                                              logits=y_detected, labels=y_actual))
#The step that adjust the weights and biases
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

#Measure how many predictions were  correct
#These nodes play NO role in the training of the model
correct_prediction = tf.equal(tf.argmax(y_detected, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
     sess.run(init)
     batch = mnist.train.next_batch(MINIBATCH_SIZE)

#     print("x.shape: {}".format(x.shape))
#     print("batch_xs.shape: {}".format(batch_xs.shape))
# =============================================================================
# _X = tf.reshape(x, shape=[-1, 28, 28, 1])
     x1 = sess.run(x_image, \
                  feed_dict={x: batch[0], \
                             y_actual: batch[1],\
                             keep_prob: 1.0})
     assert(MINIBATCH_SIZE == x1.shape[0])
     assert(28 == x1.shape[1] and \
            28 == x1.shape[2] and \
            1 == x1.shape[3])


mnist.train.reset_counts()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #Run the UNTRAINED model on the training data
    for i in range(STEPS):
        batch = mnist.train.next_batch(MINIBATCH_SIZE)

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, \
                                      feed_dict={x: batch[0], \
                                                 y_actual: batch[1],\
                                                 keep_prob: 1.0})
            print("step {}, training accuracy {}".format(i, train_accuracy))

        sess.run(train_step, feed_dict={x: batch[0], \
                                        y_actual: batch[1], \
                                        keep_prob: 0.5})

    #Now run the TRAINED model on the TEST data
    #First get the training data
    X = mnist.test.images.reshape(10, 1000, 784)
    Y = mnist.test.labels.reshape(10, 1000, 10)
    test_accuracy = np.mean(
        [sess.run(accuracy, feed_dict={x: X[i], \
                                       y_actual: Y[i], \
                                       keep_prob: 1.0}) for i in range(10)])

print("test accuracy: {}".format(test_accuracy))
