# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 17:34:43 2016

@author: tomhope
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/rm/tmp/data/", one_hot=True)

no_of_elements_per_step = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

_inputs = tf.placeholder(tf.float32,
                         shape=[None, time_steps, no_of_elements_per_step],
                         name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes], \
                       name='inputs') #Typo?

Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes],
                                     mean=0, stddev=.01))
bl = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))


#######################################################
#NOTE: No name_scopes
#######################################################

#TensorFlow’s RNN cells are abstractions that represent the basic operations 
#each recurrent “cell” carries out ... and its associated state. T
#hey are, in general terms, a “replacement” of the rnn_step() function 
#and the associated variables it required. 
#Of course, there are many variants and types of cells, 
#each with many methods and properties.
#
#(Kindle Locations 2421-2425). O'Reilly Media. Kindle Edition. 
#
rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)

#tf.nn.dynamic_rnn replaces tf.scan() in our vanilla implementation 
#(vanilla_rnn_with_tfboard.py) and creates an RNN specified by rnn_cell.
#(Kindle Locations 2427-2428). 
#
#TensorFlow includes a static and a dynamic function for creating an RNN. 
#What does this mean? The static version creates an unrolled graph 
#...   of fixed length. The dynamic version uses a tf.While loop to 
#dynamically construct the graph at execution time,
#(Kindle Locations 2429-2432).  
#
outputs, _ = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)

def get_linear_layer(vector):
    return tf.matmul(vector, Wl) + bl

########################################################
last_rnn_output = outputs[:, -1, :]
final_output = get_linear_layer(last_rnn_output)

softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_output, labels=y)
cross_entropy = tf.reduce_mean(softmax)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
########################################################
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

test_data = mnist.test.images[:batch_size].reshape(
    (-1, time_steps, no_of_elements_per_step))
test_label = mnist.test.labels[:batch_size]

for i in range(3001):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, time_steps, no_of_elements_per_step))
    sess.run(train_step, feed_dict={_inputs: batch_x,
                                    y: batch_y})
    if i % 1000 == 0:
        acc = sess.run(accuracy, feed_dict={_inputs: batch_x,
                                            y: batch_y})
        loss = sess.run(cross_entropy, feed_dict={_inputs: batch_x,
                                                  y: batch_y})
        print("Iter " + str(i) + ", Minibatch Loss= " +
              "{:.6f}".format(loss) + ", Training Accuracy= " +
              "{:.5f}".format(acc))

print("Testing Accuracy:", sess.run(
    accuracy, feed_dict={_inputs: test_data, y: test_label}))
