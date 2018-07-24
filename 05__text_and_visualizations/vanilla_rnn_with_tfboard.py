# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 17:34:43 2016

@author: tomhope

RM:
    1. Explanatory notes and comments.
    2. Purpose_indicator naming
"""
#from __future__ import print_function
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/rm/tmp/data/", one_hot=True)

# Define some parameters
no_of_elements_per_scanline = 28
no_of_scanlines_per_image = 28
no_of_classes = 10
batch_size = 128

NUM_OF_TRG_ITERS = 3000
CHECK_EVERY_ITER_NUM = 100

#no_of_hidden_layers is a misnomer. 
#The number gives the number of dimensions 
#(the dimensionality) of the state vector. 
#The state vector is taken as a row vector. The no of 
#components of this vector equals no_of_hidden_layers.
#More appropriate name is state_vector_dimensionality.
#
#The dimensionality is the number of ***independent*** features that the
#convolutional layer can detect

#<QUOTE>
#... in typical CNN models we stack convolutional layers hierarchically, 
#and feature map is simply a commonly used term referring to the output 
#of each such layer. Another way to view the output of these layers 
#is as processed images, the result of applying a filter and perhaps 
#some other operations. Here, ***this filter*** is parameterized by W, the 
#learned weights of our network representing the convolution filter.
#</QUOTE> (emphasis mine)
#
#From: "Learning TensorFlow: A Guide to Building Deep Learning Systems" 
#(Kindle Locations 1378-1382). 

#<QUOTE>
#This means that all the neurons in the first hidden layer will recognize 
#the same features, just placed differently in the input image. 
#For this reason, the map of connections from the input layer to the 
#hidden feature map ... . Obviously, we need to recognize an image of 
#more than a map of features, so *** a *** complete convolutional layer 
#is made from *** multiple *** feature maps.
#</QUOTE>
#
#From:G"etting Started with TensorFlow 
#(Kindle Locations 2065-2066). 

#no_of_hidden_layers = 128
state_vector_dimensionality = 128//2

# Where to save TensorBoard model summaries
LOG_DIR = "/home/rm/logs/RNN_with_summaries/"

# Create placeholders for inputs, labels
_input_images = tf.placeholder(tf.float32,
                         shape=[None, \
                                no_of_scanlines_per_image, \
                                no_of_elements_per_scanline],
                         name='inputs')
labels_true = tf.placeholder(tf.float32, shape=[None, \
                                      no_of_classes], \
                                name='labels')

########################################################################
# 
# This helper function taken from official TensorFlow documentation,
# simply add some ops that take care of logging summaries
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# Weights and bias for input and hidden layer
with tf.name_scope('rnn_weights'):
    with tf.name_scope("W_x"):
#        Wx is the Weight matrix that post-multiplies the input vector.
#        Each input is a line of pixels. It is thus a row vector having
#        a dimensionality of no_of_elements_per_scanline.
#        Thus post multiplying it with Wx produces a row matrix with
#        no of elements equal to state_vector_dimensionality.
#        This resultant row matrix is the incremental change in the state 
#        vector state caused by the current input. It is to be added to
#        time-evolved state vector. That evolution is described next.
        Wx = tf.Variable(tf.zeros([no_of_elements_per_scanline, \
                                   state_vector_dimensionality]))
        variable_summaries(Wx)
        
    with tf.name_scope("W_h"):
#        Wh is the weight matrix that post multiplies the state vector 
#        (a row vector). This results in the time evolution of the state vector
#        to the current instatnt of time.
#        As mentioned the state vector is a row vector having elements/components
#        whose number equals state_vector_dimensionality. So we need a 
#        transformation matrix that takes the input vector and combines its 
#        components in linearly independent(?) manner resulting in another
#        vector having the same number of elements/components. This transformation 
#        matrix, therefore, is a 2-D square matrix of 
#        shape [state_vector_dimensionality, state_vector_dimensionality]
        Wh = tf.Variable(tf.zeros([state_vector_dimensionality, \
                                   state_vector_dimensionality]))
        variable_summaries(Wh)
        
    with tf.name_scope("Bias"):
        b_rnn = tf.Variable(tf.zeros([state_vector_dimensionality]))
        variable_summaries(b_rnn)

#The above lot of name_scopes are not required by tensorflow for correctness
#of the computation, nor is it required for logging. It is needed for
#creating a heirarchy in the tensorboard display.
# NOTE the terminating tf.summary.* and variable_summaries()
########################################################################
#NOTE: No scoping here
# Processing inputs to work with scan function
# Current input shape: (batch_size, no_of_scanlines_per_image, no_of_elements_per_scanline)
transposed_input = tf.transpose(_input_images, perm=[1, 0, 2])
initial_hidden_state = tf.zeros([batch_size, \
                                 state_vector_dimensionality])

def rnn_step(previous_hidden_state, x):
        current_hidden_state = tf.tanh(
            tf.matmul(previous_hidden_state, Wh) +
            tf.matmul(x, Wx) + b_rnn)
        return current_hidden_state


# Getting all state vectors ***across time***
#This is the function that scans the images along the vertical axis
#line-by-line. The vertical axis is along dim = 0 in the transposed
#input. The scanning presents the sequence of vertical scans. The
#length of the sequence is therefore no_of_scanlines_per_image 
#(see assert below). For the selected element of the 
#scan sequence - the time-step -
#the applicable row of pixels, for the ***complete *** batch is 
#presented to rnn_step. 
#So at each point of the sequence of the vertical scan -
# - a time-step - the input is a matrix of dimension 
#[batch_size, no_of_elements_per_scanline]. Each row of the matrix has the
#pixels for the scan at a particular scan-step for the complete batch.
#SEE BELOW for assertions on all_hidden_states.
        
all_hidden_states = tf.scan(rnn_step,
                            transposed_input,
                            initializer=initial_hidden_state,
                            name='all_hidden_states') #'states')
#########################################################################
# These we would like to see as bundled together in heirarchies
# Weights for output layers
with tf.name_scope('linear_layer_weights') as scope:
    with tf.name_scope("W_linear"):
        Wl = tf.Variable(tf.truncated_normal([state_vector_dimensionality, \
                                              no_of_classes],
                                             mean=0, stddev=.01))
        variable_summaries(Wl)
    with tf.name_scope("Bias_linear"):
        bl = tf.Variable(tf.truncated_normal([no_of_classes],
                                             mean=0, stddev=.01))
        variable_summaries(bl)

# NOTE the terminating variable_summaries()
#########################################################################
#This function needs to be sandwiched here.
        
# Apply linear layer to state vector
def get_linear_layer(hidden_state):

    return tf.matmul(hidden_state, Wl) + bl


#########################################################################
#NOTE: The name scope "linear_layer_weights" continues
with tf.name_scope('linear_layer_weights') as scope:
    # Iterate across time, apply linear layer to all RNN outputs
    all_outputs = tf.map_fn(get_linear_layer, \
                            all_hidden_states,\
                            name='FilterOPofAllHiddenStages')
    # Get Last output -- h_28
    output = all_outputs[-1]
    tf.summary.histogram('outputs', output)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(\
                       tf.nn.softmax_cross_entropy_with_logits_v2(\
                                               logits=output, labels=labels_true))
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    # Using RMSPropOptimizer
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(labels_true, 1), tf.argmax(output, 1))

    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100
    tf.summary.scalar('accuracy', accuracy)

#That's the last of name spaces
# NOTE the terminating tf.summary.*()
##########################################################################
# Merge all the summaries
merged = tf.summary.merge_all()
##########################################################################
#Re-factored out the bunch of assertions of my understanding of the code.
#Refactored to tf.Assert.py
##########################################################################
#In case code is inserted above that reads batches from the data file,
#the counts will need to be reset so that training can process from 
#the start of the file    
    
mnist.train.reset_counts()

# Get a small test set
test_data = mnist.test.images[:batch_size].reshape((-1, \
                             no_of_scanlines_per_image, \
                             no_of_elements_per_scanline))
test_label = mnist.test.labels[:batch_size]

try:
    with tf.Session() as sess:
        # Write summaries to LOG_DIR -- used by TensorBoard
        train_writer = tf.summary.FileWriter(LOG_DIR + '/train',
                                             graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(LOG_DIR + '/test',
                                            graph=tf.get_default_graph())
    
        sess.run(tf.global_variables_initializer())

        for i in range(NUM_OF_TRG_ITERS):
    
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Reshape data to get 28 sequences of 28 pixels
                batch_x = batch_x.reshape((batch_size, \
                                           no_of_scanlines_per_image, \
                                           no_of_elements_per_scanline))
                summary, _ = sess.run([merged, train_step],
                                      feed_dict={_input_images: batch_x, \
                                                 labels_true: batch_y})
                # Add to summaries
                train_writer.add_summary(summary, i)
    
                if i % CHECK_EVERY_ITER_NUM == 0:
                    acc, loss, = sess.run([accuracy, cross_entropy],
                                          feed_dict={_input_images: batch_x,
                                                     labels_true: batch_y})
                    print("Iter " + str(i) + ", Minibatch Loss= " +
                          "{:.6f}".format(loss) + ", Training Accuracy= " +
                          "{:.5f}".format(acc))
                if i % CHECK_EVERY_ITER_NUM == 0:
                    # Calculate accuracy for 128 mnist test images and
                    # add to summaries
                    summary, acc = sess.run([merged, accuracy],
                                            feed_dict={_input_images: test_data,
                                                       labels_true: test_label})
                    test_writer.add_summary(summary, i)
    
        test_acc = sess.run(accuracy, feed_dict={_input_images: test_data,
                                                 labels_true: test_label})
        print("Test Accuracy:", test_acc)
    
finally:
    #This is needed only when the file is run in spyder.
    #A re-run will cause an exception.
    #If run from the command line there is no problem
    print("Exiting from finally in: ", __file__)
    tf.reset_default_graph()
