# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 17:34:43 2016

@author: tomhope
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/rm/tmp/data/", one_hot=True)

# Define some parameters
no_of_elements_per_scanline = 28
no_of_scanlines_per_image = 28
no_of_classes = 10
batch_size = 128

#no_of_hidden_layers is a misnomer. 
#The number gives the number of dimensions 
#(the dimensionality) of the state vector. 
#The state vector is taken as a row vector. The no of 
#components of this vector equals no_of_hidden_layers.
#More appropriate name is state_vector_dimensionality

#no_of_hidden_layers = 128
state_vector_dimensionality = 128//2

# Where to save TensorBoard model summaries
LOG_DIR = "/home/rm/logs/RNN_with_summaries/"

# Create placeholders for inputs, labels
_inputs = tf.placeholder(tf.float32,
                         shape=[None, \
                                no_of_scanlines_per_image, \
                                no_of_elements_per_scanline],
                         name='inputs')
y = tf.placeholder(tf.float32, shape=[None, \
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
transposed_input = tf.transpose(_inputs, perm=[1, 0, 2])
# Current input shape now: (no_of_scanlines_per_image,batch_size, no_of_elements_per_scanline)
assert_transposed = tf.Assert((no_of_scanlines_per_image + 1 == transposed_input.shape[0] and
               batch_size == transposed_input.shape[1] and
               no_of_elements_per_scanline == transposed_input.shape[2]), \
            [y], name="assert_transposed")

initial_hidden = tf.zeros([batch_size, state_vector_dimensionality])

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
                            initializer=initial_hidden,
                            name='all_hidden_states') #'states')

def true_fn():
    return (all_hidden_states)

def false_fn():
    z = tf.transpose(all_hidden_states,perm=[1,0,2])
    return (z)

cond_T_all_hidden_states = tf.cond((tf.convert_to_tensor(1 < 2)), \
                                 true_fn, \
                                 false_fn, \
                                 name="cond_T_all_hidden_states")
cond_F_all_hidden_states = tf.cond((tf.convert_to_tensor(1 < 0)), \
                                 true_fn, \
                                 false_fn, \
                                 name="cond_F_all_hidden_states")
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
                                               logits=output, labels=y))
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    # Using RMSPropOptimizer
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))

    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100
    tf.summary.scalar('accuracy', accuracy)

#That's the last of name spaces
# NOTE the terminating tf.summary.*()
##########################################################################
# Merge all the summaries
merged = tf.summary.merge_all()

try:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        pixel_val = 1.0E+04
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        assert(batch_size == batch_x.shape[0] and
               (no_of_scanlines_per_image * \
               no_of_elements_per_scanline) == batch_x.shape[1])
        assert(batch_size == batch_y.shape[0] and
               no_of_classes == batch_y.shape[1])
        
        batch0_pixel0_row1 = batch_x[0, no_of_elements_per_scanline]
        print("batch0_pixel0_row1: ", batch0_pixel0_row1)
        batch_x[0, no_of_elements_per_scanline] = pixel_val
        assert(pixel_val == batch_x[0, no_of_elements_per_scanline])
        # Reshape data to get 28 sequences of 28 pixels
        batch_x = batch_x.reshape(batch_size, \
                                   no_of_scanlines_per_image, \
                                   no_of_elements_per_scanline)
        assert(batch_size == batch_x.shape[0] and
               no_of_scanlines_per_image == batch_x.shape[1] and
               no_of_elements_per_scanline == batch_x.shape[2])

        pixel_no = 0; scanline_no = 1
        print("batch0_pixel0_row1: ", batch_x[0, scanline_no, pixel_no])
        assert(pixel_val == batch_x[0, scanline_no, pixel_no])

        x = sess.run(transposed_input,
                              feed_dict={_inputs: batch_x, \
                                         y: batch_y})
        assert(no_of_scanlines_per_image == x.shape[0] and
               batch_size == x.shape[1] and
               no_of_elements_per_scanline == x.shape[2])

#        sess.run(assert_transposed, \
#                 feed_dict={_inputs: batch_x, \
#                                              y: batch_y})

#        The transposed input - dimensions as asserted above - is 
#        tf.scan'ed and all_hidden_states built step-by-step. 
#        
#        At each step 
#            the hidden state at the previous step is multiplied by Wh. 
#            Wh has dimensions of [state_vector_dimensionality, state_vector_dimensionality])). 
#            
#            The first step uses a previous hidden state initialized as 
#            zeros[batch_size, state_vector_dimensionality]). The result is 
#            [batch_size, state_vector_dimensionality]
#            
#            The scanned input (from tf.scan) has dimension of 
#            [batch_size, no_of_elements_per_scanline]. This is multiplied 
#            by Wx which has a dimension
#            [no_of_elements_per_scanline, state_vector_dimensionality]
#            This also results in [batch_size, state_vector_dimensionality]
#            
#            The third term is the bias which is a row vector of
#            dimension [state_vector_dimensionality]. This single row is 
#            "broadcast" to all the batch_size rows of the other two terms
        x = sess.run(b_rnn,
                              feed_dict={_inputs: batch_x, \
                                         y: batch_y})
        assert(1 == np.asmatrix(x).shape[0] and
               state_vector_dimensionality == np.asmatrix(x).shape[1])

#        
#        The number of times that the input is scanned equals 
#        transposed_input[0] i.e. no_of_scanlines_per_image. 
#        Therefore, all_hidden_states[0] is no_of_scanlines_per_image 
#        as asserted below
 
        x = sess.run(all_hidden_states,
                              feed_dict={_inputs: batch_x, \
                                         y: batch_y})
        assert(no_of_scanlines_per_image == x.shape[0] and
               batch_size == x.shape[1] and
               state_vector_dimensionality == x.shape[2])
        
#        This checks the True branch of the tf.cond() node.
#        It returns the tensor, all_hidden_states, 
#        if the condition holds True
        x = sess.run(cond_T_all_hidden_states,
                              feed_dict={_inputs: batch_x, \
                                         y: batch_y})
        assert(no_of_scanlines_per_image == x.shape[0] and
               batch_size == x.shape[1] and
               state_vector_dimensionality == x.shape[2])
        
#        This checks the False branch of the tf.cond() node.
#        It returns the tensor transpose [1,0,2] of, all_hidden_states, 
#        if the condition fails
        x = sess.run(cond_F_all_hidden_states,
                              feed_dict={_inputs: batch_x, \
                                         y: batch_y})
        assert(batch_size == x.shape[0] and
               no_of_scanlines_per_image == x.shape[1] and
               state_vector_dimensionality == x.shape[2])
        
#        Each hidden layer (of all_hidden_states) is multiplied by 
#        Wl which has dimension of [state_vector_dimensionality, no_of_classes],
#        to obtain all_outputs which thus has the dimensions as 
#        asserted next
        
        x = sess.run(all_outputs,
                              feed_dict={_inputs: batch_x, \
                                         y: batch_y})
        assert(no_of_scanlines_per_image == x.shape[0] and
               batch_size == x.shape[1] and
               no_of_classes == x.shape[2])

        x = sess.run(output,
                              feed_dict={_inputs: batch_x, \
                                         y: batch_y})
        assert(batch_size == x.shape[0] and
               no_of_classes == x.shape[1])

#        Will running the graph nodes also start the process of
#        logging? No file writer object hass been created, as done
#        for the session below. Needs to be checked.
#        As a precaution clearing any cache, before going to the 
#        session where the logging is needed. 
#        tf.summary.FileWriterCache.clear()

except:
    tf.summary.FileWriterCache.clear()
    tf.reset_default_graph()
    print("\nException in the bunch of assertions.")
    raise
    
    
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
    
        for i in range(10000):
    
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Reshape data to get 28 sequences of 28 pixels
                batch_x = batch_x.reshape((batch_size, \
                                           no_of_scanlines_per_image, \
                                           no_of_elements_per_scanline))
                summary, _ = sess.run([merged, train_step],
                                      feed_dict={_inputs: batch_x, \
                                                 y: batch_y})
                # Add to summaries
                train_writer.add_summary(summary, i)
    
                if i % 1000 == 0:
                    acc, loss, = sess.run([accuracy, cross_entropy],
                                          feed_dict={_inputs: batch_x,
                                                     y: batch_y})
                    print("Iter " + str(i) + ", Minibatch Loss= " +
                          "{:.6f}".format(loss) + ", Training Accuracy= " +
                          "{:.5f}".format(acc))
                if i % 100 == 0:
                    # Calculate accuracy for 128 mnist test images and
                    # add to summaries
                    summary, acc = sess.run([merged, accuracy],
                                            feed_dict={_inputs: test_data,
                                                       y: test_label})
                    test_writer.add_summary(summary, i)
    
        test_acc = sess.run(accuracy, feed_dict={_inputs: test_data,
                                                 y: test_label})
        print("Test Accuracy:", test_acc)
    
finally:
    #This is needed only when the file is run in spyder.
    #A re-run will cause an exception.
    #If run from the command line there is no problem
    print("Exiting from finally.")
    tf.reset_default_graph()
