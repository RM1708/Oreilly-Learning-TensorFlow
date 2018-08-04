# -*- coding: utf-8 -*-
"""MNIST dimensionality reduction with TensorFlow and TensorBoard.

This demonstrates the functionality of the TensorBoard Embedding Visualization dashboard using MNIST.

https://www.tensorflow.org/versions/r0.12/how_tos/embedding_viz/index.html#tensorboard-embedding-visualization
"""

import numpy as np
import tensorflow as tf

import argparse
import sys

from Data_Even_Odd_NOTA_Sentences import get_sentence_batch, \
                                        generate_data_sentences, \
                                        labels_to_one_hot, \
                                        NUM_OF_CLASSES

from SentenceErrorMetrics import metrics_using_python
                                        
############################################################################

WORDS_IN_A_SENTENCE = 6

batch_size = 128
#QUESTION: There are just 10 words in the vocabulary, so why have a dimensionality of
#64 for embedding?
embedding_space_dimensionality = 64
hidden_layer_size = 32
times_steps = WORDS_IN_A_SENTENCE 
##############################################################
#GENERATE simulated words
#Map is needed as random ints can be generated which can then be mapped to words
#in our vocabulary
digit_to_word_map = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                     6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
digit_to_word_map[0] = "PAD"
#
NUM_OF_RUN_ITERS = 5000
NUM_OF_TEST_ITERS = 5
NUM_RANGE = len(digit_to_word_map)
#WORDS_IN_VOCABULARY = len(digit_to_word_map)
NUM_OF_SENTENCES = 20000
MIN_LEN_OF_UNPADDED_SENTENCE = 3
MAX_LEN_OF_UNPADDED_SENTENCE = WORDS_IN_A_SENTENCE   #6
MIN_ODD_NUM = 1
MIN_EVEN_NUM = 2

def main(_):
    try:
        #GENERATE the simulated sentences
        train_x, train_y, train_sentence_lens, \
        test_x, test_y, test_sentence_lens, \
        number_of_distinct_words_found = \
                    generate_data_sentences(NUM_OF_SENTENCES, \
                                    MIN_LEN_OF_UNPADDED_SENTENCE, \
                                    MAX_LEN_OF_UNPADDED_SENTENCE, \
                                    MIN_ODD_NUM, \
                                    MIN_EVEN_NUM, \
                                    NUM_RANGE, \
                                    WORDS_IN_A_SENTENCE, \
                                    digit_to_word_map)
        #################################################################
        #CONSTRUCT TensorFlow Graph
            
        _inputs = tf.placeholder(tf.int32, shape=[batch_size, times_steps])
        _labels = tf.placeholder(tf.float32, shape=[batch_size, NUM_OF_CLASSES])
        _sentence_lens = tf.placeholder(tf.int32, shape=[batch_size])
        
        with tf.name_scope("embeddings"):
            embeddings = tf.Variable(
                        tf.random_uniform([number_of_distinct_words_found,
                                           embedding_space_dimensionality],
                                          -1.0, 1.0), name='embedding')
            embed = tf.nn.embedding_lookup(embeddings, _inputs)
        
        with tf.variable_scope("lstm"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size,
                                                     forget_bias=1.0,
                                                     name='BasicLSTMCell')
            outputs, states = tf.nn.dynamic_rnn(lstm_cell, embed,
                                                sequence_length=_sentence_lens,
                                                dtype=tf.float32)
        
        with tf.variable_scope("final_output"):
            weights = {'linear_layer': \
                        tf.Variable(tf.truncated_normal([hidden_layer_size, \
                                                             NUM_OF_CLASSES], \
                                                        mean=0, \
                                                        stddev=.01), \
                                                        name='linear_layer_weights')}
            biases = {'linear_layer': \
                        tf.Variable(tf.truncated_normal([NUM_OF_CLASSES], \
                                                        mean=0, \
                                                        stddev=.01), \
                                                        name='linear_layer_biases')}
            # extract the last relevant output and use in a linear layer
            final_output = tf.matmul(states[1],
                                     weights["linear_layer"]) + biases["linear_layer"]
        
        with tf.variable_scope("train"):
            softmax = tf.nn.softmax_cross_entropy_with_logits_v2(\
                                                    logits=final_output,
                                                    labels=_labels)
            cross_entropy = tf.reduce_mean(softmax)
            train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

            with tf.variable_scope("monitor_training"):
                correct_prediction = tf.equal(tf.argmax(_labels, 1),
                                              tf.argmax(final_output, 1))
                accuracy = (tf.reduce_mean(tf.cast(correct_prediction,
                                                   tf.float32)))*100
        ######################################################################
        #METRICS
        #Use TensorFlow to collect metrics about false +ves and false -ves
        #First using tf.argmax(). This does not appear to give correct results.
        #That is not surprising. argmax returns the index so it could be 0, or
        #1, or 2. tf.metrics.false_* transforms the values to bool - check the
        #documentation.
        metric_false_neg_using_argmax = tf.metrics.false_negatives(
                                                    tf.argmax(_labels, 1),
                                                    tf.argmax(final_output, 1))
        metric_false_pos_using_argmax = tf.metrics.false_positives(
                                                    tf.argmax(_labels, 1),
                                                    tf.argmax(final_output, 1))

        #Now using one_hot coding of the labels. This appears to give correct results.
        #In this case the labels are represented as a list of three elements. 
        #Only one of the elements is a 1. The others are 0. The actual label 
        #value is the index at which the 1 occurs. Thus when tf.metrics_false_*
        #transforms the labels and predictions to bool, nothing changes.
        final_output_one_hot = tf.py_func(labels_to_one_hot, \
                   [tf.argmax(final_output, 1)], \
#                   The folowing two also work
#                   [(tf.argmax(final_output_print, 1))[:batch_size]], \
#                   [np.asarray(tf.argmax(final_output_print, 1)).tolist()[:batch_size]], \
                   (
#                   #THIS IS VERY INELEGANT. There *** has *** to be a better way
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64
                    ))
        final_output_one_hot_print_0 = tf.Print(final_output_one_hot, \
                                      [], \
                                      message="")
#                                      [tf.shape(final_output_one_hot)], \
#                                      message="final_output_one_hot shape: ")
        
        metric_false_pos_using_1hot = \
            tf.metrics.false_positives(_labels[:batch_size],
                                      final_output_one_hot_print_0)
        metric_false_neg_using_1hot = \
            tf.metrics.false_positives(_labels[:batch_size],
                                      final_output_one_hot_print_0)
#############################################################################
#OPERATE the Graph
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        
            x_batch, y_batch, seqlen_batch = \
                                        get_sentence_batch(batch_size, \
                                                            train_x, 
                                                            train_y,
                                                            train_sentence_lens, \
                                                            WORDS_IN_A_SENTENCE)
            word_embeddings, embeddings= sess.run([embed, embeddings], \
                                                  feed_dict={_inputs: x_batch,\
                                                             _labels: y_batch,
                                                            _sentence_lens: seqlen_batch})
            assert((batch_size, \
                    WORDS_IN_A_SENTENCE) == np.asarray(x_batch).shape)
            assert((number_of_distinct_words_found,  \
                    embedding_space_dimensionality) == embeddings.shape)
            assert((batch_size, \
                    WORDS_IN_A_SENTENCE, \
                    embedding_space_dimensionality) == word_embeddings.shape)

            for step in range(NUM_OF_RUN_ITERS):
                x_batch, y_batch, seqlen_batch = \
                                            get_sentence_batch(batch_size, \
                                                                train_x, 
                                                                train_y,
                                                                train_sentence_lens, \
                                                                WORDS_IN_A_SENTENCE)

                sess.run(train_step, feed_dict={_inputs: x_batch, _labels: y_batch,
                                                _sentence_lens: seqlen_batch})
        
                if step % 100 == 0:
                    acc = sess.run(accuracy, feed_dict={_inputs: x_batch,
                                                        _labels: y_batch,
                                                        _sentence_lens: seqlen_batch})
                    print("Accuracy at %d: %.5f" % (step, acc))
        
###############################################################################
#TEST how good was the learning
            for test_batch in range(NUM_OF_TEST_ITERS):
                sess.run(tf.local_variables_initializer())
                x_test, y_test, seqlen_test = \
                                            get_sentence_batch(batch_size, \
                                                                test_x, 
                                                                test_y,
                                                                test_sentence_lens, \
                                                                WORDS_IN_A_SENTENCE)
                assert((batch_size, NUM_OF_CLASSES) == np.asarray(y_test).shape)
                
                labels_pred_, batch_acc = sess.run([final_output, accuracy],
                                                 feed_dict={_inputs: x_test,
                                                            _labels: y_test,
                                                            _sentence_lens: seqlen_test})
                
                assert((batch_size, NUM_OF_CLASSES) == labels_pred_.shape)
                print("\nTest batch accuracy %d: %.5f" % (test_batch, batch_acc))

                ###################################################################
                #METRICS using python codeand values returned by sess.run()
                #
                metrics_using_python(labels_pred_, \
                                         x_test, \
                                         y_test, \
                                         seqlen_test)
                ###################################################################
                #METRICS using tf.metrics
                metrics_false_neg  = \
                    sess.run([metric_false_neg_using_argmax], feed_dict={_inputs: x_test,
                                                                    _labels: y_test,
                                                                    _sentence_lens: seqlen_test})
                print("metric_false_neg_using_argmax: ", metrics_false_neg[0])
                
                metrics_false_pos = \
                    sess.run([metric_false_pos_using_argmax], feed_dict={_inputs: x_test,
                                                                    _labels: y_test,
                                                                    _sentence_lens: seqlen_test})
                print("metric_false_pos_using_argmax: ", metrics_false_pos[0])
                
                metrics_false_pos_ = \
                    sess.run([metric_false_pos_using_1hot], feed_dict={_inputs: x_test,
                                                                    _labels: y_test,
                                                                    _sentence_lens: seqlen_test})
                print("metric_false_pos_using_1hot: ", metrics_false_pos_[0])
                
                metrics_false_neg_ = \
                    sess.run([metric_false_neg_using_1hot], feed_dict={_inputs: x_test,
                                                                    _labels: y_test,
                                                                    _sentence_lens: seqlen_test})
                print("metric_false_neg_using_1hot: ", metrics_false_neg_[0])
                ###################################################################
                
            final_output_example = sess.run([final_output], feed_dict={_inputs: x_test,
                                                            _labels: y_test,
                                                            _sentence_lens: seqlen_test})
            assert((batch_size, \
                    NUM_OF_CLASSES) == \
                final_output_example[0].shape)

            output_example = sess.run([outputs], feed_dict={_inputs: x_test,
                                                            _labels: y_test,
                                                            _sentence_lens: seqlen_test})
            assert((1, batch_size, \
                    WORDS_IN_A_SENTENCE, \
                    hidden_layer_size) == \
                np.asarray(output_example).shape)
                
            states_example = sess.run([states], feed_dict={_inputs: x_test,
                                                              _labels: y_test,
                                                              _sentence_lens: seqlen_test})
            assert((1, 2, \
                    batch_size, \
                    hidden_layer_size) == \
                np.asarray(states_example).shape)

            print("\n\tDONE: \n", __file__, "\n")
    finally:
        tf.reset_default_graph()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    
    parser.add_argument(
    "--max_steps",
    type=int,
    default=10,
    help="Number of steps to run trainer.")
    
    parser.add_argument(
    "--train_batch_size",
    type=int,
    default=100,
    help="Batch size used during training.")
    
    parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.025,
    help="Initial learning rate.")
    
    parser.add_argument(
    "--data_dir",
    type=str,
    default="/home/rm/tmp/data/mnist_data",
    help="Directory for storing data")

    parser.add_argument(
    "--ui_type",
    type=str,
    default="curses",
    help="Command-line user interface type (curses | readline)")
    
    parser.add_argument(
    "--fake_data",
    type="bool",
    nargs="?",
    const=True,
    default=False,
    help="Use fake MNIST data for unit testing")
    
    parser.add_argument(
    "--debug",
    type="bool",
    nargs="?",
    const=True,
    default=False,
    help="Use debugger to track down bad values during training. "
    "Mutually exclusive with the --tensorboard_debug_address flag.")
    
    parser.add_argument(
    "--tensorboard_debug_address",
    type=str,
    default=None,
    help="Connect to the TensorBoard Debugger Plugin backend specified by "
    "the gRPC address (e.g., localhost:1234). Mutually exclusive with the "
    "--debug flag.")
    
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
      