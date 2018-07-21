# -*- coding: utf-8 -*-
"""
File is modification of LSTM-supervised_embeddings.py

What does  this do?
------------------
    1. It CLASSIFIES input sentences as odd, or even, or NOTA
    
What is a sentence?
------------------
    See head of LSTM-supervised_embeddings.py 
    
Questions
---------
    1. 
        1. What if a sentence contains words from both the sets? Should not the test 
        set have such cases? 

What do the Tests Prove?
------------------------
        
TODO
----
    
"""

import numpy as np
import tensorflow as tf

import argparse
import sys
from tensorflow.python import debug as tf_debug


WORDS_IN_A_SENTENCE = 6

batch_size = 128
#There are just 10 words in the vocabulary, so why have a dimensionality of
#64 for embedding?
embedding_space_dimensionality = 64
NUM_OF_CLASSES = 3
hidden_layer_size = 32
times_steps = WORDS_IN_A_SENTENCE 
##############################################################
#GENERATE simulated text sentences
digit_to_word_map = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                     6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
digit_to_word_map[0] = "PAD"

NUM_RANGE = len(digit_to_word_map)
WORDS_IN_VOCABULARY = len(digit_to_word_map)
NUM_OF_SENTENCES = 20000
MIN_LEN_OF_UNPADDED_SENTENCE = 3
MAX_LEN_OF_UNPADDED_SENTENCE = WORDS_IN_A_SENTENCE   #6
MIN_ODD_NUM = 1
MIN_EVEN_NUM = 2

def main(_):
    list_of_even_sentences = []
    list_of_odd_sentences = []
    list_of_NOTA_sentences = []
    list_of_sentence_lens = []
    try:
        for i in range(NUM_OF_SENTENCES//2):
            rand_seq_len = np.random.choice(range(MIN_LEN_OF_UNPADDED_SENTENCE, \
                                                  (MAX_LEN_OF_UNPADDED_SENTENCE + 1)))
            list_of_sentence_lens.append(rand_seq_len)
            #Generate two sequences; one of odd numbers, the other of even numbers.
            #The numbers are in the range 1-9. The length of both the sequences is the
            #same
            rand_odd_ints = np.random.choice(range(MIN_ODD_NUM, NUM_RANGE, 2),
                                             rand_seq_len)
            rand_even_ints = np.random.choice(range(MIN_EVEN_NUM, NUM_RANGE, 2),
                                              rand_seq_len)
        
            if rand_seq_len < WORDS_IN_A_SENTENCE:
                #PAD out the sequences so that all sequence are of length WORDS_IN_A_SENTENCE
                rand_odd_ints = np.append(rand_odd_ints,
                                          [0]*(WORDS_IN_A_SENTENCE-rand_seq_len))
                rand_even_ints = np.append(rand_even_ints,
                                           [0]*(WORDS_IN_A_SENTENCE-rand_seq_len))
        
            #Now convert the list of Odd(even) numbers into a odd(even) sentence
            #of words
            #Uses <space> as  delimiter See pg 213 of Lutz
            list_of_even_sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
            list_of_odd_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
            tmp = []
            for ii in range(WORDS_IN_A_SENTENCE):
                if(np.random.uniform() < 0.5):
                    tmp = tmp + [digit_to_word_map[rand_even_ints[ii]]] 
                else:
                    tmp = tmp + [digit_to_word_map[rand_odd_ints[ii]]] 

            list_of_NOTA_sentences.append(" ".join(tmp))
            
            pass
        
        list_of_sentences = list_of_even_sentences+\
                            list_of_odd_sentences + \
                            list_of_NOTA_sentences
        
        list_of_sentence_lens *= NUM_OF_CLASSES
        
        #On account of the concatenation of odd and even sentences, done above,
        #the first half of the list_of_sentences has even sentences, the latter part has odd.
        #Thus the list of labels can easily be constructed as below
        #
        Label_ODD = 0 #for odd sentence
        Label_EVEN = 1 #for even sentence
        Label_NOTA = 2 #for NOTA sentence
        list_of_labels = [Label_EVEN] * (NUM_OF_SENTENCES//2) + \
                            [Label_ODD] * (NUM_OF_SENTENCES//2) + \
                            [Label_NOTA] * (NUM_OF_SENTENCES//2) 
        
        ##################################################################
        #PRE_PROCESS the sentences
                            
        for i in range(len(list_of_labels)):
            label = list_of_labels[i]
            assert(Label_ODD == label or
                   Label_EVEN == label or
                   Label_NOTA == label)
            one_hot_encoding = [0]*NUM_OF_CLASSES
            one_hot_encoding[label] = 1 
            if(Label_ODD ==label):
                assert([1, 0, 0] == one_hot_encoding)
            elif(Label_EVEN == label):
                assert([0, 1, 0] == one_hot_encoding)
            else:
                assert([0, 0, 1] == one_hot_encoding)
            list_of_labels[i] = one_hot_encoding
        
        word2index_map = {} #NOTE: This is going to be a dictionary
        number_of_distinct_words_found = 0 
        for a_sentence in list_of_sentences:
            for word in a_sentence.lower().split():
                if word not in word2index_map:
                    word2index_map[word] = number_of_distinct_words_found
                    number_of_distinct_words_found += 1
        
        assert(WORDS_IN_VOCABULARY >= number_of_distinct_words_found) #
        #Create inverse dictionary
        index2word_map = {number_of_distinct_words_found: word for \
                          word, number_of_distinct_words_found in \
                          word2index_map.items()}
        
        #Sanity check 
        assert(len(index2word_map) <= WORDS_IN_VOCABULARY)
        
        #The list of sentences is highly ordered. It needs to be shuffled. 
        #Each sentence in list_of_sentences is accessed by
        #an index. Thus if we create a list of the indices we can then shuffle the list
        # to effectively shuffle the list of sentences.
        #A sentence is thus accessed by two levels of indirection
        list_of_indices = list(range(len(list_of_sentences)))
        np.random.shuffle(list_of_indices) # This is now the permuted indices
        
        #now permute the list_of_sentences and apply the same permutation 
        #to list_of_labels and list_of_sentence_lens
        array_of_sentences = np.array(list_of_sentences)[list_of_indices]
        array_of_labels = np.array(list_of_labels)[list_of_indices]
        array_of_sentence_lengths = np.array(list_of_sentence_lens)[list_of_indices]
        
        #split the shuffled sentences, equally, into training and testing data
        train_x = array_of_sentences[:NUM_OF_SENTENCES//2]
        train_y = array_of_labels[:NUM_OF_SENTENCES//2]
        train_sentence_lens = array_of_sentence_lengths[:NUM_OF_SENTENCES//2]
        
        test_x = array_of_sentences[NUM_OF_SENTENCES//2:]
        test_y = array_of_labels[NUM_OF_SENTENCES//2:]
        test_sentence_lens = array_of_sentence_lengths[NUM_OF_SENTENCES//2:]
        #############################################################
        #Utility to be used to get batches of the generated sentences
        #The input is 
            #a list of sentences as an array and 
            #the corresponding array of labels for the sentences
            #the corresponding length of the sentences (not including PADding)
        #The output is 
            #1. A random selection from the list of sentence, equal in number to 
            #the desired batch size. The words of the sentences are represented by
            #their numeric IDs
            #2. The corresponding labels
            #3. The corresponding length of the sentences
            
        def get_sentence_batch(batch_size, data_x,
                               data_y, data_x_sentence_lengths):
            
            assert(WORDS_IN_A_SENTENCE == len(data_x[0].split()))
            
            list_of_indices = list(range(len(data_x)))
            np.random.shuffle(list_of_indices)
            batch_indices = list_of_indices[:batch_size]
            x = [[word2index_map[word] for word in data_x[i].lower().split()]
                 for i in batch_indices]
            
            assert(batch_size == len(x))
            assert((batch_size, WORDS_IN_A_SENTENCE) == np.asarray(x).shape)
            assert(int == type(x[0][0]))
            assert(int == type(x[len(x) - 1][0]))
            #Get the labels for the selected sentences 
            y = [data_y[i] for i in batch_indices]
            #Get the length of the sentences
            array_of_sentence_lengths = [data_x_sentence_lengths[i] for i in batch_indices]
            
            return x, y, array_of_sentence_lengths
        
        #################################################################
        #Construct TensorFlow Graph
            
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
        
       
        weights = {
            'linear_layer': tf.Variable(tf.truncated_normal([hidden_layer_size, \
                                                             NUM_OF_CLASSES],
                                                            mean=0, stddev=.01),
                                                        name='linear_layer_weights')
        }
        biases = {
            'linear_layer': tf.Variable(tf.truncated_normal([NUM_OF_CLASSES], \
                                                            mean=0, stddev=.01),
                                                            name='linear_layer_biases')
        }
        
        # extract the last relevant output and use in a linear layer
        final_output = tf.matmul(states[1],
                                 weights["linear_layer"]) + biases["linear_layer"]
        
        softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_output,
                                                          labels=_labels)
        cross_entropy = tf.reduce_mean(softmax)
        
        train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(_labels, 1),
                                      tf.argmax(final_output, 1))
        accuracy = (tf.reduce_mean(tf.cast(correct_prediction,
                                           tf.float32)))*100
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if FLAGS.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess, \
                                                          ui_type=FLAGS.ui_type)
        
            x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size,
                                                                train_x, 
                                                                train_y,
                                                                train_sentence_lens)
            
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

            for step in range(500):
                x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size,
                                                                    train_x, train_y,
                                                                    train_sentence_lens)
                sess.run(train_step, feed_dict={_inputs: x_batch, _labels: y_batch,
                                                _sentence_lens: seqlen_batch})
        
                if step % 100 == 0:
                    acc = sess.run(accuracy, feed_dict={_inputs: x_batch,
                                                        _labels: y_batch,
                                                        _sentence_lens: seqlen_batch})
                    print("Accuracy at %d: %.5f" % (step, acc))
        
            for test_batch in range(5):
                x_test, y_test, seqlen_test = get_sentence_batch(batch_size,
                                                                 test_x, test_y,
                                                                 test_sentence_lens)
                labels_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy],
                                                 feed_dict={_inputs: x_test,
                                                            _labels: y_test,
                                                            _sentence_lens: seqlen_test})
                print("\nTest batch accuracy %d: %.5f" % (test_batch, batch_acc))

#                print("Predicted Labels:\n", labels_pred)
#                print("Actual Labels:\n", np.argmax(y_test, 1))
                
                labels_expected =  np.argmax(y_test, 1)
                list_of_error_tuples = []
                list_of_misclassified_sentences = []
                for i in range(len(labels_pred)):
                    if(not(labels_expected[i] == labels_pred[i])):
                        list_of_error_tuples.append((i, \
                                                     seqlen_test[i], \
                                                     labels_expected[i], \
                                                     labels_pred[i]))
                        list_of_misclassified_sentences.append(x_test[i])
#                print("list_of_error_tuples:\n", list_of_error_tuples)
                for tuple_index in range(len(list_of_error_tuples)):
                    print(\
                  "Error Tuple: Index: {}, Seq_len: {}, True_label: {}, Label: {}".format(\
                          list_of_error_tuples[tuple_index][0], \
                          list_of_error_tuples[tuple_index][1],
                          list_of_error_tuples[tuple_index][2],
                          list_of_error_tuples[tuple_index][3]
                                      ))
                print("list_of_misclassified_sentences:\n", list_of_misclassified_sentences)

                num_of_NOTAs = 0
                num_of_NOTAs_found = 0
                for i in range(len(labels_expected)):
                    if(Label_NOTA == labels_expected[i]):
                        num_of_NOTAs +=1
                    if(Label_NOTA == labels_pred[i]):
                        num_of_NOTAs_found +=1
                print("No Of NOTAs Present: {}; Found: {}".format(num_of_NOTAs, \
                                                          num_of_NOTAs_found))
                
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
                
#            states_example = sess.run([states[1]], feed_dict={_inputs: x_test,
            states_example = sess.run([states], feed_dict={_inputs: x_test,
                                                              _labels: y_test,
                                                              _sentence_lens: seqlen_test})
            assert((1, 2, \
                    batch_size, \
                    hidden_layer_size) == \
                np.asarray(states_example).shape)
#            print("\nseqlen_test[1]:{}, expected between {} and {}".format(seqlen_test[1], \
#                  MIN_LEN_OF_UNPADDED_SENTENCE, MAX_LEN_OF_UNPADDED_SENTENCE))

            print("\n\tDONE:", __name__)
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
    exit(0)
      