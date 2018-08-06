# -*- coding: utf-8 -*-
"""
Created by RM in Jul,2018 by copying and editing
LSTM_supervised_embeddings.py

What does  this do?
------------------
    1. It CLASSIFIES input sentences as odd, or even.
    2. It uses MultiRNN
    
What is a sentence?
------------------
    See header comments LSTM_supervised_embeddings.py
    
Questions
---------
    1. 
        1. What if a sentence contains words from both the sets? Should not the test 
        set have such cases? 
        2. Since the test sentences are "pure" ODD/EVEN sentences, what do the tests prove?
        (See below for answer)
    2. ...
    3. What if the mnist data were to processed through this file? 
        1. MNIST data is processed in vanilla_rnn_with_tfboard.py.
        2. MNIST data is processed in BasicRNNCell.py
    4. What if 
        1. MIN_LEN_OF_UNPADDED_SENTENCE is reduced?
        2. MAX_LEN_OF_UNPADDED_SENTENCE is reduced?
        

What do the Tests Prove?
------------------------

TODO
----
    1. Process MNIST data with LSTM & embedding
    
"""

import numpy as np
import tensorflow as tf

import argparse
import sys
from tensorflow.python import debug as tf_debug

WORDS_IN_A_SENTENCE = 6

NUM_OF_LSTM_CELLS = 2              

BATCH_SIZE = 128
NUM_OF_TRAINING_RUNS = 500
NUM_OF_TEST_RUNS = 5
CHECK_EVERY_ITER = 100

#There are just 10 words in the vocabulary, so why have a dimensionality of
#64 for embedding?
embedding_space_dimensionality = 64
num_classes = 2 #Classes are ODD & EVEN Sentences
hidden_layer_size = 32
times_steps = WORDS_IN_A_SENTENCE #6 #TODO: Is this connected to WORDS_IN_A_SENTENCE?

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
            list_of_even_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
            list_of_odd_sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
        
        #list_of_sentences.shape() is ???
        #IT IS A LIST OF SENTENCES; EACH OF LENGTH NUM_RANGE.
        #Total number of sentences is NUM_OF_SENTENCES. 
        #
        #SO IS IT (20000, NUM_RANGE) OR (20000, 1)? 
        #THE LATTER. SEE BELOW 
        #odd (even) accessed as odd[0:WORDS_IN_A_SENTENCE]; and not as odd[0:WORDS_IN_A_SENTENCE :]
        #
        list_of_sentences = list_of_even_sentences+list_of_odd_sentences
        
        #The instruction below, Concatenates the list to itself. (See 109 Lutz)
        #What's the idea???
        #In the list_of_sentences the first half has even sentences the latter half has odd.
        #This is a consequence of the concatenation done above. 
        #The odd and even sentences are generated as a pair with
        #the same length. They are then appended to respective lists. Thus their indices,
        #in their respective lists are identical. The list of sequence
        #lengths, as constructed has the length of both even and odd sentences at any given
        #index. Since the two lists are concatenated, the indices for sentences in 
        #the list_of_odd_sentences will be at an offset of NUM_OF_SENTENCES//2 
            #For an index i, where 0 <= i < NUM_OF_SENTENCES//2, 
            #list_of_sentence_lens[i] == list_of_sentence_lens[i + NUM_OF_SENTENCES//2]
        #
        # Thus all that is needed is to replicate list of sequences. Now the 
        #list_of_sentence_lens amtches the length of sentences in list_of_sentences
        list_of_sentence_lens *= 2
        
        #On account of the concatenation of odd and even sentences, done above,
        #the first half of the list_of_sentences has even sentences, the latter part has odd.
        #Thus the list of labels can easily be constructed as below
        #
        # Label 0 for odd length sentence
        # Label 1 for even length sentence
        list_of_labels = [1] * (NUM_OF_SENTENCES//2) + \
                            [0] * (NUM_OF_SENTENCES//2)
        
        ##################################################################
        #PRE_PROCESS the sentences
                            
        for i in range(len(list_of_labels)):
            label = list_of_labels[i]
            assert(0 == label or
                   1 == label)
            one_hot_encoding = [0]*2
            one_hot_encoding[label] = 1 
            if(0 ==label):
                assert([1, 0] == one_hot_encoding)
            else:
                assert([0, 1] == one_hot_encoding)
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
            
        def get_sentence_batch(BATCH_SIZE, data_x,
                               data_y, data_x_sentence_lengths):
            
            assert(WORDS_IN_A_SENTENCE == len(data_x[0].split()))
            
            list_of_indices = list(range(len(data_x)))
            np.random.shuffle(list_of_indices)
            batch_indices = list_of_indices[:BATCH_SIZE]
            x = [[word2index_map[word] for word in data_x[i].lower().split()]
                 for i in batch_indices]
            
            assert(BATCH_SIZE == len(x))
            assert((BATCH_SIZE, WORDS_IN_A_SENTENCE) == np.asarray(x).shape)
            assert(int == type(x[0][0]))
            assert(int == type(x[len(x) - 1][0]))
            #Get the labels for the selected sentences 
            y = [data_y[i] for i in batch_indices]
            #Get the length of the sentences
            array_of_sentence_lengths = [data_x_sentence_lengths[i] for i in batch_indices]
            
            return x, y, array_of_sentence_lengths
        
        #################################################################
        #Construct TensorFlow Graph
            
        _inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE, times_steps])
        _labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, num_classes])
        _sentence_lens = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        
        with tf.name_scope("embeddings"):
            embeddings = tf.Variable(
                tf.random_uniform([number_of_distinct_words_found,
                                   embedding_space_dimensionality],
                                  -1.0, 1.0), name='embedding')

            embed = tf.nn.embedding_lookup(embeddings, _inputs)
        
        with tf.variable_scope("lstm"):
        
#                 lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size,
#                                                          forget_bias=1.0,
#                                                         name='BasicLSTMCell')
#                 num_units = [128, 64]
#                 cell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell(num_units[0], lstm_cell(num_units[1]))], 
#                                                    state_is_tuple=True)
#                 outputs, states = tf.nn.dynamic_rnn(cell, embed,
#                                                     sequence_length = _sentence_lens,
#                                                     dtype=tf.float32)
            #############################################################
            #The above is from the book. It throws an exception. The inner
            #dimensions for a matrix multiplication are not equal
            #The following has been taken from
            #https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
            #############################################################
            # create 2 LSTMCells
            rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in \
                          [2 * hidden_layer_size, hidden_layer_size]]
            
            # create a RNN cell composed sequentially of a number of RNNCells
            multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
            
            # 'outputs' is a tensor of shape 
            #[BATCH_SIZE, WORDS_IN_A_SENTENCE, hidden_layer_size]
            # 'state' is a N-tuple where N is the number of LSTMCells containing a
            # tf.contrib.rnn.LSTMStateTuple for each cell
            outputs, states = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                               inputs=embed,
                                               dtype=tf.float32)
             
        weights = {
            'linear_layer': tf.Variable(tf.truncated_normal([hidden_layer_size, \
                                                             num_classes],
                                                            mean=0, stddev=.01),
            name='linear_layer_weights')
        }
        biases = {
            'linear_layer': tf.Variable(tf.truncated_normal([num_classes], \
                                                            mean=0, stddev=.01),
            name='linear_layer_biases')
        }
        
        # extract the last relevant output and use in a linear layer
        #extract the final state and use in a linear layer
        final_output = tf.matmul(states[NUM_OF_LSTM_CELLS-1][1],
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
        
            x_batch, y_batch, seqlen_batch = get_sentence_batch(BATCH_SIZE,
                                                                train_x, \
                                                                train_y, \
                                                                train_sentence_lens)
            
            word_embeddings, embeddings= sess.run([embed, embeddings], \
                                                  feed_dict={_inputs: x_batch, \
                                                             _labels: y_batch,
                                            _sentence_lens: seqlen_batch})
            assert((BATCH_SIZE, \
                    WORDS_IN_A_SENTENCE) == np.asarray(x_batch).shape)
            assert((number_of_distinct_words_found,  \
                    embedding_space_dimensionality) == embeddings.shape)
            assert((BATCH_SIZE, \
                    WORDS_IN_A_SENTENCE, \
                    embedding_space_dimensionality) == word_embeddings.shape)

            for step in range(NUM_OF_TRAINING_RUNS):
                x_batch, y_batch, seqlen_batch = get_sentence_batch(BATCH_SIZE,
                                                                    train_x, train_y,
                                                                    train_sentence_lens)
                sess.run(train_step, feed_dict={_inputs: x_batch, _labels: y_batch,
                                                _sentence_lens: seqlen_batch})
        
                if step % CHECK_EVERY_ITER == 0:
                    acc = sess.run(accuracy, feed_dict={_inputs: x_batch,
                                                        _labels: y_batch,
                                                        _sentence_lens: seqlen_batch})
                    print("Accuracy at %d: %.5f" % (step, acc))

            for test_batch in range(NUM_OF_TEST_RUNS):
                x_test, y_test, seqlen_test = get_sentence_batch(BATCH_SIZE,
                                                                 test_x, test_y,
                                                                 test_sentence_lens)
                batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy],
                                                 feed_dict={_inputs: x_test,
                                                            _labels: y_test,
                                                            _sentence_lens: seqlen_test})
                print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))
        
            output_example = sess.run([outputs], feed_dict={_inputs: x_test,
                                                            _labels: y_test,
                                                            _sentence_lens: seqlen_test})
            assert((1,BATCH_SIZE, WORDS_IN_A_SENTENCE, hidden_layer_size) == \
                   np.asarray(output_example).shape)
            
            #################################################################
            states_all = sess.run([states], feed_dict={_inputs: x_test,
                                                              _labels: y_test,
                                                              _sentence_lens: seqlen_test})
            #The first dimension is owing to putting the tensor states as a list.
            #If the [] are removed the asertion fails
            assert(1 == len(states_all))
            assert(NUM_OF_LSTM_CELLS == len(states_all[0]))
            assert(2 == len(states_all[0][0]))
            #For c & h see
            #https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMStateTuple
            #and
            #https://colah.github.io/posts/2015-08-Understanding-LSTMs/
            #The latter has *** a lot *** of "gyan"
            assert((BATCH_SIZE, 2*hidden_layer_size) == \
                   states_all[0][0].c.shape)
            assert((BATCH_SIZE, 2*hidden_layer_size) == \
                   states_all[0][0].h.shape)

            assert((BATCH_SIZE, hidden_layer_size) == \
                   states_all[0][1].c.shape)
            assert((BATCH_SIZE, hidden_layer_size) == \
                   states_all[0][1].h.shape)

            #################################################################
            states_index1 = sess.run([states[1]], feed_dict={_inputs: x_test,
                                                              _labels: y_test,
                                                              _sentence_lens: seqlen_test})
            
            assert(1 == len(states_index1))
            assert(NUM_OF_LSTM_CELLS == len(states_index1[0]))
            assert((BATCH_SIZE, hidden_layer_size) == \
                    (states_index1[0][0]).shape)
            assert((BATCH_SIZE, hidden_layer_size) == \
                    (states_index1[0][1]).shape)

            #################################################################
            #Now both together
            states_all, states_index1 = sess.run([states, states[1]], feed_dict={_inputs: x_test,
                                                              _labels: y_test,
                                                              _sentence_lens: seqlen_test})

            #NOTE how the outer most level of nesting has been removed.
            #They have "lost" the first dimension.
            #See above. If the [] are removed for single tensor sess.run(), then 
            # the asserts will be the same
            assert(NUM_OF_LSTM_CELLS == len(states_index1))
            assert(BATCH_SIZE == len(states_index1[0]))
            assert(BATCH_SIZE == len(states_index1[1]))
            assert((BATCH_SIZE, hidden_layer_size) == \
                    (states_index1[0]).shape)
            assert((BATCH_SIZE, hidden_layer_size) == \
                    (states_index1[1]).shape)

            assert(NUM_OF_LSTM_CELLS == len(states_all))
            assert(NUM_OF_LSTM_CELLS == len(states_all[0]))
            assert((BATCH_SIZE, 2*hidden_layer_size) == \
                   states_all[0].c.shape)
            assert((BATCH_SIZE, 2*hidden_layer_size) == \
                   states_all[0].h.shape)

            assert((BATCH_SIZE, hidden_layer_size) == \
                   states_all[1].c.shape)
            assert((BATCH_SIZE, hidden_layer_size) == \
                   states_all[1].h.shape)


            
            print("\n\tDONE:", __file__, "\n")
    finally:
        tf.reset_default_graph()

#ipdb> states_example[0][0].c.shape
#(128, 64)
#
#ipdb> states_example[0][0].h.shape
#(128, 64)
#
#ipdb> states_example.ndim
#*** AttributeError: 'list' object has no attribute 'ndim'
#
#ipdb> len(states_example)
#1
#
#ipdb> len(states_example[0])
#2
#
#ipdb> len(states_example[0][0])
#2
#
#ipdb> states_example[0][1].h.shape
#(128, 32)
#
#ipdb> states_example[0][1].c.shape
#(128, 32)
#
#ipdb> 
#ipdb> 
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
      