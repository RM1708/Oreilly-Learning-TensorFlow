# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 00:39:23 2016

@author: tomhope

Heavily edited by RM in Jul,2018
Significant changes
    1. Intentional naming.
    2. Extensive documentation of my understanding of what is expected.
    3. Assert to confirm my undersatnding.
    4. Code to run it under tfdbg.

What does  this do?
------------------
    1. It CLASSIFIES input sentences as odd, or even.
    
What is a sentence?
------------------
    01. A sentence consists of exactly ten words. 
    02. The  words are separated by a <SPACE>
    03. The only words in the vocabulary are the words for the numerals
    1 thru 9. 0 represents the special word "PAD"
    04. The maximum number of words in a sentence is six. The minimum is
    three.
    05. A sentence is generated by first adding words from the vocabulary.
    This is subject to the condition mentioned at #4. Since the sentence 
    has to have 10 words, the deficit is made up by adding the words - PAD;
    To the  bring the number upto 10.
    06. A sentence that contains words from the set{Two, Four, Six, Eight} - apart 
    from the trailing PAD words - is an EVEN-sentence
    07. A sentence that contains words from the set{One, Three, Five, Seven, Nine} - apart 
    from the trailing PAD words - is an ODD-sentence.
    
Questions
---------
    1. 
        1. What if a sentence contains words from both the sets? Should not the test 
        set have such cases? 
        2. Since the test sentences are "pure" ODD/EVEN sentences, what do the tests prove?
        (See below for answer)
    2. What if we apply other classification methods to this data? See:
        1. Code-GettingStartedWithTF/Chapter\ 4/logistic_regression.py
        2. Code-GettingStartedWithTF/Chapter\ 4/mlp_classification.py.
        3. K-means?
    3. What if the mnist data were to processed through this file? 
        1. MNIST data is processed in vanilla_rnn_with_tfboard.py.
        2. MNIST data is processed in BasicRNNCell.py
    4. What if 
        1. MIN_LEN_OF_UNPADDED_SENTENCE is reduced?
        2. MAX_LEN_OF_UNPADDED_SENTENCE is reduced?
        

What do the Tests Prove?
------------------------
    1. Given 
        1. A vocabulary split into two mutually exclusive sets.
        2. Sentences formed with words exclusively from one set or the other. 
        The number of such words is variable between a minimum > 0 (>=?)
        and a maximum <= the prescribed length of a valid sentence.
        3. All valid sentences are the same prescribed length. 
        (In case a sentence falls short of the required number of words, 
        PAD words are added.)
    2. With embedding dimensionality of 1 i.e each word-id is mapped to a unique 
    number between -1.0 and 1.0, the model learns to distinguish between ODD and
    EVEN sentences.
    3. When embedding is onto a range set of three numbers -1.0 for odd 
    word-ids, +1.0 for even word-ids and 0 for PAD the performance is
            1. with mapping onto 10 randomly selected 0-D tensors in the 
            range -1.0 to 1.0, 100% steady accuracy is acheived at iteration
            610
            2. with mapping onto 10 randomly selected 3-D tensors, 
            with components in the range -1.0 to 1.0, 100% steady accuracy 
            is acheived at iteration 340
            3. with mapping onto 10 randomly selected 100-D tensors, 
            with components in the range -1.0 to 1.0, 100% steady accuracy 
            is acheived at iteration 80~100
            4. with mapping onto 3, 0-D tensors, with values
            of -0.75 for Odd, +0.75 for Even, and 0 for PAD, 
            100% steady accuracy is not acheived, even by iteration 1000.
        
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

NUM_OF_CLASSES = 2 #There are just two Classes ODD Sentences and EVEN Sentences

num_LSTM_layers = 2               

NUM_OF_TRAINING_RUNS = 500
NUM_OF_TEST_RUNS = 5

BATCH_SIZE = 128
#There are just 10 words in the vocabulary, so why have a dimensionality of
#64 for embedding?
embedding_space_dimensionality = 64
hidden_layer_size = 32

times_steps = WORDS_IN_A_SENTENCE #6 #TODO: 
#Is this connected to WORDS_IN_A_SENTENCE? Yes it is. 
#Comparing with the example of row-wise (col-wise) scan of an image: 
    #1. Each word corresponds to a single scan line -row or col. 
    #2. The number of the words corresponds to the number
    # of scan-lines used in scanning the image. 
    #3. The time to scan one row (col) is a time-step.
    #4. There are as many time-steps as there are scan-lines.
    #5. Thus time_steps correspond to WORDS_IN_A_SENTENCE

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
        #The criteria for classifying, whether a sentence is Odd or Even: 
        #if it consists, apart from PAD words, solely of Even words, then
        #it is an Even sentence.
        #if it consists, apart from PAD words, solely of Odd words, then
        #it is an Odd sentence.
        #If it has both Even and Odd words, then it is invalid
        #
        #Therefore if the Odd words are mapped onto one vector, the Even worda
        #onto another, and the PAD word onto a third, then the performnce
        #should not suffer.
        #
        #Unfortunately with the embedding table given below, the performance 
        #is very poor
#        embeddings_3point = \
#            np.asarray([
#                        [0.0] * embedding_space_dimensionality,
#                        [-75.0] * embedding_space_dimensionality,
#                        [+75.0] * embedding_space_dimensionality,
#                        [-75.0] * embedding_space_dimensionality,
#                        [+75.0] * embedding_space_dimensionality,
#                        [-75.0] * embedding_space_dimensionality,
#                        [+75.0] * embedding_space_dimensionality,
#                        [-75.0] * embedding_space_dimensionality,
#                        [+75.0] * embedding_space_dimensionality,
#                        [-75.0] * embedding_space_dimensionality,
#                        ]) #The odd ids are mapped to -75.0 
                            #Even ids are mapped to +75.0
                            #PAD is mapped to 0.0
        ##############################################################
        #Again the word_ids are mapped to 3 distinct vectors.
        #In this case the components of the vectors are random.
        #The performance appears to be just as good as when all unique words
        #have differing vectors
#                Accuracy at 0: 50.78125
#                Accuracy at 100: 51.56250
#                Accuracy at 200: 100.00000
#                Accuracy at 300: 100.00000
#                Accuracy at 400: 100.00000
#                Test batch accuracy 0: 100.00000
#                Test batch accuracy 1: 100.00000
#                Test batch accuracy 2: 100.00000
#                Test batch accuracy 3: 100.00000
#                Test batch accuracy 4: 100.00000  
        
        #With the full embedding table it is
#                Accuracy at 0: 51.56250
#                Accuracy at 100: 100.00000
#                Accuracy at 200: 100.00000
#                Accuracy at 300: 100.00000
#                Accuracy at 400: 100.00000
#                Test batch accuracy 0: 100.00000
#                Test batch accuracy 1: 100.00000
#                Test batch accuracy 2: 100.00000
#                Test batch accuracy 3: 100.00000
#                Test batch accuracy 4: 100.00000
        random_nums = (np.random.uniform(-1, 1, embedding_space_dimensionality))
        embeddings_3point = \
            np.asarray([
                random_nums *  0.0,
                random_nums * -1.0,
                random_nums * +1.0,
                random_nums * -1.0,
                random_nums * +1.0,
                random_nums * -1.0,
                random_nums * +1.0,
                random_nums * -1.0,
                random_nums * +1.0,
                random_nums * -1.0
                    ])
        #assert that there are just 3 distinct points
        #in the embedding table
        assert(embeddings_3point[1].tolist() == \
               embeddings_3point[3].tolist())
        assert(embeddings_3point[1].tolist() == \
               embeddings_3point[5].tolist())
        assert(embeddings_3point[1].tolist() == \
               embeddings_3point[7].tolist())
        assert(embeddings_3point[1].tolist() == \
               embeddings_3point[9].tolist())
        
        assert(not(embeddings_3point[0].tolist() == \
               embeddings_3point[2].tolist()))
        assert(not(embeddings_3point[1].tolist() == \
               embeddings_3point[2].tolist()))
        
        assert(embeddings_3point[2].tolist() == \
               embeddings_3point[4].tolist())
        assert(embeddings_3point[2].tolist() == \
               embeddings_3point[2].tolist())
        assert(embeddings_3point[2].tolist() == \
               embeddings_3point[8].tolist())
        #
        #Also see: /home/rm/Sandlot-TF-Misc/word_embeddings.py 
        #################################################################
        #Construct TensorFlow Graph
            
#        _inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE, WORDS_IN_A_SENTENCE])
        _inputs = tf.placeholder(tf.int32, shape=[None, WORDS_IN_A_SENTENCE])
        _labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_OF_CLASSES])
        _sentence_lens = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        

        with tf.name_scope("embeddings"):

            USE_3point_MAP = False 
            
            if(False == USE_3point_MAP):
                embeddings = tf.Variable(
                    tf.random_uniform([number_of_distinct_words_found,
                                       embedding_space_dimensionality],
                                      -1.0, 1.0), name='embedding')
            else:
                embeddings =tf.Variable(embeddings_3point, dtype=tf.float32)
            ###########################################################
            #EMBED the input into the embeddings space
            #SANITY CHECKS
            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            assert(number_of_distinct_words_found == \
                   len(embeddings.eval()))
            assert(number_of_distinct_words_found == \
                   (np.asarray(embeddings.eval())).shape[0])
            assert(embedding_space_dimensionality == \
                   (np.asarray(embeddings.eval())).shape[1])
            sess.close()
            
            embed = tf.nn.embedding_lookup(embeddings, _inputs)
            #Copy of assertion on embed in main session below
#            assert((BATCH_SIZE, \
#                    WORDS_IN_A_SENTENCE, \
#                    embedding_space_dimensionality) == word_embeddings.shape)
#            Each word in the sentence is looked up in the table_of_embeddings.
#            This maps each unique word_id to a random point having dimensionality
#            of embedding_space_dimensionality
            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            Test_Sentences_0 = \
                        [\
                             [0, 1, 2, 3, 4, 5],
                             [0, 1, 2, 3, 4, 5],
                             [0, 1, 2, 3, 4, 5]
                         ]

            word_embeddings = sess.run([embed], \
                                    feed_dict={_inputs: Test_Sentences_0})
            assert(4 == np.asarray(word_embeddings).ndim)
            #The embeddings of all the sentences are put in 1 list.
            assert(1 == len(word_embeddings))
            #Three sentences were fed. So the top level list has three elements
            assert(3 == len(word_embeddings[0]))
            #For each word_id in a sentence, there is a vector to which it is mapped
            assert(WORDS_IN_A_SENTENCE == len(word_embeddings[0][0]))
            #The dimensionality of the vector to which a word is mapped is
            #embedding_space_dimensionality
            assert(embedding_space_dimensionality == len(word_embeddings[0][0][0]))
            #The following asserts show that for the 3 identical sentences
            #used, identical embeddings (of shape [WORDS_IN_A_SENTENCE, 
            # embedding_space_dimensionality]) are returned .
            #
            #HENCE, it is safe to conclude that for a given unique word_id the same vector is 
            #returned, irrespective of different occurencess
            assert(word_embeddings[0][0].tolist() == \
                   word_embeddings[0][1].tolist())
            assert(word_embeddings[0][0].tolist() == \
                   word_embeddings[0][2].tolist())

            #WHAT-IF the number of unique words exceeds the rows in the
            #embedding table?
#            Attempt to run the following (with necessary changes to placeholder for
#            _inputs, and commenting out the earlier asserts, results in:
#            InvalidArgumentError (see above for traceback): 
#                indices[0,10] = 10 is not in [0, 10)
#            Test_Sentences_0 = \
#                        [\
#                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#                         ]
#
#            word_embeddings = sess.run([embed], \
#                                    feed_dict={_inputs: Test_Sentences_0})
            
#            QUESTION
#            See https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup
#            It says
#            "If len(params) > 1, each element id of ids is partitioned between 
#            the elements of params according to the partition_strategy. 
#            In all strategies, if the id space does not evenly divide the 
#            number of partitions, each of the first (max_id + 1) % len(params) 
#            partitions will be assigned one more id."
#            
#            What does that mean?

            sess.close()
            

        with tf.variable_scope("lstm"):
            pass
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
        
            x_batch, y_batch, seqlen_batch = get_sentence_batch(BATCH_SIZE,
                                                                train_x, train_y,
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
        
                if step % 100 == 0:
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
#            states_example = sess.run([states[1]], feed_dict={_inputs: x_test,
            states_example = sess.run([states], feed_dict={_inputs: x_test,
                                                              _labels: y_test,
                                                              _sentence_lens: seqlen_test})
            
            assert((1,BATCH_SIZE, WORDS_IN_A_SENTENCE, hidden_layer_size) == \
                   np.asarray(output_example).shape)
            assert((1, NUM_OF_CLASSES, BATCH_SIZE, hidden_layer_size) == \
                   np.asarray(states_example).shape)

            print("\n\tDONE:", __file__, "\n")
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
      