#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 10:46:26 2018

@author: rm
"""
import numpy as np

Label_ODD = 0 #for odd sentence
Label_EVEN = 1 #for even sentence
Label_NOTA = 2 #for NOTA sentence

word2index_map = {} #NOTE: This is going to be a dictionary

NUM_OF_CLASSES = 3

def labels_to_one_hot(list_of_labels):
    list_of_one_hot_labels = [[0, 0, 0]] * len(list_of_labels)
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
        list_of_one_hot_labels[i] = one_hot_encoding
    return list_of_one_hot_labels
            
#############################################################
#UTILTY to be used to get batches of the generated sentences
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
    
def get_sentence_batch(batch_size, \
                       data_x, \
                       data_y, \
                       data_x_sentence_lengths, \
                       WORDS_IN_A_SENTENCE):
    
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

def generate_data_sentences(NUM_OF_SENTENCES, \
                            MIN_LEN_OF_UNPADDED_SENTENCE, \
                            MAX_LEN_OF_UNPADDED_SENTENCE, \
                            MIN_ODD_NUM, \
                            MIN_EVEN_NUM, \
                            NUM_RANGE, \
                            WORDS_IN_A_SENTENCE, \
                            digit_to_word_map):
    list_of_even_sentences = []
    list_of_odd_sentences = []
    list_of_NOTA_sentences = []
    list_of_sentence_lens = []

    WORDS_IN_VOCABULARY = len(digit_to_word_map)
    
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
    list_of_labels_ = [Label_EVEN] * (NUM_OF_SENTENCES//2) + \
                        [Label_ODD] * (NUM_OF_SENTENCES//2) + \
                        [Label_NOTA] * (NUM_OF_SENTENCES//2) 
    ##################################################################
    #PRE_PROCESS the sentences
    list_of_one_hot_labels = labels_to_one_hot(list_of_labels_)
    
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
    #to list_of_one_hot_labels and list_of_sentence_lens
    array_of_sentences = np.array(list_of_sentences)[list_of_indices]
    array_of_labels = np.array(list_of_one_hot_labels)[list_of_indices]
    array_of_sentence_lengths = np.array(list_of_sentence_lens)[list_of_indices]
    
    #split the shuffled sentences, equally, into training and testing data
    train_x = array_of_sentences[:NUM_OF_SENTENCES//2]
    train_y = array_of_labels[:NUM_OF_SENTENCES//2]
    train_sentence_lens = array_of_sentence_lengths[:NUM_OF_SENTENCES//2]
    
    test_x = array_of_sentences[NUM_OF_SENTENCES//2:]
    test_y = array_of_labels[NUM_OF_SENTENCES//2:]
    test_sentence_lens = array_of_sentence_lengths[NUM_OF_SENTENCES//2:]
    
    return train_x, \
            train_y, \
            train_sentence_lens, \
            test_x, \
            test_y, \
            test_sentence_lens, \
            number_of_distinct_words_found

