# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 00:39:23 2016

@author: tomhope

@Modified: RM, Aug, 2018
@CHANGES
    1. Purpose-Indicating Naming.
    2. Both Left and Right Context Words extracted.
    3. Only Left selected for use
    4. Explanatory comments.
    5. Asserted my understanding.
    6. Magic numbers removed
    7. Code added to facilitate understanding of the construction of Skip-Grams. 
    Switch EXPLORE set to True selects the code. Code is a bunch of asserts and 
    prints.

@TODO
-----
    1. Right context words not relevant. Only the left context is valid for 
    predicting the TARGET_WORD. So why generte it at all? 
        From: (Kindle Locations 3496-3500). O'Reilly Media. Kindle Edition. 
        "...The major advantage of this representation is its ability to 
        capture the context of words from both directions, which enables 
        richer understanding of natural language and the underlying semantics 
        in text. In practice, in complex tasks, it often leads to improved 
        accuracy.   For example, in part-of-speech (POS) tagging, we want to 
        output a predicted tag for each word in a sentence 
        (such as “noun,” “adjective,” etc.). In order to predict a POS tag 
        for a given word, it is useful to have information on its surrounding 
        words, from both directions."

    2. Make WORDS_PER_PHRASE > 3. Make TARGET_WORD slide from left to right.
    3. Check processing time reduces for lower NUM_RANDOM_SAMPLES_PER_BATCH.
    4. For each word that appears in the left context, 
        1. what is the histogram of the TARGET WORDS?
        2. Histogram of predicted words. (Where are the predictions?)
"""
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

EXPLORE = False     #Switch

EMBEDDING_DIMENSIONALITY = 5
LOG_DIR = "/home/rm/logs/word2vec_intro"


digit_to_word_map = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                     6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
list_of_phrases = []

SIZE_OF_VOCAB = len(digit_to_word_map)
if(EXPLORE):
    NUM_OF_PHRASES = 20
    BATCH_SIZE = 8
    TRAINING_ITERS = 20
    CHECK_AT = 1
    NUM_RANDOM_SAMPLES_PER_BATCH = BATCH_SIZE
else:
    NUM_OF_PHRASES = 20000
    BATCH_SIZE = 64
    TRAINING_ITERS = 3000
    CHECK_AT = 100
    NUM_RANDOM_SAMPLES_PER_BATCH =  9
    '''
    NUM_RANDOM_SAMPLES_PER_BATCH values 1 thru 9, Is OK.
    Min value of NUM_RANDOM_SAMPLES_PER_BATCH is 1.
    
    Lower the NUM_RANDOM_SAMPLES_PER_BATCH, shorter the processing time. CHECK
    
    NUM_RANDOM_SAMPLES_PER_BATCH value of 10 throws an error 
    InvalidArgumentError ...: Sampler's range is too small.
	 [[Node: nce_loss/LogUniformCandidateSampler = 
         LogUniformCandidateSampler[num_sampled=10, 
         num_true=1, range_max=9, seed=0, seed2=0, 
         unique=true, ...
     '''
    
WORDS_PER_PHRASE = 3
CONTEXT_LEFT = -1; CONTEXT_RIGHT = +1; TARGET_WORD = 0

# Create two kinds of phrases - sequences of odd and even digits.
for i in range(NUM_OF_PHRASES//2):
    # WORDS_PER_PHRASE numbers are picked at random from the set {1, 3, 5, 7, 9} and returned as
    # an array
    rand_odd_ints = np.random.choice(range(1, 10, 2), WORDS_PER_PHRASE)
    #The ints in the array are taken as keys and the corresponding values retreived
    # from the dictionary digit_to_word_map and joined into a sequence(phrase).
    # the phrase is then appended to the list of sentences
    list_of_phrases.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
    
    #WORDS_PER_PHRASE numbers are picked at random from the set {2, 4, 6, 8}
    rand_even_ints = np.random.choice(range(2, 10, 2), WORDS_PER_PHRASE)
    list_of_phrases.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))

assert(NUM_OF_PHRASES == len(list_of_phrases))
# Map words to indices
word2index_map = {}
index = 0
for phrase in list_of_phrases:
    for word in phrase.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1
#Form the dictionary for reverse look-up
index2word_map = {index: word for word, index in word2index_map.items()}

unique_words_found = len(index2word_map)

#There is a remote chance that this assert may fail. It would mean that the number of
#iterations failed to generate all the words n the vocabulary
assert(unique_words_found == SIZE_OF_VOCAB)

# Generate skip-gram pairs
skip_gram_pairs = []
for phrase in list_of_phrases:
    tokenized_phrase = phrase.lower().split()
    assert(WORDS_PER_PHRASE == len(tokenized_phrase))
    for word_pos in range(1, len(tokenized_phrase)-1):
        assert(1 == word_pos)
        #Since WORDS_PER_PHRASE equals 3, it is only the center word that
        #has a context.
        # The module is valid only for phrases of 3 words
        word_index_left = word2index_map[tokenized_phrase[word_pos + \
                                                          CONTEXT_LEFT]]                        
        word_index_right = word2index_map[tokenized_phrase[word_pos + \
                                                           CONTEXT_RIGHT]] 
        word_index_center = word2index_map[tokenized_phrase[word_pos + \
                                                            TARGET_WORD]]

        skip_gram_pairs.append([word_index_center,
                               word_index_left])
        skip_gram_pairs.append([word_index_center,
                                word_index_right])
assert((NUM_OF_PHRASES * 2) == len(skip_gram_pairs)) 

def get_skipgram_batch(batch_size):
    assert(0 == (len(skip_gram_pairs) % 2)) #Has to be an even number
    assert(0 == (batch_size % 2))
    assert(batch_size <= len(skip_gram_pairs))
    instance_indices = list(range(len(skip_gram_pairs)//2))
    np.random.shuffle(instance_indices)
    skip_gram_indices_for_batch = instance_indices[:batch_size]
    skip_grams_pairs_for_batch = \
        [[skip_gram_pairs[(2 * i)], \
           skip_gram_pairs[(2 * i) + 1]] \
           for i in skip_gram_indices_for_batch]
    if(EXPLORE):
        print("\nSkip-Gram Indices For The Batch:\n", \
              skip_gram_indices_for_batch)
        print("Skip-Grams For The Batch:\n", \
              skip_grams_pairs_for_batch)
        
    words_indices_list = [skip_gram_pairs[2 * i][0] \
                          for i in skip_gram_indices_for_batch]
    contexts_indices_list = [[skip_gram_pairs[2 * i][1], \
                              skip_gram_pairs[(2 * i) + 1][1]] \
                             for i in skip_gram_indices_for_batch]
    return words_indices_list, contexts_indices_list


if(EXPLORE):
    EXPLORATORY_BATCH_SIZE = 8
    print("\nlist_of_phrases:\n", list_of_phrases[:5])
    print(list_of_phrases[5:10])
    print(list_of_phrases[10:15])
    print(list_of_phrases[15:])
    
    print("\nIndex To Word:\n", index2word_map)
    print("Word To Index:\n", word2index_map)

    print("\nskip-gram pairs for all phrases:\n", skip_gram_pairs[:10])
    print(skip_gram_pairs[10:20])
    print(skip_gram_pairs[20:30])
    print(skip_gram_pairs[30:])
    
    words_indices_list, \
    contexts_indices_list = get_skipgram_batch(EXPLORATORY_BATCH_SIZE)
    
    print("words_indices_list:\n",words_indices_list)
    print("contexts_indices_list:\n", contexts_indices_list)
    words_in_batch = [index2word_map[index] for index in words_indices_list]
    contexts_of_words_in_batch = [[index2word_map[index_pair[0]], \
         index2word_map[index_pair[1]]] for index_pair in contexts_indices_list]
    assert(len(words_in_batch) == len(contexts_of_words_in_batch))
    print("words in the batch:\n",words_in_batch)
    print("context of the words:\n",contexts_of_words_in_batch)
#################################################################
# GRAPH    
# Input data, labels
train_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name="TRAIN_INPUTS")
train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name="TRAIN_LABELS")

# Embedding lookup table currently only implemented in CPU
with tf.name_scope("embeddings"):
    embeddings = tf.Variable(
        tf.random_uniform([unique_words_found, EMBEDDING_DIMENSIONALITY],
                          -1.0, 1.0), name='embedding')
    # This is essentialy a lookup table
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Create variables for the NCE loss
nce_weights = tf.Variable(
        tf.truncated_normal([unique_words_found, EMBEDDING_DIMENSIONALITY],
                            stddev=1.0 / math.sqrt(EMBEDDING_DIMENSIONALITY)))
nce_biases = tf.Variable(tf.zeros([unique_words_found]))


loss = tf.reduce_mean(
                  tf.nn.nce_loss(weights=nce_weights, \
                                 biases=nce_biases, \
                                 inputs=embed, \
                                 labels=tf.cast(train_labels, tf.int64),
                                 num_sampled=NUM_RANDOM_SAMPLES_PER_BATCH, \
                                 num_classes=unique_words_found))
tf.summary.scalar("NCE_loss", loss)

# Learning rate decay
#global_step = tf.Variable(0, trainable=False)
#Since global_step is not going to change, it can be set as below
global_step = 0 
#From:
#    https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay
#decayed_learning_rate = learning_rate *
#                        decay_rate ^ (global_step / decay_steps)

learningRate = tf.train.exponential_decay(learning_rate=0.1,
                                          global_step=global_step,
                                          decay_steps=100,  #1000,
                                          decay_rate=0.95,
                                          #imaterial if staircase is set to False.
                                          #See comments above
                                          staircase=True)
#Ref: https://www.tensorflow.org/api_docs/python/tf/train/
#GradientDescentOptimizer#minimize
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(LOG_DIR,
                                         graph=tf.get_default_graph())
    saver = tf.train.Saver()

    with open(os.path.join(LOG_DIR, 'metadata.tsv'), "w") as metadata:
        metadata.write('Name\tClass\n')
        for k, v in index2word_map.items():
            metadata.write('%s\t%d\n' % (v, k))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
    projector.visualize_embeddings(train_writer, config)

    tf.global_variables_initializer().run()
    for step in range(TRAINING_ITERS):
        x_batch, y_batch = get_skipgram_batch(BATCH_SIZE)
        
        left_context_indices = [[row[0]] for row in y_batch]
#        left_context_indices = [row[0] for row in y_batch]
        #Note nesting of the elements row[0]
        #Not nesting, as in the commented statement above, throws:
            #ValueError: Cannot feed value of shape (64,) for Tensor 
            #'TRAIN_LABELS_2:0', which has shape '(64, 1)'
        ###########################################################
        #In this block convert to array, set the type and convert back 
        #again to list. See below for reason to go thru this
        left_context_indices = np.asarray(left_context_indices)
        #
        # NOTE: not converting to type np.int32, caused a *** lot of grief ***
        #Causes the error:
            #InvalidArgumentError ... : You must feed a value for placeholder 
            #tensor 'TRAIN_LABELS' with dtype int32 and shape [64,1]
            #	    [[Node: TRAIN_LABELS = Placeholder[dtype=DT_INT32, shape=[64,1], 
            #_device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]
        #
        left_context_indices = left_context_indices.astype(np.int32)
        #Now back to list
        left_context_indices = left_context_indices.tolist()
        ###########################################################
        summary, _ = sess.run([merged, train_step],
                              feed_dict={train_inputs: x_batch,
                                         train_labels: left_context_indices})
        train_writer.add_summary(summary, step)

        if step % CHECK_AT == 0:
            saver.save(sess, os.path.join(LOG_DIR, "w2v_model.ckpt"), step)
            loss_value = sess.run(loss,
                      feed_dict={train_inputs: x_batch,
                                 train_labels: left_context_indices})
            print("Loss at %d: %.5f" % (step, loss_value))
    # NOTE:
    #   1. Only training. No test.
    #   2. What was the point of introducing "noise/error" during training?
###########################################################################    
#The following was moved out into a session of its own.
# It can now be positioned anywhere after the definition of embeddings.
# NOTE: It does not need data to be fed to the placeholders
#
# Normalize embeddings before using.WHAT-IF you did not?
# Numbers do not look too different with or without normalization.
# Keeps magnitude of cosine_dists below 1.0   
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm
normalized_embeddings_matrix = sess.run(normalized_embeddings)

sess.close()
#

ref_word = normalized_embeddings_matrix[word2index_map["one"]]

cosine_dists = np.dot(normalized_embeddings_matrix, ref_word)
ff = np.argsort(cosine_dists)[::-1][1:10]
for f in ff:
    # SO?
    # See: Checking Out Our Embeddings
    # Learning TensorFlow: A Guide to Building Deep Learning Systems 
    #(Kindle Location 3230). O'Reilly Media. Kindle Edition. 
    print("word: {}, cosine_dists: {}".format(index2word_map[f], \
                                          cosine_dists[f]))
tf.reset_default_graph()   

print("\n\tDONE: ", __file__)