# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:18:27 2017

@author: tomhope

@MODIFIED: RM

@TODO:
    1. Check performance with NOTA sentences.See:
    /home/rm/Code-LearningTF/05__text_and_visualizations/\
    LSTM_embeddings-Odd-Even-NOTA.py

"""
import zipfile
import numpy as np
import tensorflow as tf

path_to_glove = "/home/rm/Downloads/glove.840B.300d.zip"
GLOVE_SIZE = 300
batch_size = 128
embedding_dimension = 64
num_classes = 2
hidden_layer_size = 32

NUM_OF_SENTENCES = 20000
WORDS_PER_SENTENCE = 6

times_steps = WORDS_PER_SENTENCE

digit_to_word_map = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                     6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
digit_to_word_map[0] = "PAD_TOKEN"

even_sentences = []
odd_sentences = []
seqlens = []
for i in range(NUM_OF_SENTENCES//2):
    rand_seq_len = np.random.choice(range(3, WORDS_PER_SENTENCE + 1))
    seqlens.append(rand_seq_len)
    rand_odd_ints = np.random.choice(range(1, 10, 2),
                                     rand_seq_len)
    rand_even_ints = np.random.choice(range(2, 10, 2),
                                      rand_seq_len)

    if rand_seq_len < 6:
        rand_odd_ints = np.append(rand_odd_ints,
                                  [0]*(WORDS_PER_SENTENCE-rand_seq_len))
        rand_even_ints = np.append(rand_even_ints,
                                   [0]*(WORDS_PER_SENTENCE-rand_seq_len))

    even_sentences.append(" ".join([digit_to_word_map[r]
                          for r in rand_odd_ints]))
    odd_sentences.append(" ".join([digit_to_word_map[r]
                         for r in rand_even_ints]))

data = even_sentences+odd_sentences
# same seq lengths for even, odd sentences
seqlens *= 2
labels = [1]*(NUM_OF_SENTENCES//2) + [0]*(NUM_OF_SENTENCES//2)
for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0]*2
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding

word2index_map = {}
index = 0
for sent in data:
    for word in sent.split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

index2word_map = {index: word for word, index in word2index_map.items()}

vocabulary_size = len(index2word_map)
assert(len(digit_to_word_map) == vocabulary_size)#There is a small probability
                                                #this may not hold

'''
There are 2.2 million words in the vocabulary of the pretrained GloVe 
embeddings we downloaded, and in our toy example we have only 9. 
So, we take the GloVe vectors only for words that appear in our own tiny 
vocabulary:

We go over the GloVe file line by line, take the word vectors we need, 
and normalize them. Once we have extracted the nine words we need, 
we stop the process and exit the loop. 

The output of our function is a dictionary, mapping from each word to its vector.

Learning TensorFlow: A Guide to Building Deep Learning Systems 
(Kindle Locations 3400-3401). O'Reilly Media. Kindle Edition. 
'''
def get_glove(path_to_glove, word2index_map):
    embedding_weights = {}
    count_all_words = 0
    with zipfile.ZipFile(path_to_glove) as z:
        with z.open("glove.840B.300d.txt") as f:
            for line in f:
                vals = line.split()
                word = str(vals[0].decode("utf-8"))
                assert("PAD_TOKEN" != word)
                if word in word2index_map:
                    print(word)
                    count_all_words += 1
                    coefs = np.asarray(vals[1:], dtype='float32')
                    coefs /= np.linalg.norm(coefs)
                    embedding_weights[word] = coefs
                if count_all_words == len(word2index_map) - 1:
                    break
    return embedding_weights

'''
Both trained and pre-trained appear to work the same - once embeddings dimensionality
for both case is made the same. 

So what's the advantage of using GLOVE?
'''
PRE_TRAINED = True

if(PRE_TRAINED):
    word2embedding_dict = get_glove(path_to_glove, word2index_map)
    assert(len(word2index_map) == len(word2embedding_dict) + \
                                   1)  #Account for PAD_TOKEN not present
                                       #in word2embedding_dict
    embedding_matrix = np.zeros((vocabulary_size, GLOVE_SIZE))
    
    for word, index in word2index_map.items():
        if not word == "PAD_TOKEN":
            assert(GLOVE_SIZE == len(word2embedding_dict[word]))
            assert((GLOVE_SIZE,) == word2embedding_dict[word].shape)
            word_embedding = word2embedding_dict[word]
            embedding_matrix[index, :] = word_embedding
    

data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
data = np.array(data)[data_indices]
labels = np.array(labels)[data_indices]
seqlens = np.array(seqlens)[data_indices]
train_x = data[:NUM_OF_SENTENCES//2]
train_y = labels[:NUM_OF_SENTENCES//2]
train_seqlens = seqlens[:NUM_OF_SENTENCES//2]

test_x = data[NUM_OF_SENTENCES//2:]
test_y = labels[NUM_OF_SENTENCES//2:]
test_seqlens = seqlens[NUM_OF_SENTENCES//2:]


def get_sentence_batch(batch_size, data_x,
                       data_y, data_seqlens):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word2index_map[word] for word in data_x[i].split()]
         for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x, y, seqlens


_inputs = tf.placeholder(tf.int32, shape=[batch_size, times_steps])
embedding_placeholder = tf.placeholder(tf.float32, [vocabulary_size,
                                                    GLOVE_SIZE])

_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
_seqlens = tf.placeholder(tf.int32, shape=[batch_size])

if PRE_TRAINED:
        embeddings = tf.Variable(tf.constant(10.0, \
                                             #0.0, \
                                shape=[vocabulary_size, GLOVE_SIZE]),
                                 trainable=True)
        # if using pre-trained embeddings, assign them to the embeddings variable
        embedding_init = embeddings.assign(embedding_placeholder)
        embed = tf.nn.embedding_lookup(embeddings, _inputs)

else:
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size,
                              # embedding_dimension], #why not GLOVE_SIZE
                              GLOVE_SIZE],
                              -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, _inputs)

'''
"Gated recurrent unit (GRU) cells are a simplification of sorts of LSTM cells. 
They also have a memory mechanism, but with considerably fewer parameters than LSTM. 
They are often used when there is less available data, and are faster to compute.

Learning TensorFlow: A Guide to Building Deep Learning Systems 
(Kindle Locations 3500-3503). O'Reilly Media. Kindle Edition. 
'''
with tf.name_scope("biGRU"):
    with tf.variable_scope('forward'):
        gru_fw_cell = tf.contrib.rnn.GRUCell(hidden_layer_size)
        gru_fw_cell = tf.contrib.rnn.DropoutWrapper(gru_fw_cell)

    with tf.variable_scope('backward'):
        gru_bw_cell = tf.contrib.rnn.GRUCell(hidden_layer_size)
        gru_bw_cell = tf.contrib.rnn.DropoutWrapper(gru_bw_cell)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell,
                                                          cell_bw=gru_bw_cell,
                                                          inputs=embed,
                                                          sequence_length=_seqlens,
                                                          dtype=tf.float32,
                                                          scope="biGRU")
states = tf.concat(values=states, axis=1)
weights = {
    'linear_layer': tf.Variable(tf.truncated_normal([2*hidden_layer_size,
                                                    num_classes],
                                                    mean=0, stddev=.01))
}
biases = {
    'linear_layer': tf.Variable(tf.truncated_normal([num_classes],
                                                    mean=0, stddev=.01))
}

# extract the final state and use in a linear layer
final_output = tf.matmul(states,
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
    print("\nPRE_TRAINED: ", PRE_TRAINED)
    if(PRE_TRAINED):
        sess.run(embedding_init,
                 feed_dict={embedding_placeholder: embedding_matrix})
    for step in range(1000):
        x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size,
                                                            train_x, train_y,
                                                            train_seqlens)
        sess.run(train_step, feed_dict={_inputs: x_batch, _labels: y_batch,
                                        _seqlens: seqlen_batch})

        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict={_inputs: x_batch,
                                                _labels: y_batch,
                                                _seqlens: seqlen_batch})
            print("Accuracy at %d: %.5f" % (step, acc))

# Why is this needed here? It can be moved out of this session and put
# where it is used
#    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),
#                                 1, keepdims=True))
#    normalized_embeddings = embeddings / norm
#    normalized_embeddings_matrix = sess.run(normalized_embeddings)

    for test_batch in range(5):
        x_test, y_test, seqlen_test = get_sentence_batch(batch_size,
                                                         test_x, test_y,
                                                         test_seqlens)
        batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy],
                                         feed_dict={_inputs: x_test,
                                                    _labels: y_test,
                                                    _seqlens: seqlen_test})
        print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
if(PRE_TRAINED):
    sess.run(embedding_init,
             feed_dict={embedding_placeholder: embedding_matrix})

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),
                             1, keepdims=True))
normalized_embeddings = embeddings / norm
normalized_embeddings_matrix = sess.run(normalized_embeddings)
sess.close()

ref_word = normalized_embeddings_matrix[word2index_map["One"]]        #Three"]]

cosine_dists = np.dot(normalized_embeddings_matrix, ref_word)
ff = np.argsort(cosine_dists)[::-1][1:10]
for f in ff:
    print("Word: {}. cosine_dist: {}".format(index2word_map[f], cosine_dists[f]))

'''
With PRE_TRAINED == False
Word: Three. cosine_dist: 0.0823516696691513
Word: Two. cosine_dist: 0.02026534453034401
Word: Nine. cosine_dist: 0.013921257108449936
Word: PAD_TOKEN. cosine_dist: -0.01276022382080555
Word: Six. cosine_dist: -0.017897550016641617
Word: Four. cosine_dist: -0.020764529705047607
Word: Eight. cosine_dist: -0.03127116709947586
Word: Seven. cosine_dist: -0.04936635121703148
Word: Five. cosine_dist: -0.05595245957374573

With PRE_TRAINED == True
Word: One. cosine_dist: 1.0000001192092896
Word: Two. cosine_dist: 0.6557701230049133
Word: Three. cosine_dist: 0.6331722736358643
Word: Five. cosine_dist: 0.6030528545379639
Word: Four. cosine_dist: 0.5933799743652344
Word: Six. cosine_dist: 0.5691505670547485
Word: Seven. cosine_dist: 0.5008598566055298
Word: Eight. cosine_dist: 0.49751734733581543
Word: Nine. cosine_dist: 0.44801828265190125
'''
tf.reset_default_graph()
print("\n\tDONE: ", __file__)