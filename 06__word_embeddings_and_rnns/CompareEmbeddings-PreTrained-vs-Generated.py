# -*- coding: utf-8 -*-
"""

@author: RM

@ANALYSIS & FIX:
    1. Why does PAD_TOKEN get evaluated for cosine distance in the case of
    Generated embeddings but not for GloVe one? 
        1. PAD_TOKEN is not present in GloVe embedding. Hence, when the
        initialization matrix is constructed, it is only with the vectors for the
        other words in the Vocabulary ("One", ... "Nine"). Since the 
        initialization matrix was initialized to 0s, a 0-vector is present
        in the word-id of PAD_TOKEN. When computing unit vectors, 
        this results in a NaN caused by the div by 0. Initializing the 
        initialization matrix to 1.0, takes care of this, and PAD_TOKEN is  
        evaluated.
    2. Why is the ref word not printed
        1. Each word is arranged in descending order of the dot-product between its 
        unit embedded vector and the unit embedded vector of a ref word. The 
        dot product of the unit ref vector with itself would be the largest 
        (== 1.0). The print indexing for printing the ordered list started at 1; 
        not at 0. Fixed that.
        
@HYPOTHESIS
    1. Why have GloVe embeddings? The performance when classifying ODD 
    sentences and EVEN sentences seem to be the same.
    2. My hypothesis is that pretrained embeddings reflect the proximity of words.
    In the present case the proximity to the ref word. Thus taking the ref word
    as "One" we get a proximity order that follows the number sequence. "Two" is
    closest. "Nine" is farthest. PAD_TOKEN, though it is 0 is the farthest. This,
    I believe, is owing to it having a locally assigned value; not one from the
    GloVe embeddings. The first 2 million lines of GloVe file were checked. 
    PAD_TOKEN was not found. There is good chance it does not exist. Even if it 
    does, for the purposes of the exercise it is "NaN" though it figures at 
    index 0 of digit_to_word_map.
"""
import zipfile
import numpy as np
import tensorflow as tf

path_to_glove = "/home/rm/Downloads/glove.840B.300d.zip"
GLOVE_SIZE = 300

#NUM_OF_SENTENCES = 20000
#WORDS_PER_SENTENCE = 6
#
#times_steps = WORDS_PER_SENTENCE

digit_to_word_map = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                     6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
digit_to_word_map[0] = "PAD_TOKEN"

word2index_map = {"PAD_TOKEN": 0, "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5,
                     "Six": 6, "Seven": 7, "Eight": 8, "Nine": 9}
index2word_map = {index: word for word, index in word2index_map.items()}

VOCABULARY_SIZE = len(index2word_map)
assert(len(digit_to_word_map) == VOCABULARY_SIZE)#There is a small probability
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
            line_count = 0
            for line in f:
                line_count += 1
                vals = line.split()
                word = str(vals[0].decode("utf-8"))
                assert("PAD_TOKEN" != word)
                if (word in word2index_map and \
                    (count_all_words < len(word2index_map) - 1)):
#                    print(word)
                    count_all_words += 1
                    coefs = np.asarray(vals[1:], dtype='float32')
                    coefs /= np.linalg.norm(coefs)
                    embedding_weights[word] = coefs
                if count_all_words == len(word2index_map) - 1: #-1 accounts for the
                                                            #missing PAD_TOKEN
                    break
#                Checked for 2 Million lines assert holds.
#                To check comment break, uncomment below
#                    if(0 == line_count % 10000):
#                        print("Line Count: ", line_count)
#                    continue
                
    print("Lines read from glove.840B.300d.txt: ", line_count)
    return embedding_weights


embedding_placeholder = tf.placeholder(tf.float32, [VOCABULARY_SIZE,
                                                    GLOVE_SIZE])
###########################################
USE_GLOVE = True
EVAL_TENSORS = True
###########################################
if(USE_GLOVE):
    word2embedding_dict = get_glove(path_to_glove, word2index_map)
    assert(len(word2index_map) == len(word2embedding_dict) + \
                                   1)  #Account for PAD_TOKEN not present
                                       #in word2embedding_dict
    embedding_matrix = np.ones((VOCABULARY_SIZE, GLOVE_SIZE))
    assert([1.0] * GLOVE_SIZE == \
           [(embedding_matrix.tolist()[0][i]) for i in \
            range(GLOVE_SIZE)])
    
    for word, index in word2index_map.items():
        if not word == "PAD_TOKEN":
            assert(GLOVE_SIZE == len(word2embedding_dict[word]))
            assert((GLOVE_SIZE,) == word2embedding_dict[word].shape)
            word_embedding = word2embedding_dict[word]
            embedding_matrix[index, :] = word_embedding

    assert([1.0] * GLOVE_SIZE == \
           [(embedding_matrix.tolist()[0][i]) for i in \
            range(GLOVE_SIZE)])


if USE_GLOVE:
        embeddings = tf.Variable(tf.constant(0.0, \
                                shape=[VOCABULARY_SIZE, GLOVE_SIZE]),
                                 trainable=True)
        # if using GloVe embeddings, assign them to the embeddings variable
        embedding_init = embeddings.assign(embedding_placeholder)
else:
        embeddings = tf.Variable(
            tf.random_uniform([VOCABULARY_SIZE,
                              # embedding_dimension], #why not GLOVE_SIZE
                              GLOVE_SIZE],
                              -1.0, 1.0, seed=1234))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
if(USE_GLOVE):
    sess.run(embedding_init,
             feed_dict={embedding_placeholder: embedding_matrix})

if (not EVAL_TENSORS):
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),
                                 1, keepdims=True))
    normalized_embeddings = embeddings / norm #norm is a tensor, so is embeddings
                            #<tf.Tensor 'Sqrt:0' shape=(10, 1) dtype=float32>
                            #<tf.Variable 'Variable_1:0' shape=(10, 300) dtype=float32_ref>
    normalized_embeddings_matrix = sess.run(normalized_embeddings)
else:        
    '''
    This variant exposes the probable cause, for PAD_TOKEN not
    showing up. Gives the run time warning. (Run with USE_GLOVE:  True)
        RuntimeWarning: invalid value encountered in true_divide
                normalized_embeddings = embedding_unnorm / norm_val
    
    The code completes.
    '''
    embedding_unnorm =sess.run(embeddings)
#    embedding_unnorm = tf.identity(embeddings) #This is equivalent to the
                                                #previous statement
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_unnorm),
                                 1, keepdims=True))
    norm_val = sess.run(norm)
    normalized_embeddings = embedding_unnorm / norm_val #Both are 
                                                    #<class 'numpy.ndarray'>
    normalized_embeddings_matrix = normalized_embeddings

sess.close()

ref_word = normalized_embeddings_matrix[word2index_map["One"]]        #Three"]]

cosine_dists = np.dot(normalized_embeddings_matrix, ref_word)
'''
See
Extended slicing: The third limit and slice objects
Lutz, Mark. Learning Python (p. 203). O'Reilly Media. Kindle Edition. 

[::-1] reverses the order
'''
#ff = np.argsort(cosine_dists)[::-1][0:VOCABULARY_SIZE]
ff = np.argsort(cosine_dists)[::-1] #This gives the same result as the one above
print("USE_GLOVE: ", USE_GLOVE)
print("EVAL_TENSORS: ", EVAL_TENSORS)
for f in ff:
    print("Word: {}. cosine_dist: {}".format(index2word_map[f], cosine_dists[f]))

print("\n\tDONE: ", __file__)

'''
USE_GLOVE:  False
EVAL_TENSORS:  False
Word: One. cosine_dist: 1.0000001192092896
Word: Three. cosine_dist: 0.11146818101406097
Word: Nine. cosine_dist: 0.04966232553124428
Word: Two. cosine_dist: 0.027441823855042458
Word: Seven. cosine_dist: 0.015385191887617111
Word: Four. cosine_dist: -0.0011444836854934692
Word: Eight. cosine_dist: -0.03179740533232689
Word: PAD_TOKEN. cosine_dist: -0.03755336254835129
Word: Five. cosine_dist: -0.0442391037940979
Word: Six. cosine_dist: -0.047348640859127045

        DONE:  /home/rm/Code-LearningTF/06__word_embeddings_and_rnns/
        CompareEmbeddings-PreTrained-vs-Generated.py        

USE_GLOVE:  False
EVAL_TENSORS:  True
Word: One. cosine_dist: 1.0000001192092896
Word: Three. cosine_dist: 0.11146818101406097
Word: Nine. cosine_dist: 0.04966232553124428
Word: Two. cosine_dist: 0.027441823855042458
Word: Seven. cosine_dist: 0.015385191887617111
Word: Four. cosine_dist: -0.0011444836854934692
Word: Eight. cosine_dist: -0.03179740533232689
Word: PAD_TOKEN. cosine_dist: -0.03755336254835129
Word: Five. cosine_dist: -0.0442391037940979
Word: Six. cosine_dist: -0.047348640859127045

        DONE:  /home/rm/Code-LearningTF/06__word_embeddings_and_rnns/
        CompareEmbeddings-PreTrained-vs-Generated.py

QUESTION.
--------
    Q1. Why is the ref_word ("One") missing from USE_GLOVE:  False, but not
    from USE_GLOVE:  True?
    Q2. Why is the word PAD_TOKEN missing from USE_GLOVE:  True, but not
    from USE_GLOVE:  False?
    
FIXED
-----
    1. See initialization of  embedding_matrix. np.zeros changed to np.ones
    2. ff = np.argsort(cosine_dists)[::-1][1:VOCABULARY_SIZE] changed to
    ff = np.argsort(cosine_dists)[::-1][0:VOCABULARY_SIZE]

USE_GLOVE:  True
EVAL_TENSORS:  False
Word: One. cosine_dist: 1.0000001192092896
Word: Two. cosine_dist: 0.6557701230049133
Word: Three. cosine_dist: 0.6331722736358643
Word: Five. cosine_dist: 0.6030528545379639
Word: Four. cosine_dist: 0.5933799743652344
Word: Six. cosine_dist: 0.5691505670547485
Word: Seven. cosine_dist: 0.5008598566055298
Word: Eight. cosine_dist: 0.49751731753349304
Word: Nine. cosine_dist: 0.44801828265190125
Word: PAD_TOKEN. cosine_dist: -0.008176516741514206

        DONE:  /home/rm/Code-LearningTF/06__word_embeddings_and_rnns/
        CompareEmbeddings-PreTrained-vs-Generated.py

USE_GLOVE:  True
EVAL_TENSORS:  True
Word: One. cosine_dist: 1.0000001192092896
Word: Two. cosine_dist: 0.6557701230049133
Word: Three. cosine_dist: 0.6331722736358643
Word: Five. cosine_dist: 0.6030528545379639
Word: Four. cosine_dist: 0.5933799743652344
Word: Six. cosine_dist: 0.5691505670547485
Word: Seven. cosine_dist: 0.5008598566055298
Word: Eight. cosine_dist: 0.49751731753349304
Word: Nine. cosine_dist: 0.44801828265190125
Word: PAD_TOKEN. cosine_dist: -0.008176516741514206

        DONE:  /home/rm/Code-LearningTF/06__word_embeddings_and_rnns/
        CompareEmbeddings-PreTrained-vs-Generated.py

  '''


'''
With USE_GLOVE == True
    ipdb> norm_val.shape
    (10, 1)
    
    ipdb> norm_val[0]
    array([0.99999994], dtype=float32)
    
    ipdb> norm_val
    array([[0.99999994],
           [0.99999994],
           [0.        ], <<<<<<<<<<<<<<<<<<<<<<<<<
           [0.99999994],
           [0.99999994],
           [0.99999994],
           [0.99999994],
           [0.99999994],
           [1.        ],
           [1.        ]], dtype=float32)
    
    ipdb> PAD_TOKEN
    *** NameError: name 'PAD_TOKEN' is not defined
    
    ipdb> word2index_map["PAD_TOKEN"]
    2                    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    ipdb> embedding_unnorm.shape
    (10, 300)
    
    ipdb> embedding_unnorm[2][:10]
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
    
    ipdb> embedding_unnorm[:, :10]
    array([[-3.82410698e-02, -2.46162172e-02, -7.61006325e-02,
            -5.92644922e-02, -3.35333943e-02, -8.07620138e-02,
            -3.32815275e-02, -4.18746099e-03,  6.90687522e-02,
             2.10900918e-01],
           [-5.33220870e-03, -4.36212234e-02, -6.42399937e-02,
            -1.03849836e-01, -3.01927049e-02, -4.74422090e-02,
            -3.10233533e-02,  8.27532075e-03,  3.95071767e-02,
             2.40813643e-01],
           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,     <<<<<<<<<<<<<<<<
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00],
           [-6.63952678e-02,  2.94664484e-02,  1.32901163e-03,
             1.17572876e-04, -5.19375578e-02, -6.84273019e-02,
            -8.68876129e-02,  8.57753213e-03, -5.48012462e-03,
             2.69940108e-01],
           [-2.27102675e-02,  9.10495780e-03, -3.77948396e-02,
            -6.84511438e-02, -2.19842810e-02, -5.97014092e-02,
            -2.71135680e-02,  3.04591656e-02,  7.68634751e-02,
             2.17530280e-01],
           [ 5.01343468e-03, -2.98853517e-02, -2.71744952e-02,
            -1.17817819e-01, -6.44267797e-02, -7.08082840e-02,
            -4.85444628e-02, -1.60385575e-02,  5.36621213e-02,
             1.55868694e-01],
           [-7.93781411e-03,  1.66579001e-02, -3.77636105e-02,
            -1.03353918e-01,  4.17818828e-03, -7.74870813e-02,
            -9.07040667e-03,  3.04094777e-02,  3.47289853e-02,
             2.02921987e-01],
           [ 2.42987578e-03,  3.30427960e-02, -1.49107128e-02,
            -5.98614104e-02, -3.58530395e-02, -5.84209412e-02,
            -7.69119943e-03,  2.03021374e-02,  6.86436519e-02,
             1.51136011e-01],
           [-2.96887010e-02,  4.35930975e-02, -5.90912849e-02,
            -7.38156214e-02, -1.39068943e-02, -1.04648784e-01,
            -2.50686239e-02,  4.32186536e-02,  8.76931399e-02,
             2.04331860e-01],
           [-3.17113847e-02, -6.56980872e-02, -7.10621253e-02,
            -4.09503467e-02, -5.11276983e-02, -9.20816734e-02,
            -6.52651265e-02, -3.86569090e-03,  5.19573987e-02,
             1.76612750e-01]], dtype=float32)

    ipdb> embedding_unnorm / norm_val
    /home/rm/Code-LearningTF/06__word_embeddings_and_rnns/CompareEmbeddings-PreTrained-vs-Generated.py:1: 
        RuntimeWarning: invalid value encountered in true_divide
      # -*- coding: utf-8 -*-
    array([[-0.03824107, -0.02461622, -0.07610064, ..., -0.05147702,
             0.04443773, -0.075308  ],
           [-0.00533221, -0.04362123, -0.06424   , ...,  0.00603761,
             0.0543786 , -0.10800309],
           [        nan,         nan,         nan, ...,         nan,
                    nan,         nan],  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ...,
           [ 0.00242988,  0.0330428 , -0.01491071, ..., -0.07107946,
             0.02411918, -0.09388999],
           [-0.0296887 ,  0.0435931 , -0.05909128, ..., -0.01779287,
             0.02460201, -0.07145758],
           [-0.03171138, -0.06569809, -0.07106213, ..., -0.06592634,
            0.0335936 , -0.09450191]], dtype=float32)
'''
