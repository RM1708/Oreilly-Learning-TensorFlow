
# coding: utf-8
'''
 FROM: Simple Introduction to Tensorboard Embedding Visualisation
 by Roland Meertens on April 19, 2017	
     "Visualising embeddings is a powerful technique! 
     It helps you understand what your algorithm learned, and if this is 
     what you expected it to learn. 
     Embedding visualisation is a standard feature in Tensorboard. 
     Unfortunately many people on the internet seem to have some problems 
     with getting a simple visualisation running. 
     This is my attempt at creating the most simple code to get a simple 
     visualisation of MNIST digits running."
     
The file EmbeddingVisualization.py in the folder
    /home/rm/Sandlot-TensorFlow/SimpleIntroTFEmbeddingVisualization/
has been modified so that it instead of MNIST data it takes the sentence data

'''


import tensorflow as tf
import numpy as np
import os

from tensorflow.contrib.tensorboard.plugins import projector


from Data_Even_Odd_NOTA_Sentences import generate_data_sentences, \
                                        get_sentence_batch
                                        
#GENERATE simulated words
#Map is needed as random ints can be generated which can then be mapped to words
#in our vocabulary
digit_to_word_map = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                     6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
digit_to_word_map[0] = "PAD"
#
WORDS_IN_A_SENTENCE = 6
NUM_OF_RUN_ITERS = 5000
NUM_OF_TEST_ITERS = 5
NUM_RANGE = len(digit_to_word_map)
NUM_OF_SENTENCES = 20000
MIN_LEN_OF_UNPADDED_SENTENCE = 3
MAX_LEN_OF_UNPADDED_SENTENCE = WORDS_IN_A_SENTENCE   #6
MIN_ODD_NUM = 1
MIN_EVEN_NUM = 2


#MNIST is not the input data
#LOG_DIR = "/home/rm/Sandlot-TensorFlow/SimpleIntroTFEmbeddingVisualization/minimalsample"
LOG_DIR = "/home/rm/logs/Visualization-Data-Odd_Even_NOTA_Sentences"
NAME_TO_VISUALISE_VARIABLE = "WORD_EMBEDDING"
TO_EMBED_COUNT = 500

#MNIST_DATA_DIR = "/home/rm/tmp/data"
#
#
#path_for_mnist_sprites =  os.path.join(LOG_DIR,'mnistdigits.png')
#path_for_mnist_metadata =  os.path.join(LOG_DIR,'metadata.tsv')
path_for_sentences_metadata = os.path.join(LOG_DIR,'metadata.tsv')
#
################################################################
#NOT PROCESSING IMAGES. SPRITES NOT RELEVANT
#def create_sprite_image(images):
#    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
#    if isinstance(images, list):
#        images = np.array(images)
#    img_h = images.shape[1]
#    img_w = images.shape[2]
#    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
#    
#    
#    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))
#    
#    for i in range(n_plots):
#        for j in range(n_plots):
#            this_filter = i * n_plots + j
#            if this_filter < images.shape[0]:
#                this_img = images[this_filter]
#                spriteimage[i * img_h:(i + 1) * img_h,
#                  j * img_w:(j + 1) * img_w] = this_img
#    
#    return spriteimage

#def vector_to_matrix_mnist(mnist_digits):
#    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
#    return np.reshape(mnist_digits,(-1,28,28))
#
#def invert_grayscale(mnist_digits):
#    """ Makes black white, and white black """
#    return 1-mnist_digits
###################################################################
# 
# ### What to visualise
# 
# Although the embedding visualiser is meant for visualising 
# embeddings obtained after training, you can also use it to apply 
#visualisation of normal MNIST digits. 
#In this case, each digit is represented by a vector with 
#length 28*28=784 dimensions.
#
#mnist = input_data.read_data_sets(MNIST_DATA_DIR, #"MNIST_data/", 
#                                  one_hot=False)
#batch_xs, batch_ys = mnist.train.next_batch(TO_EMBED_COUNT)
#
#We need the sentences that are to be handled
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

# ### Creating the embeddings
# 
# For this example the embeddings are extremely simple: 
# they are the direct values of the traindata. 
# Your graph will probably be more complicated, 
# but the important thing is that you know the name of the variable 
# you want to visualise

#embedding_var = tf.Variable(batch_xs, name=NAME_TO_VISUALISE_VARIABLE)
batch_size = 500
x_batch, y_batch, seqlen_batch = \
                            get_sentence_batch(batch_size, \
                                                train_x, 
                                                train_y,
                                                train_sentence_lens, \
                                                WORDS_IN_A_SENTENCE)
embedding_var = tf.Variable(x_batch, name=NAME_TO_VISUALISE_VARIABLE)
summary_writer = tf.summary.FileWriter(LOG_DIR)


# ### Create the embedding projectorc
# 
# This is the important part of your embedding visualisation. 
# Here you specify what variable you want to project, 
# what the metadata path is (the names and classes), 
# and where you save the sprites.
# 
# We will create the sprites later!

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# Specify where you find the metadata
embedding.metadata_path = path_for_sentences_metadata #'metadata.tsv'

# Specify where you find the sprite (we will create this later)
#embedding.sprite.image_path = path_for_mnist_sprites #'mnistdigits.png'
#embedding.sprite.single_image_dim.extend([28,28])

# Say that you want to visualise the embeddings
projector.visualize_embeddings(summary_writer, config)


# ### Saving the data
# 
# Tensorboard loads the saved variable from the saved graph. 
# Initialise a session and variables, and save them in your 
# logging directory.
# 

#sess = tf.InteractiveSession()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #See: https://www.tensorflow.org/api_docs/python/tf/train/Saver#save
    saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)
    
    
    # ### Visualisation helper functions
    # 
    # Mentioned above are the sprites. 
    # If you don’t load sprites each digit is represented as a simple point (does not give you a lot of information). To add labels you have to create a ‘sprite map’: basically all images in what you want to visualise…
    # 
    # There are three functions which are quite important for the visualisation:
    # 
    #     create_sprite_image: neatly aligns image sprits on a square canvas, as specified in the images section here: (https://www.tensorflow.org/get_started/embedding_viz)
    #     vector_to_matrix_mnist: MNIST characters are loaded as a vector, not as an image… this function turns them into images
    #     invert_grayscale: matplotlib treats a 0 as black, and a 1 as white. The tensorboard embeddings visualisation looks way better with white backgrounds, so we invert them for the visualisation
    # 
    # 
    
    # ### Save the sprite image
    # 
    # Pretty straightforward: convert our vectors to images, invert the grayscale, and create and save the sprite image.
    # 
#NO SPRITES. DEALING WITH SENTENCES; NOT IMAGES    
#    to_visualise = batch_xs
#    to_visualise = vector_to_matrix_mnist(to_visualise)
#    to_visualise = invert_grayscale(to_visualise)
#    
#    sprite_image = create_sprite_image(to_visualise)
#    
#    plt.imsave(path_for_mnist_sprites,sprite_image,cmap='gray')
#    plt.imshow(sprite_image,cmap='gray')
    
    
    # ### Save the metadata
    # 
    # To add colors to your mnist digits the embedding visualisation 
    #tool needs to know what label each image has. 
    #This is saved in a “TSV (tab seperated file)”.
    # 
    # Each line of our file contains the following:
    # 
    # "Index" , "Label" 
    # 
    # The Index is simply the index in our embedding matrix. The label is the label of the MNIST character.
    # 
    # This code writes our data to the metadata file.
    # 
    with open(path_for_sentences_metadata,'w') as f:
        f.write("Index\tLabel\n")
#        y_batch_list = y_batch.tolist()
        index = 0
#        for index,label in enumerate(train_y_list):
        for label in (y_batch):
            f.write("%d\t%d\n" % (index, np.argmax(label)))
            index += 1
            
###########################################################################    
# ### How to run
# We saved our MNIST characters, time to visualise it! If you did not change any of the variables above you can run the visualisation with:
# 
# tensorboard –logdir=minimalsample
# 
# Now open a browser and navigate to http://127.0.0.1:6006 (note: this can change depending on your computer setup). You should see this after navigating to the Embeddings tab (note: if you have an older tensorflow version you will NOT see the Embeddings tab. This can only be resolved by upgradeing Tensorflow):

# Click the embeddings tab to see the PCA of our MNIST digits. Click on the left on the “color by” selector and select the Label. You probably see some nice groupings (zeroes close to each other, sixes close to each other, etc.).

print("\n\tDONE: ", __file__, "\n")
