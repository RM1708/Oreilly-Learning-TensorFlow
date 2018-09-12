
# coding: utf-8

# In[ ]:


from tensorflow.contrib import slim
import sys
#from datasets import dataset_utils
import tensorflow as tf
#import urllib2
import urllib
from nets import vgg
from preprocessing import vgg_preprocessing
import os

sys.path.append("/home/rm/tensorflow_models/slim")

target_dir = '/home/rm/tmp/TF-Slim-PretrainedVGG16/checkpoints/'

url = ("http://54.68.5.226/car.jpg")

IMAGE_SIZE = vgg.vgg_16.default_image_size

IMAGE_FROM_COMPUTER = True
if(not IMAGE_FROM_COMPUTER):
    #im_as_string = urllib2.urlopen(url).read()  
    im_as_string = urllib.request.urlopen(url).read()  
    im = tf.image.decode_jpeg(im_as_string, channels=3)
    processed_im = vgg_preprocessing.preprocess_image(im,
                                                     IMAGE_SIZE,
                                                     IMAGE_SIZE,
                                                     is_training=False)
else:
    #Hope, Tom; Resheff, Yehezkel S.; Lieder, Itay. 
    #Learning TensorFlow: A Guide to Building Deep Learning Systems 
    #(Kindle Locations 4920-4926). O'Reilly Media. Kindle Edition. 
#    list_of_files = tf.train.match_filenames_once("/home/rm/Downloads/Images/Car.jpg") 
#    file_queue = tf.train.string_input_producer(list_of_files) 
    #https://www.tensorflow.org/api_docs/python/tf/WholeFileReader#read
#    image_reader = tf.WholeFileReader() 
#    key_value_pair = image_reader.read(file_queue) 
#    filename_at_head_of_Q = key_value_pair[0]
#    content_of_file_at_head_of_Q = key_value_pair[1]
#    im_ = tf.image.decode_jpeg(content_of_file_at_head_of_Q, channels=0)
    from PIL import Image
#    im_ = Image.open('/home/rm/tmp/Images/Car.jpg')
#    im_ = Image.open('/home/rm/tmp/Images/Car-BMW-Orig.jpeg')
#    im_ = Image.open('/home/rm/tmp/Images/cat0.jpg')
    im_ = Image.open(\
        '/home/rm/Sandlot-TensorFlow/tensorflow_input_image_by_tfrecord/src/steak/100135.jpg')
    
    processed_im = vgg_preprocessing.preprocess_image(im_,
                                                     IMAGE_SIZE,
                                                     IMAGE_SIZE,
                                                     is_training=False)
    

processed_images  = tf.expand_dims(processed_im, 0)

with slim.arg_scope(vgg.vgg_arg_scope()):
     logits, _ = vgg.vgg_16(processed_images,
                            num_classes=1000,
                             is_training=False)
probabilities = tf.nn.softmax(logits)

def vgg_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                     activation_fn=tf.nn.relu,
                     weights_regularizer=slim.l2_regularizer(weight_decay),
                     biases_initializer=tf.zeros_initializer):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc



load_vars = slim.assign_from_checkpoint_fn(
     os.path.join(target_dir, 'vgg_16.ckpt'),
     slim.get_model_variables('vgg_16'))


from datasets import imagenet
#imagenet.create_readable_names_for_imagenet_labels()


# ### Infer class and probability

# In[ ]:


names = []
import numpy as np
with tf.Session() as sess:
    load_vars(sess)     
    network_input, probabilities = sess.run([processed_images,
                                             probabilities])
    probabilities = probabilities[0, 0:]
    names_ = imagenet.create_readable_names_for_imagenet_labels()
    idxs = np.argsort(-probabilities)[:5]
    probs = probabilities[idxs]
#    classes = np.array(names_.values())[idxs+1]
    x_list = list(names_.values())
    x_array = np.asarray(x_list)
    classes = x_array[idxs + 1]
    for c,p in zip(classes,probs):
        print('Class: '+ c + ' |Prob: ' + str(p))

