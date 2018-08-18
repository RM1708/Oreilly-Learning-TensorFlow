
# coding: utf-8

# In[ ]:


from tensorflow.contrib import slim
import sys
from datasets import dataset_utils
import tensorflow as tf
#import urllib2
import urllib
from nets import vgg
from preprocessing import vgg_preprocessing
import os

sys.path.append("/home/rm/tensorflow_models/slim")

target_dir = '/home/rm/tmp/TF-Slim-PretrainedVGG16/checkpoints/'

url = ("http://54.68.5.226/car.jpg")

#im_as_string = urllib2.urlopen(url).read()  
im_as_string = urllib.request.urlopen(url).read()  
im = tf.image.decode_jpeg(im_as_string, channels=3)

image_size = vgg.vgg_16.default_image_size

processed_im = vgg_preprocessing.preprocess_image(im,
                                                         image_size,
                                                         image_size,
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
imagenet.create_readable_names_for_imagenet_labels()


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

