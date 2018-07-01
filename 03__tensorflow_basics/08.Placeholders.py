# coding: utf-8

import tensorflow as tf 
import numpy as np

# ### Placeholders

# In[22]:


x_data = np.random.randn(4,3) 
w_data = np.random.randn(3,1) 

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32,shape=(4,3))
    w = tf.placeholder(tf.float32,shape=(3,1))
    b = tf.fill((4,1),-1.) 
    xw = tf.matmul(x,w)

    xwb = xw + b
    s = tf.reduce_max(xwb)
    with tf.Session() as sess:
        print("x_data: \n{}".format(x_data))
        print("w_data: \n{}".format(w_data))
        print("xw: \n", sess.run(xw, \
                               feed_dict={x: x_data,w: w_data}))
        print("xwb: \n", sess.run(xwb, \
                                feed_dict={x: x_data,w: w_data}))
        outs = sess.run(s,feed_dict={x: x_data,w: w_data}) 

print("outs = {}".format(outs))



