# coding: utf-8

import tensorflow as tf 
# ### Names  

# In[18]:


with tf.Graph().as_default():
    c1 = tf.constant(4,dtype=tf.float64,name='c') 
    c2 = tf.constant(4,dtype=tf.int32,name='c') 
print("Name of tensor c1: ", c1.name)
print("Name of tensor c2: ", c2.name)


# ### Name scopes

# In[19]:


with tf.Graph().as_default():
    c1 = tf.constant(4,dtype=tf.float64,name='c') 
    with tf.name_scope("prefix_name"):
        c2 = tf.constant(4,dtype=tf.int32,name='c') 
        c3 = tf.constant(4,dtype=tf.float64,name='c')

print("Name of tensor c1: ", c1.name)
print("Name of tensor c2 (inside namespace 'prefix_name'): ", c2.name)
print("Name of tensor c3 (inside namespace 'prefix_name'): ", c3.name)



