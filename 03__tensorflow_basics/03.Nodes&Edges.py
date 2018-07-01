# coding: utf-8

import tensorflow as tf 
# ### Nodes are operations, edges are Tensor objects  

# In[16]:


c = tf.constant(4.0)
print(c)


# ### Data types  

# In[7]:


c = tf.constant(4.0, dtype=tf.float64)
print("node c: " ,c)
print("node c dtype: ",c.dtype)


# In[20]:


x = tf.constant([1,2,3],name='x',dtype=tf.float32) 
print("node x: ",x)
print("str(x): ",str(x))
print("x.dtype: ",x.dtype)
print("x.shape: ",x.shape)
print("type(x): ",type(x))
x = tf.cast(x,tf.int64)
print("x.dtype aftercast: ", x.dtype)
print("type(x) aftercast: ", type(x))

sess = tf.InteractiveSession()

print("After executing node x: ",sess.run(x))
sess.close()


