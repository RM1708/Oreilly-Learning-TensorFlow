# coding: utf-8

import tensorflow as tf 
# ### Variables
# 

# In[20]:


init_val = tf.random_normal((1,5),0,1, seed=0)
var = tf.Variable(init_val, name='var1') 
print("\npre run: \n{}".format(var))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    post_var = sess.run(var)

print("post run: \n{}".format(post_var), "\n")


# ### New variables are created each time

# In[21]:


#init_val = tf.random_normal((1,5),0,1, seed=0) #Not needed
var = tf.Variable(init_val, name='var2') 
print("pre run: \n{}".format(var))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    post_var = sess.run(var) #The nearest node with output var is run
                        #Uncomment the nearest var and see.
                        #Then we get two distinct variables.
                        #With it commented, variable remains the same
print("post run: \n{}".format(post_var))



