# coding: utf-8
# ### Constructing and managing our graph 

import tensorflow as tf

a = tf.constant(5) 
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a,b) 
e = tf.add(c,b) 
f = tf.subtract(d,e) 

print("\nDefault Graph: ", tf.get_default_graph())

g = tf.Graph()
print("Graph g: ", g)

# In[14]:


#a = tf.constant(7) # NOTE: This overrides the previous node
a1 = tf.Print(a, [a], "\nNode a is: ")
aa = tf.multiply(a1, a)

print("\nGraph with Node aa  is Graph g: ", aa.graph is g)
print("Graph with Node aa is Default Graph: ",\
      aa.graph is tf.get_default_graph())
print("\nNode aa: ", aa)

import numpy as  np
print("Node aa as str: ", str(np.array(aa)))


# In[4]:


g1 = tf.get_default_graph() 
g2 = tf.Graph() 

print("\nGraph g1 is the Default Graph: ", \
      g1 is tf.get_default_graph())

with g2.as_default(): 
    print("Graph g1 is the Default Graph: ", \
          g1 is tf.get_default_graph())

print("Graph g1 is the Default Graph: ", \
      g1 is tf.get_default_graph())


# ### Fetches 

# In[15]:


with tf.Session() as sess:
   fetches = [a,b,c,d,e,f]
   outs = sess.run(fetches)
   print("Result of running node aa: ", sess.run(aa))
   
print("\nList of results of fetches = {}".format(outs))
print("Type of result of node a", type(outs[0]))

with tf.Session() as sess:
   print("Result of running node aa: ", sess.run(aa))
   

