
# coding: utf-8

# ### Our first TensorFlow graph

import tensorflow as tf 

a = tf.constant(5) 
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a,b) 
e = tf.add(c,b) 
f = tf.subtract(d,e) 

sess = tf.Session() 
outs_f = sess.run(f) 
outs_e = sess.run(e) 
outs_d = sess.run(d) 
sess.close() 

print("5 x 2 = {}".format(outs_d))
print("3 + 2 = {}".format(outs_e))
print("(5 x 2) - (3 + 2) = {}".format(outs_f))


