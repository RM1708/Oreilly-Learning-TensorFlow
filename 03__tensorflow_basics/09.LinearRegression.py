# coding: utf-8

import tensorflow as tf 
import numpy as np
# ### Example 1: Linear Regression

# In[23]:


# === Create data and simulate results =====
x_data = np.random.randn(2000,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2


noise = np.random.randn(1,2000)*0.1
y_data = np.matmul(w_real,x_data.T) + b_real + noise

#print("np.matmul result: ", np.matmul(w_real,x_data.T) )
#print("np.matmul result + bias: ", np.matmul(w_real,x_data.T) + b_real )

# In[24]:


NUM_STEPS = 10

g = tf.Graph()
wb_ = []
with g.as_default():
    x = tf.placeholder(tf.float32,shape=[None,3])
    y_true = tf.placeholder(tf.float32,shape=None)
    
    with tf.name_scope('Predictor') as scope:
        w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
        b = tf.Variable(0,dtype=tf.float32,name='bias')
        y_pred = tf.matmul(w,tf.transpose(x)) + b

    with tf.name_scope('Loss') as scope:
        loss = tf.reduce_mean(tf.square(y_true-y_pred))
  
    with tf.name_scope('Train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    # Before starting, initialize the variables.  
    #We will 'run' this first.
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)      
        for step in range(NUM_STEPS):
            sess.run(train,{x: x_data, y_true: y_data})
            if (step % 5 == 0):
                print("Step No: {}; Estimated Wts: {}; Estimated bias: {}".format(step, \
                      sess.run(w), sess.run(b))) 
                wb_.append(sess.run([w,b]))
                
        print("Step No: {}; Estimated Wts: {}; Estimated bias: {}".\
              format(10, \
                      sess.run([w,b])[0], sess.run([w,b])[1])) 
        print("\n loss: ", \
              sess.run(loss, \
                       {x: x_data, y_true: y_data}))
        print("Shape of Estimated y: ", \
              sess.run(y_pred, \
                       {x: x_data, y_true: y_data}).shape)




