# coding: utf-8

import tensorflow as tf 
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

def display_partition(x_values,y_values,assignment_values):
    labels = []
    colors = ["red","blue","green","yellow"]
    for i in range(len(assignment_values)):
      labels.append(colors[(assignment_values[i])])
#    color = labels
    df = pd.DataFrame\
            (dict(x =x_values,y = y_values ,color = labels ))
    fig, ax = plt.subplots()
    ax.scatter(df['x'], df['y'], c=df['color'])
    plt.show()
# ### Example 2: Logistic Regression
# 

# In[25]:


N = 200
SIGMOID_SLOPE_FACTOR = 10.0
def sigmoid(x):
    return 1 / (1 + np.exp(-x * SIGMOID_SLOPE_FACTOR))

# === Create data and simulate results =====
#x_data = np.random.randn(N,3) #THIS IS THE INPUT DATA
#                            #It is a 3-D vector
x_data = np.random.randn(N,2) #THIS IS THE INPUT DATA
                            #It is a 2-D vector
#w_real = [0.3,0.5,0.1]
w_real = [0.3,0.5]
b_real = -0.2
wxb = np.matmul(w_real,x_data.T) + b_real

y_data_pre_noise = sigmoid(wxb)
plt.plot(wxb, y_data_pre_noise, 'o', label='Y Data Pre-Noise')
plt.ylabel('Sigmoid')
plt.xlabel('Weight * X_Transpose + b')
plt.legend()
plt.show()

#See file:///home/rm/Desktop/Docs/numpy-html-1.14.0/reference/generated/numpy.random.binomial.html#numpy.random.binomial
#
#numpy.random.binomial(n, p, size=None)
#size : int or tuple of ints, optional
#    Output shape. If the given shape is, e.g., (m, n, k), then m * n * k 
#    samples are drawn. If size is None (default), 
#     a single value is returned if n and p are both scalars. 
#    Otherwise, np.broadcast(n, p).size samples are drawn.

#Samples are drawn from a binomial distribution with 
#specified parameters, n trials and p probability of success 
#where n an integer >= 0 and p is in the interval [0,1]
#Example
#>>> n, p = 10, .5  # number of trials, probability of each trial
#>>> s = np.random.binomial(n, p, 1000)
## result of flipping a coin 10 times, tested 1000 times.
#
y_data = np.random.binomial(1,y_data_pre_noise) #THIS IS THE TARGET DATA
plt.plot(wxb, y_data, 'o', label='Y Data Post-Noise')
plt.ylabel('Y')
plt.xlabel('Weight * X_Transpose + b')
plt.legend()
plt.show()

#print ("x_data[:10]: \n", x_data[:10])
#print ("(Weight * X_Transpose + b)[ : 10]: \n", wxb[ : 10])
#print ("Sigmoid O/P[0: 10]: \n", y_data_pre_noise[0: 10])
#print ("After 'Heads-Tail' [0: 10]: \n", y_data[0: 10])
#
#print("\nx-data.T shape: ", (x_data.T).shape)
#print("\nx-data.T[0] shape: ", (x_data.T[0]).shape)
#print("y_data shape: ", y_data.shape, "\n")

print("X-axis: x_data.T[0], Y-axis: x_data.T[1]")
display_partition(x_data.T[0], x_data.T[1], y_data)

#Instead of a cyclic ordering, we choose to keep the axis, common to
#both plots, as the Y-axis
#print("X-axis: x_data.T[2], Y-axis: x_data.T[1]")
#display_partition(x_data.T[2], x_data.T[1], y_data)

#This third projection is reduntant. 
#Two straight lines define a plane. A plane A's intersection with 
#A plane' A's intersection with another plane B is a straight. So is 
#A's intersection with another plane C. Even if B & C are parallel planes,
#these two lines of intersection define the plane A. 
#
#print("X-axis: x_data.T[2], Y-axis: x_data.T[1]")
#display_partition(x_data.T[2], x_data.T[0], y_data)



# In[26]:


NUM_STEPS = 20000


g = tf.Graph()
wb_ = []
with g.as_default():
#    x = tf.placeholder(tf.float32,shape=[None,3])
    x = tf.placeholder(tf.float32,shape=[None,2])
    y_true = tf.placeholder(tf.float32,shape=None)#THIS IS "HEADS" OR "TAILS"
    
    with tf.name_scope('Prediction') as scope:
#        w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
        w = tf.Variable([[0,0]],dtype=tf.float32,name='weights')
        b = tf.Variable(0,dtype=tf.float32,name='bias')
        y_pred = (tf.matmul(w,tf.transpose(x)) + b)

    with tf.name_scope('Loss') as scope:
        #https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        #Measures the probability error in discrete classification tasks in 
        #which each class is independent and ***not mutually exclusive***
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred) 
        loss = tf.reduce_mean(tf.divide(loss, SIGMOID_SLOPE_FACTOR))
  
    with tf.name_scope('train') as scope:
        learning_rate = 0.7
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)



    # Before starting, initialize the variables.  We will 'run' this first.
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)      
        for step in range(NUM_STEPS):
            sess.run(train,{x: x_data, y_true: y_data})
            if (step % (NUM_STEPS//10) == 0):
#                print(step, sess.run([w,b]))
                print("Step No: {}; Estimated Wts: {}; Estimated bias: {}; Loss: {}".format(step, \
                      sess.run(w)/SIGMOID_SLOPE_FACTOR, \
                      sess.run(b)/SIGMOID_SLOPE_FACTOR, 
                      sess.run(loss, \
                               feed_dict={x: x_data, y_true: y_data}))) 
                wb_.append(sess.run([w,b]))

        print("Step No: {}; Estimated Wts: {}; Estimated bias: {}".\
              format(NUM_STEPS, \
                      sess.run([w,b])[0]/SIGMOID_SLOPE_FACTOR, \
                      sess.run([w,b])[1]/SIGMOID_SLOPE_FACTOR)) 
        print("\n name of the variable loss: ",loss.name)
        print(" name of the variable w: ",w.name)
        print(" name of the variable b: ",b.name)
        print("\n loss: ", \
              sess.run(loss, \
                       {x: x_data, y_true: y_data}))
        print("Shape of Estimated y: ", \
              sess.run(y_pred, \
                       {x: x_data, y_true: y_data}).shape)
        print ("(Weight * X_Transpose + b)[ : 10]: \n", wxb[ : 10])
        print("Estimated y[0, :10] - Compare with wxb: \n", \
              ((sess.run(y_pred, \
                       {x: x_data, y_true: y_data}))[0, : 10])/SIGMOID_SLOPE_FACTOR)


