
# coding: utf-8

import tensorflow as tf 
# ### Matrix multiplication

# In[5]:


A = tf.constant([ [1,2,3],
                  [4,5,6] ]) # A 2 x 3 matrix
print("Shape of matrix A: ", A.get_shape())

ListOfWts = [1, 0, 1]
x = tf.constant(ListOfWts) # This is a ***LIST***. 
                        #It is ***NOT*** a Col/Row matrix
print("Shape of list of wts x: ", x.get_shape())

x1 = tf.expand_dims(x,-1) #***NOW*** it is a matrix with 3 Rows and
                        # 1 Col
#x1 = tf.expand_dims(x,1) #This is equivalent
#x1 = tf.expand_dims(x,0) #This is not. 
                            #It creates a 1-Row x 3-Col matrix
print("Shape of matrix of wts: ", x1.get_shape())

b = tf.matmul(A,x1)

sess = tf.InteractiveSession()
print("\nx[0]: {} ".format(x.eval()[0]))
#print("\nx[0, 1]: {} ".format(x.eval()[0, 1])) #This will throw an error
print("\nx1:\n {} ".format(x1.eval()))
print("x1[0]: {} ".format(x1.eval()[0]))
print("x1[0, 0]: {} ".format(x1.eval()[0, 0]))
#print("x1[0, 1]: {} ".format(x1.eval()[0, 1])) #This will throw an error

print('\nmatmul result:\n {}'.format(b.eval()))
print('matmul result shape:\n {}'.format(b.eval().shape))
sess.close()


