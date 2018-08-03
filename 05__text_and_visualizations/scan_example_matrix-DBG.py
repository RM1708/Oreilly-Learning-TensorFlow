# -*- coding: utf-8 -*-
"""
Created on 12Jul2018

@author: RM
"""

import numpy as np
import tensorflow as tf

elems_3x3x3x3 = np.asarray([
                        [
                            [
                                [1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]
                            ],
                            [
                                [10, 20, 30],
                                [40, 50, 60],
                                [70, 80, 90]
                            ],
                            [
                                [100, 200, 300],
                                [400, 500, 600],
                                [700, 800, 900]
                            ]
                        ],
                        [
                            [
                                [10, 20, 30],
                                [40, 50, 60],
                                [70, 80, 90]
                            ],
                            [
                                [10, 20, 30],
                                [40, 50, 60],
                                [70, 80, 90]
                            ],
                            [
                                [10, 20, 30],
                                [40, 50, 60],
                                [70, 80, 90]
                            ]
                        ],
                        [
                            [
                                [100, 200, 300],
                                [400, 500, 600],
                                [700, 800, 900]
                            ],
                            [
                                [100, 200, 300],
                                [400, 500, 600],
                                [700, 800, 900]
                            ],
                            [
                                [100, 200, 300],
                                [400, 500, 600],
                                [700, 800, 900]
                            ]
                        ]
                    ]
                    )
scan_sum = tf.scan(lambda a, x: a + x, elems_3x3x3x3)

with tf.Session() as sess:
    print("tf.scan uses lambda function that adds.")
    sess.run(tf.global_variables_initializer())
    print("\nelems_3x3x3x3.shape: ", elems_3x3x3x3.shape)
    print("elems_3x3x3x3: \n", elems_3x3x3x3)
    scanned = (sess.run(scan_sum))
    print("\nscanned by scan_sum using a lambda function that adds: \n", scanned)
    print("******************************")


def scan_row_sum_W_3x1(prev_OutPut, A):
    W =np.asarray([[1.0],[1.0],[1.0]])
    W_tensor = tf.constant(W)
    return (prev_OutPut + tf.matmul(A, W_tensor))

Data_0_0_0_0to2 = \
    tf.Variable(np.array(elems_3x3x3x3[0][0][0, 0:3].astype(np.float64)))
Data_0_0_0to2_0to2 = \
    tf.Variable(np.array(elems_3x3x3x3[0][0][0:3, 0:3].astype(np.float64)))
Data_0_0to2_0to2_0to2 = \
    tf.Variable(np.array(elems_3x3x3x3[0][0:3][0:3, 0:3].astype(np.float64)))

Data_0to2_0to2_0to2_0to2 = \
    tf.Variable(np.array(elems_3x3x3x3[0:3][0:3][0:3, 0:3].astype(np.float64)))
Data_1_0to2_0to2_0to2 = \
    tf.Variable(np.array(elems_3x3x3x3[1][0:3][0:3, 0:3].astype(np.float64)))
Data_2_0to2_0to2_0to2 = \
    tf.Variable(np.array(elems_3x3x3x3[2][0:3][0:3, 0:3].astype(np.float64)))
    
with tf.Session() as sess:
#    print("elems: \n", elems)
    sess.run(tf.global_variables_initializer())
    print("NOTE: tf.scan uses scan_row_sum_W_3x1() that does a " +
          "Matrix-Multiplication and Add")
    
    print("\nelems_3x3x3x3.shape: ", np.array(elems_3x3x3x3).shape)
    
    print("\n\"sess.run(Data_0_0_0_0to2).shape\"): ", \
          sess.run(Data_0_0_0_0to2).shape)
    print("\"sess.run(Data_0_0_0_0to2))\": ", \
          sess.run(Data_0_0_0_0to2))
    
    print("\n\"sess.run(Data_0_0_0to2_0to2).shape)\": ", \
          sess.run(Data_0_0_0to2_0to2).shape)
    print("\"sess.run(Data_0_0_0to2_0to2))\": \n", \
          sess.run(Data_0_0_0to2_0to2))
    
    print("\n\"sess.run(Data_0_0to2_0to2_0to2).shape\"): ", \
          sess.run(Data_0_0to2_0to2_0to2).shape)
    print("\"sess.run(Data_0_0to2_0to2_0to2))\": \n", \
          sess.run(Data_0_0to2_0to2_0to2))
    print("\"sess.run(Data_0_0to2_0to2_0to2)[0]\"): \n", \
          sess.run(Data_0_0to2_0to2_0to2)[0])
    print("\"sess.run(Data_0_0to2_0to2_0to2)[1]\"): \n", \
          sess.run(Data_0_0to2_0to2_0to2)[1])
    print("\"sess.run(Data_0_0to2_0to2_0to2)[2]\"): \n", \
          sess.run(Data_0_0to2_0to2_0to2)[2])
    
    print("\"sess.run(Data_0to2_0to2_0to2_0to2)\": \n", \
          sess.run(Data_0to2_0to2_0to2_0to2))
    print("\"sess.run(Data_1_0to2_0to2_0to2)\": \n", \
          sess.run(Data_1_0to2_0to2_0to2))
    
    print("******************************")

    
##########################################################
#Scan multiple inputs
def scan_row_sum_W_3x1_Multi(prev_Out, In):
    W =np.asarray([[1.0],[1.0],[1.0]])
    W_tensor = tf.constant(W)
    Out_0 = prev_Out[0] + tf.matmul(In[0], W_tensor)
    Out_1 = prev_Out[1] + tf.matmul(In[1], W_tensor)
    Out_2 = prev_Out[2] + tf.matmul(In[2], W_tensor)
    return (Out_0, Out_1, Out_2)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

scanned0 = (sess.run(tf.scan(scan_row_sum_W_3x1, \
                    Data_0_0to2_0to2_0to2)))
print("\nscanned Data_0_0to2_0to2_0to2: \n", scanned0)

scanned = (sess.run(tf.scan(scan_row_sum_W_3x1, \
                    (Data_1_0to2_0to2_0to2))))
print("\n\"scanned Data_0_0_0to2_0to2\": \n", scanned)

scanned0, scanned1, scanned2 = (sess.run(tf.scan(scan_row_sum_W_3x1_Multi, \
                    (Data_0_0to2_0to2_0to2,
                    Data_1_0to2_0to2_0to2,
                    Data_2_0to2_0to2_0to2))))
print("******************************")
print("\n\"scanned0 Data_0_0_0to2_0to2\": \n", \
                  scanned0)
print("\n\"scanned1 Data_1_0to2_0to2_0to2\": \n", \
                  scanned1)
print("\n\"scanned2 Data_2_0to2_0to2_0to2\": \n", \
                  scanned2)

sess.close()

print("******************************")

