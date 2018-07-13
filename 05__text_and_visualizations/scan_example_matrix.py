# -*- coding: utf-8 -*-
"""
Created on 12Jul2018

@author: RM
"""

import numpy as np
import tensorflow as tf


array_3x3 = np.asarray(
                    [[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
scan_sum = tf.scan(lambda a, x: a + x, array_3x3)

with tf.Session() as sess:
    print("tf.scan uses lambda function that adds.")
    print("array_3x3: \n", array_3x3)
    print("array_3x3.shape: ", \
          array_3x3.shape)
    print("\nnp.asarray([array_3x3]).shape: ", \
          np.asarray([array_3x3]).shape)
    print("np.asarray([array_3x3]):\n ", \
          np.asarray([array_3x3]))
    print("np.asarray([array_3x3[0]]).shape: ", \
          np.asarray([array_3x3[0]]).shape)
    print("np.asarray([array_3x3[0]]): ", \
          np.asarray([array_3x3[0]]))
    scanned = (sess.run(scan_sum))
    print("\nscanned: \n", scanned)
    print("******************************")

elems_3x3x3x3 = np.asarray([
                        [
                            [
                                [1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]
                            ],
                            [
                                [1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]
                            ],
                            [
                                [1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]
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
    print("\nscanned: \n", scanned)
    print("******************************")


def scan_row_sum_W_3x1(prev_OutPut, A):
    W =np.asarray([[1.0],[1.0],[1.0]])
    W_tensor = tf.constant(W)
    return (prev_OutPut + tf.matmul(A, W_tensor))

def scan_row_sum_W_3x2(prev_OutPut, A):
    W =np.asarray([[1.0],[1.0],[1.0]])
    W_tensor = tf.constant(W)
    return (prev_OutPut + tf.matmul(A, W_tensor))

Data_0_0_0_0to2 = \
    tf.Variable(np.array(elems_3x3x3x3[0][0][0, 0:3].astype(np.float64)))
Data_0_0_0to2_0to2 = \
    tf.Variable(np.array(elems_3x3x3x3[0][0][0:3, 0:3].astype(np.float64)))
Data_0_0to2_0to2_0to2 = \
    tf.Variable(np.array(elems_3x3x3x3[0][0:3][0:3, 0:3].astype(np.float64)))
with tf.Session() as sess:
#    print("elems: \n", elems)
    sess.run(tf.global_variables_initializer())
    print("tf.scan uses scan_row_sum_W_3x1() that does a " +
          "Matrix-Multiplication and Add")
    
    print("\nelems_3x3x3x3.shape: ", np.array(elems_3x3x3x3).shape)
    
    print("\nsess.run(Data_0_0_0_0to2).shape): ", \
          sess.run(Data_0_0_0_0to2).shape)
    print("sess.run(Data_0_0_0_0to2)): ", \
          sess.run(Data_0_0_0_0to2))
    
    print("\nsess.run(Data_0_0_0to2_0to2).shape): ", \
          sess.run(Data_0_0_0to2_0to2).shape)
    print("sess.run(Data_0_0_0to2_0to2)): \n", \
          sess.run(Data_0_0_0to2_0to2))
    
    print("\nsess.run(Data_0_0to2_0to2_0to2).shape): ", \
          sess.run(Data_0_0to2_0to2_0to2).shape)
    print("sess.run(Data_0_0to2_0to2_0to2)): \n", \
          sess.run(Data_0_0to2_0to2_0to2))
    
    scanned = (sess.run(tf.scan(scan_row_sum_W_3x1, \
                        Data_0_0to2_0to2_0to2)))
    print("\nscanned Data_0_0to2_0to2_0to2: \n", scanned)

#    scanned = (sess.run(tf.scan(scan_row_sum_W_3x2, \
#                        Data_0_0to2_0to2_0to2, \
#                        initializer=np.array([0.0, 0.0]))))
#    print("\nscanned Data_0_0to2_0to2_0to2: \n", scanned)

#    scanned = (sess.run(tf.scan(scan_row_sum_W_3x1, \
#                        np.asarray([Data_0_0_0to2_0to2]))))
#    print("\nscanned [Data_0_0_0to2_0to2]: \n", scanned)
    print("******************************")

