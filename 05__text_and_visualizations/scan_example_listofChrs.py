# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 23:20:29 2016

@author: tomhope
"""

import numpy as np
import tensorflow as tf


elems = np.array(["T", "e", "n", "s", "o", "r",  " ",  "F", "l", "o", "w"])
scan_sum = tf.scan(lambda a, x: a + x, elems)

with tf.Session() as sess:
    print("elems:\n",elems)
    scanned = (sess.run(scan_sum))
    print("scanned:\n", scanned)
    print(scanned[0].decode())
    print(scanned[5].decode())
    print(scanned[10].decode())
    for elem in scanned: print(elem.decode())
    [print(elem.decode(), end=", ") for elem in scanned]
    print()
    for i in range(len(scanned) - 1):
        print(scanned[i].decode(), end=", ")        
    print(scanned[len(scanned) - 1].decode())

elems = np.array([["T", "e", "n", "s", "o", "r",  " ",  "F", "l", "o", "w"], \
                 ["1", "2", "3", "4", "5", "6",  "7",  "8", "9", "10", "11"], \
                 ["11", "12", "13", "14", "15", "16",  "17",  "18", "19", "20", "21"]])
scan_sum = tf.scan(lambda a, x: a + x, elems)

with tf.Session() as sess:
    print("elems: \n", elems)
    scanned = (sess.run(scan_sum))
    print("scanned: \n", scanned)
    print("scanned[0,0]: \n", scanned[0,0].decode())
    print("scanned[0, 5]: \n", scanned[0,5].decode())
    print("scanned[1, 5]: \n", scanned[1,5].decode())
    print("scanned[2, 5]: \n", scanned[2,5].decode())
