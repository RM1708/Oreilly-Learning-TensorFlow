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
    print(elems)
    scanned = (sess.run(scan_sum))
    print(scanned)
    print(scanned[0].decode())
    print(scanned[5].decode())
    print(scanned[10].decode())
    for elem in scanned: print(elem.decode())
    [print(elem.decode(), end=", ") for elem in scanned]
    print()
    for i in range(len(scanned) - 1):
        print(scanned[i].decode(), end=", ")        
    print(scanned[len(scanned) - 1].decode())
