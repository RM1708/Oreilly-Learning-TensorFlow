#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 18:55:34 2018

@author: rm

Module computes Error metrics by scanning the predicted & expected
sentences word-by-word.
"""

import numpy as np
from Data_Even_Odd_NOTA_Sentences import Label_NOTA

def metrics_using_python(labels_pred_, \
                         x_test, \
                         y_test, \
                         seqlen_test):
    labels_pred = np.argmax(labels_pred_, 1)
    labels_expected =  np.argmax(y_test, 1)
    list_of_error_tuples = []
    list_of_misclassified_sentences = []
    for i in range(len(labels_pred)):
        if(not(labels_expected[i] == labels_pred[i])):
            list_of_error_tuples.append((i, \
                                         seqlen_test[i], \
                                         labels_expected[i], \
                                         labels_pred[i]))
            list_of_misclassified_sentences.append(x_test[i])
#                print("list_of_error_tuples:\n", list_of_error_tuples)
    for tuple_index in range(len(list_of_error_tuples)):
        print(\
      "Error Tuple: Index: {}, Seq_len: {}, True_label: {}, Label: {}".format(\
              list_of_error_tuples[tuple_index][0], \
              list_of_error_tuples[tuple_index][1],
              list_of_error_tuples[tuple_index][2],
              list_of_error_tuples[tuple_index][3]
                          ))
    print("list_of_misclassified_sentences:\n", list_of_misclassified_sentences)

    num_of_NOTAs = 0
    num_of_NOTAs_found = 0
    for i in range(len(labels_expected)):
        if(Label_NOTA == labels_expected[i]):
            num_of_NOTAs +=1
        if(Label_NOTA == labels_pred[i]):
            num_of_NOTAs_found +=1
    print("No Of NOTAs Present: {}; Found: {}".format(num_of_NOTAs, \
                                              num_of_NOTAs_found))


