#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:35:30 2018
@author: rm
# SEE: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
#
# Given 
#   1. An input tensor of shape 
#       [batch, in_height, in_width, in_channels] and 
#   2. A filter / kernel tensor of shape 
#       [filter_height, filter_width, in_channels, out_channels], 
#    
# This op conv2d() performs the following:
#
    #1. Flattens the filter to a 2-D matrix with shape 
    #       [filter_height * filter_width * in_channels, output_channels].
    #
    #2. Extracts image patches from the input tensor 
    #   to form a virtual tensor of shape 
    #      [batch, out_height, out_width, 
    #                 filter_height * filter_width * in_channels].
    #
    #3. For each patch, right-multiplies the filter matrix 
    #   and the image patch vector.
#
    # QUESTION: How is the above mentioned flattening reflected 
    # in the expression below?
    
# In detail, with the default NHWC format,
    #output[b, i, j, k] =
    #    sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
    #                    filter[di, dj, q, k]
#
#    MY EXPLANATORY NOTE for the above
#    --------------------------------
#    The output for 
#        the kth out_channel
#            of the jth pixel
#                of the ith row
#                    of the bth item in the batch
#    
#    is given by the triple nested summation (starting with the innermost summation)
#        Sum over all the input channels (all values of q) for a given filter coeff (di, dj, q. k). k is fixed by the
#                                            output. di & dj are fixed by the enclosing summation. 
#                                            q is the summation variable
#            Sum over all filter coeffs for a given row (di)of the filter(the elements of row of the filter, correspond to
#                                                              the horizontal dimension of the input). 
#                                                       k is fixed by the output. di is fixed by the enclosing summation.
#                                                       dj is the summation variable
#                Sum over all rows of the filter (filter row dimension corresponds 
#                                                 to input row dimension ) 
#                                                k is fixed by the output. 
#                                                di is the summation variable.
#                    for a given batch element b
#
#    The expression within the triple summation is:
#        1. Input pixels to be processed by the filter belong to the item b (as specified by the output) of the input batch.
#        2. The specific pixels of the input that are to be filtered. These start at (i.e. the top-left corner, with ), 
#            2.1. in the height dimension, at row = desired_row_of_output(i.e. i) * stride_in-height_direction, and
#            2.2. in the width dimension, at col = desired_col_output(i.e. j) * stride_in_width_direction 
#            2.3. in the channel dimension, at q = 0
#                NOTE: 
#                    1. The i & j are decided by the height and width component of the output pixel 
#                    at which the value is to be determined.
#                    2. k is decided by the channel of the output pixel
#                    
#            2.4. The filter is a set of cuboids. Each cuboid in the set is for each element
#            in the k elements in the set of output channels. The set of cuboids is applied to a cuboid of pixels. 
#                2.4.1. The Height of each cuboid in the filter = filter.shape[0]
#                2.4.2. The Width of each cuboid in the filter = filter.shape[1]
#                2.4.3. The depth of each cuboid in the filter = filter.shape[2]
#                2.4.4. The number of cuboids in the filter = filter.shape[3]
#

#***Must*** have strides[0] = strides[3] = 1. 
#
#For the most common case of the same horizontal and vertices strides, 
#strides = [1, stride, stride, 1].

#from https://www.tensorflow.org/api_docs/python/tf/nn/convolution
#
#given 
#    a rank (N+2) input Tensor of shape
#    [num_batches, input_spatial_shape[0], ..., input_spatial_shape[N-1], 
#     num_input_channels],  
#    
#    a rank (N+2) filter Tensor of shape
#    [spatial_filter_shape[0], ..., spatial_filter_shape[N-1], 
#     num_input_channels, num_output_channels],  
#
#    an optional dilation_rate tensor of shape N specifying the filter 
#    upsampling/input downsampling rate, and an optional list of N strides 
#    (defaulting [1]*N), this computes for each N-D spatial output position 
#    (x[0], ..., x[N-1]):
#
#          output[b, x[0], ..., x[N-1], k] =
#          sum_{z[0], ..., z[N-1], q}
#              filter[z[0], ..., z[N-1], q, k] *
#              padded_input[b,
#                           x[0]*strides[0] + dilation_rate[0]*z[0],
#                           ...,
#                           x[N-1]*strides[N-1] + dilation_rate[N-1]*z[N-1],
#                           q]
#
#    where b is the index into the batch, k is the output channel number, 
#    q is the input channel number, 
#    and z is the N-D spatial offset within the filter.

"""
import tensorflow as tf
import numpy as np

NO_OF_OUT_CHANNELS = 2

filter_coeffs_as_nested_list = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], \
                                [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
filter_coeffs_as_2Darray = np.asarray(filter_coeffs_as_nested_list)
tmp = np.reshape(filter_coeffs_as_2Darray, [1, 18])
print("tmp: \n",tmp)
print("Shape of filter_coeffs_as_2Darray:\n", filter_coeffs_as_2Darray.shape)
filter_coeffs = np.reshape(tmp, \
                               [NO_OF_OUT_CHANNELS, 1] + \
                                [filter_coeffs_as_2Darray.shape[0]//2, \
                                 filter_coeffs_as_2Darray.shape[1]]
                                )
print("Before Transpose: filter_coeffs:\n", filter_coeffs)
print("Before Transpose: Shape of filter_coeffs:\n", filter_coeffs.shape)
filter_coeffs_T = np.transpose(filter_coeffs, (2, 3, 1, 0))
print("After Transpose: filter_coeffs_T:\n", filter_coeffs_T)
print("After Transpose: Shape of filter_coeffs_T:\n", filter_coeffs_T.shape)
print("filter_coeffs_T[0, :, 0, 0]: ", filter_coeffs_T[0, :, 0, 0])
print("filter_coeffs_T[1, :, 0, 0]: ", filter_coeffs_T[1, :, 0, 0])
print("filter_coeffs_T[2, :, 0, 0]: ", filter_coeffs_T[2, :, 0, 0])
print()
print("filter_coeffs_T[0, :, 0, 1]: ", filter_coeffs_T[0, :, 0, 1])
print("filter_coeffs_T[1, :, 0, 1]: ", filter_coeffs_T[1, :, 0, 1])
print("filter_coeffs_T[2, :, 0, 1]: ", filter_coeffs_T[2, :, 0, 1])
print()
input_as_1D_array = tf.placeholder(tf.int64, \
                                   [1, 100], \
                                   name="Placeholder_Pixels")

filter_kernel = tf.placeholder(tf.float32, \
                               [None, None, 1, NO_OF_OUT_CHANNELS], \
                               name="Placeholder_Filter")

input_as_2D_array = tf.reshape(tf.cast(input_as_1D_array, tf.float32), [1, 10, 10, 1])

#
filter_def = tf.Variable(tf.zeros([3, 3, 1, NO_OF_OUT_CHANNELS], tf.float32))
filter = tf.assign(filter_def, filter_kernel)
#
z = tf.identity(filter)

PADDING_TYPE = 'VALID'

filtered_out = tf.nn.conv2d(input_as_2D_array, \
                            filter, \
                            strides=[1, 1, 1, 1], \
                            padding=PADDING_TYPE)

init_vars = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_vars)
    
    list_of_pixel_values = [x for x in range(100)]
    pixel_values_1D_array = np.asarray(list_of_pixel_values).reshape(1, 100)
    z = sess.run(z,  feed_dict={input_as_1D_array: pixel_values_1D_array, \
                                     filter_kernel: filter_coeffs_T})
    print("filter array[:, :, 0, 0]: \n", z[:, :, 0, 0])
    print("filter array[:, :, 0, 1]: \n", z[:, :, 0, 1])
    
    y = sess.run(input_as_2D_array,  feed_dict={input_as_1D_array: pixel_values_1D_array, \
                                     filter_kernel: filter_coeffs_T})
    print("\ninput_as_2D_array: \n", y[0, :, :, 0])
    
    filtered_out = sess.run(filtered_out, \
                         feed_dict={input_as_1D_array: pixel_values_1D_array, \
                                     filter_kernel: filter_coeffs_T})
    
    print("\nShape of filtered_out: \n", filtered_out.shape)
    print("Padding is: {}".format(PADDING_TYPE))
    print("\nChannel 0 filtered_out as 2D array: \n{}\n".format( \
          filtered_out[0, :, :, 0]))
    print("\nChannel 1 filtered_out as 2D array: \n{};\n".format( \
          filtered_out[0, :, :, 1]))
