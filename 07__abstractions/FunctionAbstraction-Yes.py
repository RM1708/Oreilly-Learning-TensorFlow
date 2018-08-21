'''
See chapter 4 
    1. layers.py. & 
    2. mnist_cnn.py
'''
def weight_variable( shape):
       initial = tf.truncated_normal( shape, stddev = 0.1)
       return tf.Variable( initial)

def bias_variable( shape):
     initial = tf.constant( 0.1, shape = shape)     
     return tf.Variable( initial)   

def conv2d( x, W):
     return tf.nn.conv2d( x, W, strides =[ 1, 1, 1, 1],
                                padding ='SAME') 

def conv_layer( input, shape):
       W = weight_variable( shape)       
       b = bias_variable([ shape[ 3]])       
       h = tf.nn.relu( conv2d( input, W) + b)       
       hp = max_pool_2x2( h)       
       return hp 
       
def max_pool_2x2( x):
       return tf.nn.max_pool( x, ksize =[ 1, 2, 2, 1],                             
                                strides =[ 1, 2, 2, 1], 
                                padding ='SAME') 
                                
x = tf.placeholder( tf.float32, shape =[ None, 784]) 
x_image = tf.reshape( x, [-1, 28, 28, 1]) 
h1 = conv_layer( x_image, shape =[ 5, 5, 1, 32])   
h2 = conv_layer( h1, shape =[ 5, 5, 32, 64]) 
h3 = conv_layer( h2, shape =[ 5, 5, 64, 32])


