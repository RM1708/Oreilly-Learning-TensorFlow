x = tf.placeholder( tf.float32, shape =[ None, 784]) 
x_image = tf.reshape( x, [-1, 28, 28, 1]) 
W1 = tf.truncated_normal([ 5, 5, 1, 32], stddev = 0.1) 
b1 = tf.constant( 0.1, shape =[ 32]) 
h1 = tf.nn.relu( tf.nn.conv2d( x_image, W1, strides =[ 1, 1, 1, 1], padding ='SAME') + b1) 
hp1 = tf.nn.max_pool( h1, ksize =[ 1, 2, 2, 1], strides =[ 1, 2, 2, 1], padding ='SAME') 

W2 = tf.truncated_normal([ 5, 5, 32, 64], stddev = 0.1) 
b2 = tf.constant( 0.1, shape =[ 64]) 
h2 = tf.nn.relu( tf.nn.conv2d( hp1, W2, strides =[ 1, 1, 1, 1],   padding ='SAME') + b2) 
hp2 = tf.nn.max_pool( h2, ksize =[ 1, 2, 2, 1], strides =[ 1, 2, 2, 1], padding ='SAME') 

W3 = tf.truncated_normal([ 5, 5, 64, 32], stddev = 0.1) 
b3 = tf.constant( 0.1, shape =[ 32]) 
h3 = h1 = tf.nn.relu( tf.nn.conv2d( hp2, W3, strides =[ 1, 1, 1, 1], padding ='SAME') + b3) 
#Q. Why the feed-back to h1? Not there in the abstracted version.
hp3 = tf.nn.max_pool( h3, ksize =[ 1, 2, 2, 1], strides =[ 1, 2, 2, 1], padding ='SAME')


