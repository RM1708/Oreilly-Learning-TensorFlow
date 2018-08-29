
# coding: utf-8

# ### a toy CNN autoencoder with Keras.
# 1. Learning TensorFlow: A Guide to Building Deep Learning Systems (Kindle Locations 4623-4624). O'Reilly Media. Kindle Edition. 

# In[1]:


#import keras # keras now available from tensorflow
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.datasets import cifar10
import numpy as np

CLASS_AUTOMOBILE = 1
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[np.where(y_train==CLASS_AUTOMOBILE)[0],:,:,:]
x_test = x_test[np.where(y_test==CLASS_AUTOMOBILE)[0],:,:,:]

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

NOISE_MEAN = 0.0; NOISE_STD_DEV = 0.4

x_train_plusNoise = x_train + 0.5 * np.random.normal(loc=0.0, scale=0.4, size=x_train.shape) 

x_test_plusNoise = x_test + 0.5 * np.random.normal(loc=0.0, scale=0.4, size=x_test.shape) 

# Keep the values in the range [0, 1.0]. 
# Why?
x_train_plusNoise = np.clip(x_train_plusNoise, 0., 1.)
x_test_plusNoise = np.clip(x_test_plusNoise, 0., 1.)

#https://www.tensorflow.org/api_docs/python/tf/keras/layers/InputLayer
#It is generally recommend to use the functional layer API via Input, 
#(which creates an InputLayer) without directly using InputLayer
inp_img = Input(shape=(32, 32, 3))   
inp_img.shape


# In[5]:


img= Conv2D(32, #filters: Integer, the dimensionality of the output space
            (3, 3), # height and width of the 2D convolution window
            activation='relu', 
            padding='same')(inp_img)
'''
From: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D#output
output
Retrieves the output tensor(s) of a layer.

Only applicable if the layer has exactly one output, i.e. if it is connected to one incoming layer.

Returns:
Output tensor or list of output tensors.
'''
assert img.input == inp_im


# In[ ]:


img = MaxPooling2D((2, 2), padding='same')(img)
img.shape


# In[ ]:


img = Conv2D(32, (3, 3), activation='relu', padding='same')(img)
img.shape


# In[ ]:


img = UpSampling2D((2, 2))(img)
img.shape


# In[ ]:


decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(img)
decoded.shape


# In[ ]:


autoencoder = Model(inp_img, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

tensorboard = TensorBoard(log_dir='./models/autoencoder',              histogram_freq=0, write_graph=True, write_images=True)

#Learning TensorFlow: A Guide to Building Deep Learning Systems (Kindle Location 4685). 
#O'Reilly Media. Kindle Edition. 
model_saver = ModelCheckpoint(
                    filepath='./models/autoencoder/autoencoder_model',\
                     verbose=0, period=2)

autoencoder.fit(x_train_plusNoise, x_train,
                epochs=10,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test_plusNoise, x_test),
                callbacks=[tensorboard, model_saver])


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

n_imgs = 10
f,axarr = plt.subplots(2,n_imgs,figsize=[20,5])
decoded_imgs = autoencoder.predict(x_test_n)

for i in range(n_imgs):
    
    ax = axarr[0,i]
    ax.get_yaxis().set_visible(False)
    ax.imshow(x_test_n[i,:,:,:])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = axarr[1,i]
    ax.get_yaxis().set_visible(False)
    ax.imshow(decoded_imgs[i,:,:,:])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
            
plt.tight_layout()
plt.show()

