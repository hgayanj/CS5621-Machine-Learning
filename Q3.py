#!/usr/bin/env python
# coding: utf-8

# # Importing required packages

# In[9]:


import os
import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


# # Downloading the Dataset

# In[2]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# # Reshaping the Data

# In[3]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[ ]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Converting to float values
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the data
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)


# # Creating and compiling the new model

# In[ ]:


def create_model_new(activation = 'relu',
              dropout_rate = 0.2,
              optimizer = 'Adam'):
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3),
              activation=activation,
              input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer=optimizer, 
        metrics=['accuracy']
    )
    
    return model


# # Adding the Noise

# ## Noise factor =0.25

# In[5]:


noise_factor=0.25
x_train_noisy=x_train+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_train.shape)
x_test_noisy=x_test+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_test.shape)
x_train_noisy=np.clip(x_train_noisy,0.,1.)
x_test_noisy=np.clip(x_test_noisy,0.,1.)


# In[7]:


noisy_model_opt=create_model_new()
noisy_model_opt.fit(x=x_train_noisy,y=y_train, epochs=10)


# In[8]:


noisy_model_opt.evaluate(x_test_noisy, y_test)


# ## Noise Factor=0.3

# In[15]:


noise_factor=0.3
x_train_noisy_2=x_train+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_train.shape)
x_test_noisy_2=x_test+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_test.shape)
x_train_noisy_2=np.clip(x_train_noisy_2,0.,1.)
x_test_noisy_2=np.clip(x_test_noisy_2,0.,1.)


# In[16]:


noisy_model_opt_2=create_model_new()
noisy_model_opt_2.fit(x=x_train_noisy_2,y=y_train, epochs=10)


# In[17]:


noisy_model_opt_2.evaluate(x_test_noisy_2, y_test)


# ## Noise Factor=0.5

# In[18]:


noise_factor=0.5
x_train_noisy_3=x_train+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_train.shape)
x_test_noisy_3=x_test+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_test.shape)
x_train_noisy_3=np.clip(x_train_noisy_3,0.,1.)
x_test_noisy_3=np.clip(x_test_noisy_3,0.,1.)


# In[19]:


noisy_model_opt_3=create_model_new()
noisy_model_opt_3.fit(x=x_train_noisy_3,y=y_train, epochs=10)


# In[20]:


noisy_model_opt_3.evaluate(x_test_noisy_3, y_test)

