#!/usr/bin/env python
# coding: utf-8

# # Importing Required Packages

# In[8]:


import os
import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


# # Downloading the dataset

# In[2]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# # Reshaping the Data

# In[3]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[4]:


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


# # Creating and Compling the Model

# In[5]:


def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    
    return model


# In[6]:


model=create_model()
model.fit(x=x_train,y=y_train, epochs=10) #Fitting the model


# # Evaluating the Model

# In[7]:


model.evaluate(x_test, y_test)

