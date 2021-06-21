#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
import time
m = 10000  # number of training data
n = 500  # number of test data
k = 1000  # number of iterations
alpha = 0.01  # learning rate

x_train = np.load("x_train.npy")# (2,m)
x_train = x_train.transpose()
y_train = np.load("y_train.npy")  # (1, m)
x_test = np.load("x_test.npy")  # (n, 2)
y_test = np.load("y_test.npy")  # (1, n)


# In[2]:


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Sequential()
    # model.add(Input(2,))
    model.add(Dense(3, activation="sigmoid"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
    training_start_time = time.time()
    model.fit(x_train, y_train, epochs=k, batch_size=128)
print("Training_time : ", time.time()-training_start_time)


# In[ ]:





# In[50]:


model.evaluate(x_test, y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




