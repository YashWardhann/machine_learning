#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import time 
import h5py 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import expit, logit


# In[13]:


def load_dataset(): 
    train_set = h5py.File('train_catvnoncat.h5', 'r')
    train_set_x = np.array(train_set['train_set_x'])
    train_set_y = np.array(train_set['train_set_y'])
    train_classes = np.array(train_set['list_classes'])
    return train_set_x, train_set_y, train_classes

X, y, classes = load_dataset()

# Get the features of the training data
m_train = y.shape[0]
n = X.shape[1] * X.shape[2] * X.shape[3]

# Reshape y matrix
y = y.reshape(1, y.shape[0])

# Flatten the X matrix 
X_flatten = X.reshape(m_train, n).T

# Standardize the values of X
X_flatten = X_flatten/255


# In[20]:


# Set the hyperparameters of the model 
learning_rate = 0.075 
num_iterations = 2000

layer_dims = [n, 20, 7, 5, 1]

# Set the parameters of the network 
def initialize_parameters(layer_dims): 
    parameters = {} 
    for i in range(1, len(layer_dims)): 
        parameters['W'+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])
        parameters['b'+str(i)] = np.zeros((layer_dims[i], 1))
    return parameters

parameters = initialize_parameters(layer_dims)   


# In[24]:


# Helper functions 
def relu(z): 
    return z * (z > 0)

def feed_forward(X, parameters): 
    L = len(parameters)//2 # Number of hidden layers
    cache = [] 
    A = X
    for i in range(1, L): 
        A_prev = A
        Z = np.dot(parameters['W'+str(i)], A) + parameters['b'+str(i)]
        A = relu(Z)
        cache.append((A, Z))
    # Compute the output 
    Z_output = np.dot(parameters['W'+str(L)], A) + parameters['b'+str(L)]
    A_output = expit(Z_output) # Apply sigmoid function. expit function prevents exp overflow
    cache.append((A_output, Z_output))
    return A_output, cache

output, cache = feed_forward(X_flatten, parameters)        


# In[28]:


def computeCost(X, y, parameters): 
    m = y.shape[1]
    epsilon = 1e-6 # Prevent zero error in log
    predictions, cache = feed_forward(X, parameters)
    cost = -1/m * np.sum(y * np.log(predictions + epsilon) + (1-y) * np.log(1-predictions+epsilon))
    
    return cost
initialCost = computeCost(X_flatten, y, parameters)
print('Initial Cost is:', initialCost)


# In[ ]:


def back_propagation(X, y, parameters): 
    output, cache = feed_forward(X, parameters)
    grads = {} # Stores the gradient for each layer 
    for i in reversed(range(L-1))

