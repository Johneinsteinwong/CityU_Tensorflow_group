#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:21:33 2017

@author: wongtszlunjohneinstein

regression without categorial features
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


batch_size = 323
feature_size = 11

whole = pd.read_csv('boston_housing.csv')
train, test = train_test_split(whole, test_size=0.2, random_state=42)
train, valid = train_test_split(train, test_size=0.2, random_state=42)

train_labels = train[['MEDV']].as_matrix().astype(np.float32)
valid_labels = valid[['MEDV']].as_matrix().astype(np.float32)
test_labels = test[['MEDV']].as_matrix().astype(np.float32)

train_dataset = train[['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','TAX','PTRATIO','B','LSTAT']].as_matrix().astype(np.float32)
valid_dataset = valid[['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','TAX','PTRATIO','B','LSTAT']].as_matrix().astype(np.float32)
test_dataset = test[['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','TAX','PTRATIO','B','LSTAT']].as_matrix().astype(np.float32)


num_hidden = 4
cate_size = 18

graph = tf.Graph()
with graph.as_default():
    #data
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, feature_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,1))
    
    l1_weights = tf.Variable(tf.random_uniform(shape=[feature_size, num_hidden],minval=-1,maxval=1))
    l1_bias = tf.Variable(tf.zeros(shape=[num_hidden]))
      
    l2_weights = tf.Variable(tf.random_uniform(shape=[num_hidden,1],minval=-1,maxval=1))
    l2_bias = tf.Variable(tf.zeros(shape=[1]))
    
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    def model1(data):
        hidden1 = tf.matmul(data,l1_weights) + l1_bias
        return tf.matmul(hidden1,l2_weights) + l2_bias
        
    logits = model1(tf_train_dataset)
    loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(logits, tf_train_labels)))
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    
    train_prediction = logits
    valid_prediction = model1(tf_valid_dataset)
    test_prediction = model1(tf_test_dataset)
    
num_steps = 150
cost = []
steps = np.arange(num_steps)

def rmse(data, labels):
    return np.sqrt(np.mean((data - labels)**2))

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
       
       
       batch_data = train_dataset
       batch_labels = train_labels
  
       feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
       _, l, predictions, valid, test = session.run(
        [optimizer, loss, train_prediction, valid_prediction, test_prediction], feed_dict=feed_dict)
       
       if (step % 1 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch RMSE: %.4f' % rmse(predictions, batch_labels))
            print('Validation RMSE: %.4f' % rmse(valid, valid_labels))
            print('Test RMSE: %.4f' % rmse(test, test_labels))
            cost.append(l)

plt.title('Cost function')
plt.plot(steps, cost)

plt.show()  
plt.title('Train prediction')
plt.plot(np.arange(predictions.shape[0]),train_labels,'r*', np.arange(predictions.shape[0]), predictions, 'g.')

plt.show() 
plt.title('Valid prediction')
plt.plot(np.arange(valid.shape[0]),valid_labels,'r*', np.arange(valid.shape[0]), valid, 'g.')

plt.show()
plt.title('Test prediction')
plt.plot(np.arange(test.shape[0]),test_labels,'r*', np.arange(test.shape[0]), test, 'g.')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
