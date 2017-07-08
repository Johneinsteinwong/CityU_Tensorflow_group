#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:21:33 2017

@author: wongtszlunjohneinstein
numberic features and encoded categorica one-hot vector as input
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


batch_size = 323
feature_size = 29

whole = pd.read_csv('boston_housing.csv')
train, test = train_test_split(whole, test_size=0.2, random_state=42)
train, valid = train_test_split(train, test_size=0.2, random_state=42)

train_labels = train[['MEDV']].as_matrix().astype(np.float32)
valid_labels = valid[['MEDV']].as_matrix().astype(np.float32)
test_labels = test[['MEDV']].as_matrix().astype(np.float32)

train_dataset = train[['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','TAX','PTRATIO','B','LSTAT']].as_matrix().astype(np.float32)
valid_dataset = valid[['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','TAX','PTRATIO','B','LSTAT']].as_matrix().astype(np.float32)
test_dataset = test[['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','TAX','PTRATIO','B','LSTAT']].as_matrix().astype(np.float32)

chas = whole[['CHAS']].astype(str).drop_duplicates()
rad = whole[['RAD']].astype(str).drop_duplicates()

category = []
for i in range(chas.shape[0]):
    for j in range(rad.shape[0]):
        category.append(chas['CHAS'].iloc[i]+','+rad['RAD'].iloc[j])


train_cate = train[['CHAS','RAD']].astype(str)
valid_cate = valid[['CHAS','RAD']].astype(str)
test_cate = test[['CHAS','RAD']].astype(str)


for i in range(train_cate.shape[0]):
    train_cate['CHAS'].iloc[i] = train_cate['CHAS'].iloc[i] + ',' + train_cate['RAD'].iloc[i]
train_cate = train_cate.drop('RAD',axis=1)
train_cate = train_cate.rename(columns = {'CHAS':'c'})

for i in range(valid_cate.shape[0]):
    valid_cate['CHAS'].iloc[i] = valid_cate['CHAS'].iloc[i] + ',' + valid_cate['RAD'].iloc[i]
valid_cate = valid_cate.drop('RAD',axis=1)
valid_cate = valid_cate.rename(columns = {'CHAS':'c'})

for i in range(test_cate.shape[0]):
    test_cate['CHAS'].iloc[i] = test_cate['CHAS'].iloc[i] + ',' + test_cate['RAD'].iloc[i]
test_cate = test_cate.drop('RAD',axis=1)
test_cate = test_cate.rename(columns = {'CHAS':'c'})

train_cate = pd.Categorical(train_cate['c'], categories=category)
valid_cate = pd.Categorical(valid_cate['c'], categories=category)
test_cate = pd.Categorical(test_cate['c'], categories=category)

train_dum = pd.get_dummies(train_cate)
valid_dum = pd.get_dummies(valid_cate)
test_dum = pd.get_dummies(test_cate)

train_dum = pd.DataFrame(train_dum).as_matrix().astype(np.float32)
valid_dum = pd.DataFrame(valid_dum).as_matrix().astype(np.float32)
test_dum = pd.DataFrame(test_dum).as_matrix().astype(np.float32)

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
    
    tf_valid_dataset = tf.constant(np.concatenate((valid_dataset,valid_dum),axis=1))
    tf_test_dataset = tf.constant(np.concatenate((test_dataset,test_dum),axis=1))
 
    def model(data):
        hidden1 = tf.matmul(data,l1_weights) + l1_bias
        return tf.matmul(hidden1,l2_weights) + l2_bias
        
    logits = model(tf_train_dataset)
    loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(logits, tf_train_labels)))
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    
    train_prediction = logits
    valid_prediction = model(tf_valid_dataset)
    test_prediction = model(tf_test_dataset)
    
def rmse(data, labels):
    return np.sqrt(np.mean((data - labels)**2))

num_steps = 150
cost = []
steps = np.arange(num_steps)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
       #offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
       #batch_data = train_dataset[offset:(offset + batch_size),:]
       #batch_labels = train_labels[offset:(offset + batch_size),:]
       #batch_dum = train_dum[offset:(offset + batch_size),:]
       
       batch_data = np.concatenate((train_dataset,train_dum),axis=1)
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
plt.plot(np.arange(predictions.shape[0]),train_labels,'r*',label='train label')
plt.plot(np.arange(predictions.shape[0]),predictions, 'g.',label='train prediction')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show() 
plt.title('Valid prediction')
plt.plot(np.arange(valid.shape[0]),valid_labels,'r*',label='valid label') 
plt.plot(np.arange(valid.shape[0]),valid, 'g.',label='valid prediction')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plt.title('Test prediction')
plt.plot(np.arange(test.shape[0]),test_labels,'r*',label='test label')
plt.plot(np.arange(test.shape[0]),test, 'g.',label='test prediction')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
