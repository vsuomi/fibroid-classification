# -*- coding: utf-8 -*-
'''
Created on Mon Sep 10 12:44:11 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    October 2018
    
@description:
    
    This script is used for evaluating the peformance of pretrained 
    neural network model using unseen test data.
    
'''

#%% clear variables

%reset -f
%clear

#%% import necessary libraries

#import math

from IPython import display
#from matplotlib import cm
#from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
#from tensorflow.python.data import Dataset
#import time
import seaborn as sns

from scale_features import scale_features
from save_load_variables import save_load_variables
from test_neural_network_softmax_classification_model import test_neural_network_softmax_classification_model

#%% load variables

model_dir = 'models\\20181220-132846'
variables_to_save = None
variables = save_load_variables(model_dir, variables_to_save, 'variables', 'load')
for key,val in variables.items():
        exec(key + '=val')
               
#%% calculate test predictions
        
dnn_classifier, testing_predictions = test_neural_network_softmax_classification_model(
        learning_rate, 
        steps, 
        batch_size, 
        hidden_units,
        n_classes,
        weight_column,
        dropout,
        batch_norm,
        optimiser,
        model_dir,
        testing_features,
        testing_targets
        )

#%% calculate evaluation metrics

# accuracy

training_accuracy = metrics.accuracy_score(training_targets, training_predictions)
validation_accuracy = metrics.accuracy_score(validation_targets, validation_predictions)
testing_accuracy = metrics.accuracy_score(testing_targets, testing_predictions)

# LogLoss

training_pred_one_hot = tf.keras.utils.to_categorical(training_predictions, n_classes)
validation_pred_one_hot = tf.keras.utils.to_categorical(validation_predictions, n_classes) 
testing_pred_one_hot = tf.keras.utils.to_categorical(testing_predictions, n_classes)  

training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
testing_log_loss = metrics.log_loss(testing_targets, testing_pred_one_hot)

# confusion matrix

cm_training = metrics.confusion_matrix(training_targets, training_predictions)
cm_validation = metrics.confusion_matrix(validation_targets, validation_predictions)
cm_testing = metrics.confusion_matrix(testing_targets, testing_predictions)

# confusion matrix (normalised)

cm_training_normalized = cm_training.astype('float') / cm_training.sum(axis = 1)[:, np.newaxis]
cm_validation_normalized = cm_validation.astype('float') / cm_validation.sum(axis = 1)[:, np.newaxis]
cm_testing_normalized = cm_testing.astype('float') / cm_testing.sum(axis = 1)[:, np.newaxis]

#%% plot figures

# confusion matrix

plt.figure(figsize = (6, 4))
ax = sns.heatmap(cm_training_normalized, cmap = 'bone_r')
ax.set_aspect(1)
#plt.title('Confusion matrix (training)')
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.figure(figsize = (6, 4))
ax = sns.heatmap(cm_validation_normalized, cmap = 'bone_r')
ax.set_aspect(1)
#plt.title('Confusion matrix (validation)')
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.figure(figsize = (6, 4))
ax = sns.heatmap(cm_testing_normalized, cmap = 'bone_r')
ax.set_aspect(1)
#plt.title('Confusion matrix (testing)')
plt.ylabel('True label')
plt.xlabel('Predicted label')
           