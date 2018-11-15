# -*- coding: utf-8 -*-
'''
Created on Wed Nov 14 14:53:46 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    October 2018
    
@description:
    
    This script is used for evaluating the peformance of pretrained 
    neural network model (Keras) using unseen test data
    
'''

#%% clear variables

%reset -f
%clear

#%% import necessary libraries

import keras as k
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from save_load_variables import save_load_variables

#%% load model and variables

model_dir = 'Keras models\\20181114-143637_TA88_VA30'

variables_to_save = None
variables = save_load_variables(model_dir, variables_to_save, 'load')
for key,val in variables.items():
        exec(key + '=val')
        
model = k.models.load_model(model_dir + '\\keras_model.h5')
               
#%% evaluate model performance

# calculate loss metrics

training_loss, training_accuracy = model.evaluate(training_features, training_targets)
validation_loss, validation_accuracy = model.evaluate(validation_features, validation_targets)
testing_loss, testing_accuracy = model.evaluate(testing_features, testing_targets)

# make predictions

training_predictions = model.predict(training_features)
training_predictions = np.argmax(training_predictions, axis = 1)
training_predictions = pd.DataFrame(training_predictions, columns = target_label,
                                    index = training_features.index, dtype = float)

validation_predictions = model.predict(validation_features)
validation_predictions = np.argmax(validation_predictions, axis = 1)
validation_predictions = pd.DataFrame(validation_predictions, columns = target_label,
                                      index = validation_features.index, dtype = float)

testing_predictions = model.predict(testing_features)
testing_predictions = np.argmax(testing_predictions, axis = 1)
testing_predictions = pd.DataFrame(testing_predictions, columns = target_label,
                                      index = testing_features.index, dtype = float)

# confusion matrix

cm_training = confusion_matrix(training_targets, training_predictions)
cm_training = cm_training.astype('float') / cm_training.sum(axis = 1)[:, np.newaxis]

cm_validation = confusion_matrix(validation_targets, validation_predictions)
cm_validation = cm_validation.astype('float') / cm_validation.sum(axis = 1)[:, np.newaxis]

cm_testing = confusion_matrix(testing_targets, testing_predictions)
cm_testing = cm_testing.astype('float') / cm_testing.sum(axis = 1)[:, np.newaxis]

#%% plot figures

# confusion matrix

f1 = plt.figure(figsize = (6, 4))
ax = sns.heatmap(cm_training, cmap = 'bone_r')
ax.set_aspect(1)
#plt.title('Confusion matrix (training)')
plt.ylabel('True label')
plt.xlabel('Predicted label')

f2 = plt.figure(figsize = (6, 4))
ax = sns.heatmap(cm_validation, cmap = 'bone_r')
ax.set_aspect(1)
#plt.title('Confusion matrix (validation)')
plt.ylabel('True label')
plt.xlabel('Predicted label')

f3 = plt.figure(figsize = (6, 4))
ax = sns.heatmap(cm_testing, cmap = 'bone_r')
ax.set_aspect(1)
#plt.title('Confusion matrix (testing)')
plt.ylabel('True label')
plt.xlabel('Predicted label')

#%% save figures

f1.savefig(model_dir + '\\' + 'cm_training.eps', dpi = 600, format = 'eps',
            bbox_inches = 'tight', pad_inches = 0)
f1.savefig(model_dir + '\\' + 'cm_training.pdf', dpi = 600, format = 'pdf',
            bbox_inches = 'tight', pad_inches = 0)
f1.savefig(model_dir + '\\' + 'cm_training.png', dpi = 600, format = 'png',
            bbox_inches = 'tight', pad_inches = 0)

f2.savefig(model_dir + '\\' + 'cm_validation.eps', dpi = 600, format = 'eps',
            bbox_inches = 'tight', pad_inches = 0)
f2.savefig(model_dir + '\\' + 'cm_validation.pdf', dpi = 600, format = 'pdf',
            bbox_inches = 'tight', pad_inches = 0)
f2.savefig(model_dir + '\\' + 'cm_validation.png', dpi = 600, format = 'png',
            bbox_inches = 'tight', pad_inches = 0)

f3.savefig(model_dir + '\\' + 'cm_testing.eps', dpi = 600, format = 'eps',
            bbox_inches = 'tight', pad_inches = 0)
f3.savefig(model_dir + '\\' + 'cm_testing.pdf', dpi = 600, format = 'pdf',
            bbox_inches = 'tight', pad_inches = 0)
f3.savefig(model_dir + '\\' + 'cm_testing.png', dpi = 600, format = 'png',
            bbox_inches = 'tight', pad_inches = 0)