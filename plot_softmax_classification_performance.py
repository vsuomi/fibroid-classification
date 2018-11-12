# -*- coding: utf-8 -*-
'''
Created on Fri Nov  9 09:03:03 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    November 2018
    
@description:
    
    This function is used for plotting the performance metrics from a trained
    Keras model
    
'''

#%% import necessary libraries

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#%% define function

def plot_softmax_classification_performance(history, cm_training, cm_validation):
    
    # training logloss
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('LogLoss')
    plt.plot(history.epoch, np.array(history.history['loss']),
             label='Training loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
             label = 'Validation loss')
    plt.legend()
    plt.grid()
    
    # confusion matrix (training)
    
    plt.figure()
    ax = sns.heatmap(cm_training, cmap = 'bone_r')
    ax.set_aspect(1)
    #plt.title('Confusion matrix (training)')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # confusion matrix (validation)
    
    plt.figure()
    ax = sns.heatmap(cm_validation, cmap = 'bone_r')
    ax.set_aspect(1)
    #plt.title('Confusion matrix (validation)')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
