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

def plot_softmax_classification_performance(model, losses, cm_training, cm_validation):
    
    # training logloss
    
    f1 = plt.figure(figsize = (18, 4))
    plt.subplot(1, 3, 1)
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    if model == 'keras':
        plt.plot(losses.epoch, np.array(losses.history['loss']),
                 label = 'Training')
        plt.plot(losses.epoch, np.array(losses.history['val_loss']),
                 label = 'Validation')
    if model == 'xgboost':
        plt.plot(np.array(losses['training']['merror']),
                 label = 'Training')
        plt.plot(np.array(losses['validation']['merror']),
                 label = 'Validation')
    plt.grid()
    plt.legend()
    plt.legend()
    plt.grid()
    
    # confusion matrix (training)
    
#    plt.figure()
    plt.subplot(1, 3, 2)
    ax = sns.heatmap(cm_training, cmap = 'bone_r')
    ax.set_aspect(1)
    plt.title('Confusion matrix (training)')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    
    # confusion matrix (validation)
    
#    plt.figure()
    plt.subplot(1, 3, 3)
    ax = sns.heatmap(cm_validation, cmap = 'bone_r')
    ax.set_aspect(1)
    plt.title('Confusion matrix (validation)')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    
    return f1
