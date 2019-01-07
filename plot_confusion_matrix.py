# -*- coding: utf-8 -*-
'''
Created on Mon Jan  7 13:10:03 2019

@author:
    
    Visa Suomi
    Turku University Hospital
    January 2019
    
@description:
    
    This function is used for plotting the confusion matrix
    
'''

#%% import necessary libraries

import matplotlib.pyplot as plt
import seaborn as sns

#%% define function

def plot_confusion_matrix(cm):
    
    f1 = plt.figure(figsize = (6, 4))
    
    ax = sns.heatmap(cm, cmap = 'bone_r')
    ax.set_aspect(1)
#    plt.title('Confusion matrix')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    
    return f1