# -*- coding: utf-8 -*-
'''
Created on Tue Sep 18 16:00:21 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    September 2018
    
@description:
    
    This function is used to calculate class weights for imbalanced classes
    
'''

#%% import necessary packages

import numpy as np

#%% define function

def calculate_class_weights(classes):
    
    ''' Calculated weights for imbalanced datasets
    
    Args:
        classes: array of classes
    Returns:
        weight_column: weight column for classes
    '''
    
    # number of instances and unique classes
    
    n_instances = len(classes)
    unique_values, unique_counts = np.unique(classes, return_counts = True)
    n_classes = len(unique_values)
    
    # calculate class weights for each class
    
    weights = n_instances / (n_classes * unique_counts)
    weights = weights / weights.sum()
    
    # create weight column
    
    weight_column = classes.copy()
    weight_column = weight_column.as_matrix()
    
    for i in range(0, n_classes):
        weight_column = np.where(weight_column == unique_values[i], weights[i], weight_column)
        
    return weight_column