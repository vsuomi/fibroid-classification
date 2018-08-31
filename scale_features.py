# -*- coding: utf-8 -*-
'''
Created on Thu May 31 11:38:48 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    May 2018
    
@description:
    
    This function is used to scale features using standard deviation
    
'''

#%% import necessary packages

import numpy as np

#%% define function

def scale_features(features, scaling):
    
    ''' Scales given features with standard deviation
    
    Args:
        features: pandas Dataframe of features
        scaling: type of scaling: linear ('linear'), logarithmic ('log') or
        z-score ('z-score')
    Returns:
        scaled_features: scaled features with zero mean and unit variance
    '''
    
    if scaling == 'linear':
        min_val = features.min()
        max_val = features.max()
        scale = (max_val - min_val) / 2.0
        scaled_features = (features - min_val) / scale - 1.0
    elif scaling == 'log':
        scaled_features = np.log(features + 1.0)
    elif scaling == 'z-score':
        scaled_features = (features - features.mean()) / features.std()
    else:
        print('Unknown scaling type')
        scaled_features = features
    
    return scaled_features