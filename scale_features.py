# -*- coding: utf-8 -*-
'''
Created on Thu May 31 11:38:48 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    May 2018
    
@description:
    
    This function is used to scale features using different scaling types
    
'''

#%% import necessary packages

import numpy as np
import pandas as pd

#%% define function

def scale_features(features, scaling):
    
    ''' Scales given features with standard deviation
    
    Args:
        features: pandas Dataframe of features
        scaling: type of scaling: linear ('linear'), logarithmic ('log') or
        z-score ('z-score')
    Returns:
        scaled_features: scaled features
    '''
    
    if scaling == 'linear':
        min_val = features.min()
        max_val = features.max()
        scale = (max_val - min_val) / 2.0
        a = (features - min_val)
        b = scale
        scaled_features = np.divide(a, b, out=np.zeros_like(a), where=b!=0) - 1.0   # NaN to zero - 1
    elif scaling == 'log':
        scaled_features = np.log(features + 1.0)
    elif scaling == 'z-score':
        a = (features - features.mean())
        b = features.std()
        scaled_features = np.divide(a, b, out=np.zeros_like(a), where=b!=0)     # NaN to zero
    else:
        print('Unknown scaling type')
        scaled_features = features
        
    scaled_features = pd.DataFrame(scaled_features, columns = list(features), 
                                            index = features.index, dtype = float)
    
    return scaled_features