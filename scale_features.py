# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:38:48 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    May 2018
    
@description:
    
    This function is used to scale features using standard deviation
    
"""

#%% import necessary packages



#%% define function

def scale_features(features):
    
    """ Scales given features with standard deviation
    
    Args:
        features: pandas Dataframe of features
    Returns:
        scaled_features: scaled features with zero mean and unit variance
    """
    
    scaled_features = (features - features.mean()) / features.std()
    
    return scaled_features