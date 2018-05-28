# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:34:26 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    May 2018
    
@description:
    
    This function defines input function for given features and targets
    
"""

#%% import necessary packages

import numpy as np
from tensorflow.python.data import Dataset

#%% define function

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    
    """ Trains model with given features
    
    Args:
        features: pandas Dataframe of features
        targets: pandas Dataframe of targets
        batch_size: number of examples to calculate the gradiet
        shuffle: boolean to shuffle the data
        num_epochs: number of iterations, None = repeat indefinitely
    Returns:
        (features, labels) for next data batch
    """
    
    # convert pandas data into a dict of np arrays
    
    features = {key:np.array(value) for key, value in dict(features).items()}
    
    # construct a dataset and configure batching/repeating
    
    ds = Dataset.from_tensor_slices((features, targets)) # 2 GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # shuffle data if selected
    
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
        
    # return the next batch of data
    
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels