# -*- coding: utf-8 -*-
"""
Created on Tue May 29 13:39:30 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    May 2018
    
@description:
    
    This function is used to construct feature columns for TensorFlow input
    
"""

#%% import necessary packages

import tensorflow as tf

#%% define function

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])