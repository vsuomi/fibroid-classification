# -*- coding: utf-8 -*-
'''
Created on Fri May 25 09:31:49 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    May 2018
    
@description:
    
    This code is used to classify the treatability of uterine fibroids before
    HIFU therapy based on their pre-treatment parameters
    
'''

#%% clear variables

%reset -f
%clear

#%% import necessary libraries

#import math

from IPython import display
#from matplotlib import cm
#from matplotlib import gridspec
#from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
#from sklearn import metrics
from sklearn import model_selection
import tensorflow as tf
#from tensorflow.python.data import Dataset
import time

#from train_linear_classification_model import train_linear_classification_model
from train_neural_network_classification_model import train_neural_network_classification_model
from scale_features import scale_features
from save_load_variables import save_load_variables
from calculate_class_weights import calculate_class_weights

#%% define logging and data display format

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#%% read data

fibroid_dataframe = pd.read_csv(r'C:\Users\visa\Documents\TYKS\Machine learning\Uterine fibroid\training_and_validation_data.csv', sep = ',')

#%% define class label (training target)

class_label = 'Reintervention'

#%% plot NPV histogram

fibroid_dataframe['NPV_percent'].hist(bins = 20)

#%% add new feature for logistic regression

NPV_threshold = 80
fibroid_dataframe['NPV_is_high'] = (fibroid_dataframe['NPV_percent'] > NPV_threshold).astype(float)

#%% create weight column

fibroid_dataframe['weight_column'] = calculate_class_weights(fibroid_dataframe[class_label])
weight_column = 'weight_column'

#weight_column = None

#%% format data

# randomise the data

fibroid_dataframe = fibroid_dataframe.reindex(np.random.permutation(fibroid_dataframe.index))

# examine data

print('\nFirst five entries of the data:\n')
display.display(fibroid_dataframe.head())
print('\nSummary of the data:\n')
display.display(fibroid_dataframe.describe())

#%% divide data into training and validation sets

# stratified splitting for unbalanced datasets

split_ratio = 0.15
training_set, validation_set = model_selection.train_test_split(fibroid_dataframe, test_size = split_ratio,
                                              stratify = fibroid_dataframe[class_label])

#%% display correlation matrix to help select suitable features

print('\nCorrelation matrix:\n')
display.display(training_set.corr())

#%% select features and targets

training_features = training_set[['white', 'black', 'asian', 'Age', 'History_of_pregnancy',
                                  'Live_births', 'esmya', 'open_myomectomy', 
                                  'laprascopic_myomectomy', 'hysteroscopic_myomectomy',
                                  'Subcutaneous_fat_thickness', 'Front-back_distance', 'Abdominal_scars',
                                  'bleeding', 'pain', 'mass', 'urinary', 'infertility',
                                  'Fibroid_distance', 'intramural', 'subserosal', 
                                  'submucosal', 'anterior', 'posterior', 'lateral', 'fundus',
                                  'anteverted', 'retroverted', 'Type_I', 'Type_II', 'Type_III',
                                  'Fibroid_volume']]
#training_features = training_set[['white', 'black', 'asian', 'Age', 'Weight', 'History_of_pregnancy',
#                                  'Live_births', 'C-section', 'esmya', 'open_myomectomy', 
#                                  'laprascopic_myomectomy', 'hysteroscopic_myomectomy',
#                                  'Subcutaneous_fat_thickness', 'Front-back_distance', 'Abdominal_scars',
#                                  'bleeding', 'pain', 'mass', 'urinary', 'infertility',
#                                  'Fibroid_size', 'Fibroid_distance', 'intramural', 'subserosal', 
#                                  'submucosal', 'anterior', 'posterior', 'lateral', 'fundus',
#                                  'anteverted', 'retroverted', 'Type_I', 'Type_II', 'Type_III',
#                                  'Fibroid_volume']]
training_targets = training_set[[class_label]]

validation_features = validation_set[['white', 'black', 'asian', 'Age', 'History_of_pregnancy',
                                  'Live_births', 'esmya', 'open_myomectomy', 
                                  'laprascopic_myomectomy', 'hysteroscopic_myomectomy',
                                  'Subcutaneous_fat_thickness', 'Front-back_distance', 'Abdominal_scars',
                                  'bleeding', 'pain', 'mass', 'urinary', 'infertility',
                                  'Fibroid_distance', 'intramural', 'subserosal', 
                                  'submucosal', 'anterior', 'posterior', 'lateral', 'fundus',
                                  'anteverted', 'retroverted', 'Type_I', 'Type_II', 'Type_III',
                                  'Fibroid_volume']]
#validation_features = validation_set[['white', 'black', 'asian', 'Age', 'Weight', 'History_of_pregnancy',
#                                  'Live_births', 'C-section', 'esmya', 'open_myomectomy', 
#                                  'laprascopic_myomectomy', 'hysteroscopic_myomectomy',
#                                  'Subcutaneous_fat_thickness', 'Front-back_distance', 'Abdominal_scars',
#                                  'bleeding', 'pain', 'mass', 'urinary', 'infertility',
#                                  'Fibroid_size', 'Fibroid_distance', 'intramural', 'subserosal', 
#                                  'submucosal', 'anterior', 'posterior', 'lateral', 'fundus',
#                                  'anteverted', 'retroverted', 'Type_I', 'Type_II', 'Type_III',
#                                  'Fibroid_volume']]
validation_targets = validation_set[[class_label]]

#%% scale features

scaling_type = 'z-score'
scaled_training_features = scale_features(training_features, scaling_type)
scaled_validation_features = scale_features(validation_features, scaling_type)

#%% add weight column

if weight_column is not None:
    scaled_training_features['weight_column'] = training_set['weight_column']
    scaled_validation_features['weight_column'] = validation_set['weight_column']

#%% train using neural network classification model function

# define parameters

learning_rate = 0.001
steps = 3200
batch_size = 5
hidden_units = [25]
dropout = None
batch_norm = True
optimiser = 'Adam'
save_model = True

# directory for saving the model

if save_model is True:
    timestr = time.strftime('%Y%m%d-%H%M%S')
    model_dir = 'models\\' + timestr
else:
    model_dir = None

# train the model

dnn_classifier, training_probabilities, validation_probabilities = train_neural_network_classification_model(
    learning_rate = learning_rate,
    steps = steps,
    batch_size = batch_size,
    hidden_units = hidden_units,
    weight_column = weight_column,
    dropout = dropout,
    batch_norm = batch_norm,
    optimiser = optimiser,
    model_dir = model_dir,
    training_features = scaled_training_features,
    training_targets = training_targets,
    validation_features = scaled_validation_features,
    validation_targets = validation_targets)

# save variables

if save_model is True:

    variables_to_save = {'learning_rate': learning_rate,
                         'steps': steps,
                         'batch_size': batch_size,
                         'hidden_units': hidden_units,
                         'weight_column': weight_column,
                         'dropout': dropout,
                         'batch_norm': batch_norm,
                         'optimiser': optimiser,
                         'model_dir': model_dir,
                         'training_set': training_set,
                         'training_features': training_features,
                         'scaled_training_features': scaled_training_features,
                         'training_targets': training_targets,
                         'training_probabilites': training_probabilities,
                         'validation_set': validation_set,
                         'validation_features': validation_features,
                         'scaled_validation_features': scaled_validation_features,
                         'validation_targets': validation_targets,
                         'validation_probabilities': validation_probabilities,
                         'fibroid_dataframe': fibroid_dataframe,
                         'split_ratio': split_ratio,
                         'timestr': timestr,
                         'scaling_type': scaling_type,
                         'NPV_threshold': NPV_threshold,
                         'class_label': class_label}
    
    save_load_variables(model_dir, variables_to_save, 'save')
