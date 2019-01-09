# -*- coding: utf-8 -*-
'''
Created on Tue Sep 18 10:34:21 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    September 2018
    
@description:
    
    This code is used to classify (using softmax) the treatability of uterine 
    fibroids before HIFU therapy based on their pre-treatment parameters
    
'''

#%% clear variables

%reset -f
%clear

#%% import necessary libraries

from IPython import display
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import tensorflow as tf
import time

from train_neural_network_softmax_classification_model import train_neural_network_softmax_classification_model
from save_load_variables import save_load_variables

#%% define logging and data display format

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#%% read data

fibroid_dataframe = pd.read_csv(r'C:\Users\visa\Documents\TYKS\Machine learning\Uterine fibroid\fibroid_dataframe_combined.csv', sep = ',')

#%% display NPV histogram

fibroid_dataframe['NPV_percent'].hist(bins = 20)

#%% categorise NPV into classes according to bins

NPV_bins = [-1, 29.9, 80, 100]
fibroid_dataframe['NPV_class'] = fibroid_dataframe[['NPV_percent']].apply(lambda x: pd.cut(x, NPV_bins, labels = False))

#%% define feature and target labels

feature_labels = ['V2_system',
                  'Subcutaneous_fat_thickness', 'Abdominal_scars',
                  'Fibroid_diameter', 'Fibroid_distance', 
                  'anteverted', 'retroverted', 'vertical']

#feature_labels = ['white', 'black', 'asian', 'Age', 'Weight', 'History_of_pregnancy',
#                  'Live_births', 'C-section', 'esmya', 'open_myomectomy', 
#                  'laprascopic_myomectomy', 'hysteroscopic_myomectomy',
#                  'Subcutaneous_fat_thickness', 'Front-back_distance', 'Abdominal_scars',
#                  'bleeding', 'pain', 'mass', 'urinary', 'infertility',
#                  'Fibroid_diameter', 'Fibroid_distance', 'intramural', 'subserosal', 
#                  'submucosal', 'anterior', 'posterior', 'lateral', 'fundus',
#                  'anteverted', 'retroverted', 'Type_I', 'Type_II', 'Type_III',
#                  'Fibroid_volume']

target_label = ['NPV_class']

#%% randomise and divive data for cross-validation

# stratified splitting for unbalanced datasets

split_ratio = 40
training_set, holdout_set = train_test_split(fibroid_dataframe, test_size = split_ratio,
                                             stratify = fibroid_dataframe[target_label])
validation_set, testing_set = train_test_split(holdout_set, test_size = int(split_ratio / 2),
                                               stratify = holdout_set[target_label])

del holdout_set

#%% define features and targets

training_features = training_set[feature_labels]
validation_features = validation_set[feature_labels]
testing_features = testing_set[feature_labels]

training_targets = training_set[target_label]
validation_targets = validation_set[target_label]
testing_targets = testing_set[target_label]

#%% scale features

scaling_type = 'z-score'

z_mean = training_features.mean()
z_std = training_features.std()

training_features = (training_features - z_mean) / z_std
validation_features = (validation_features - z_mean) / z_std
testing_features = (testing_features - z_mean) / z_std

#%% create weight column

weight_column = 'weight_column'

training_features[weight_column] = compute_sample_weight('balanced', training_targets)
validation_features[weight_column] = np.ones(validation_targets.shape[0])
testing_features[weight_column] = np.ones(testing_targets.shape[0])

#weight_column = None

if weight_column is not None:
    feature_labels.append(weight_column)

#%% train using neural network classification model function

# define parameters

learning_rate = 0.001
steps = 7000
batch_size = 5
hidden_units = [32]
n_classes = 3
dropout = 0.2
batch_norm = False
optimiser = 'Adam'
save_model = True

# directory for saving the model

if save_model is True:
    timestr = time.strftime('%Y%m%d-%H%M%S')
    model_dir = 'models\\' + timestr
else:
    model_dir = None

# train the model

dnn_classifier, training_predictions, validation_predictions = train_neural_network_softmax_classification_model(
    learning_rate = learning_rate,
    steps = steps,
    batch_size = batch_size,
    hidden_units = hidden_units,
    n_classes = n_classes,
    weight_column = weight_column,
    dropout = dropout,
    batch_norm = batch_norm,
    optimiser = optimiser,
    model_dir = model_dir,
    training_features = training_features,
    training_targets = training_targets,
    validation_features = validation_features,
    validation_targets = validation_targets)

# save variables

if save_model is True:

    variables_to_save = {'learning_rate': learning_rate,
                         'steps': steps,
                         'batch_size': batch_size,
                         'hidden_units': hidden_units,
                         'n_classes': n_classes,
                         'weight_column': weight_column,
                         'class_weights': class_weights,
                         'dropout': dropout,
                         'batch_norm': batch_norm,
                         'optimiser': optimiser,
                         'model_dir': model_dir,
                         'training_set': training_set,
                         'training_features': training_features,
                         'training_targets': training_targets,
                         'training_predictions': training_predictions,
                         'validation_set': validation_set,
                         'validation_features': validation_features,
                         'validation_targets': validation_targets,
                         'validation_predictions': validation_predictions,
                         'testing_set': testing_set,
                         'testing_features': testing_features,
                         'testing_targets': testing_targets,
                         'fibroid_dataframe': fibroid_dataframe,
                         'split_ratio': split_ratio,
                         'timestr': timestr,
                         'scaling_type': scaling_type,
                         'z_mean': z_mean,
                         'z_std': z_std,
                         'NPV_bins': NPV_bins,
                         'feature_labels': feature_labels,
                         'target_label': target_label}
    
    save_load_variables(model_dir, variables_to_save, 'variables', 'save')
