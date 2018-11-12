# -*- coding: utf-8 -*-
'''
Created on Thu Nov  8 13:11:40 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    November 2018
    
@description:
    
    This code is used to classify (using softmax) the treatability of uterine 
    fibroids before HIFU therapy based on their pre-treatment parameters using
    Keras
    
'''

#%% clear variables

%reset -f
%clear

#%% import necessary libraries

import keras as k
import pandas as pd
import numpy as np
import sklearn as sk
import scipy as sp

from plot_softmax_classification_performance import plot_softmax_classification_performance

#%% define logging and data display format

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#%% read data

fibroid_dataframe = pd.read_csv(r'C:\Users\visa\Documents\TYKS\Machine learning\Uterine fibroid\fibroid_dataframe.csv', sep = ',')

#%% display NPV histogram

fibroid_dataframe['NPV_percent'].hist(bins = 20)

#%% categorise NPV into classes according to bins

NPV_bins = [-1, 29.9, 80, 100]
fibroid_dataframe['NPV_class'] = fibroid_dataframe['NPV_percent'].apply(lambda x: pd.cut(x, NPV_bins, labels = False))

#%% define feature and target labels

feature_labels = ['Subcutaneous_fat_thickness', 'Abdominal_scars',
                  'Fibroid_diameter', 'Fibroid_distance', 
                  'anteverted', 'retroverted']

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

#%% extract features and targets

features = fibroid_dataframe[feature_labels]
targets = fibroid_dataframe[target_label]

#%% scale features

scaled_features = pd.DataFrame(sp.stats.mstats.zscore(features),
                               columns = list(features), 
                               index = features.index, dtype = float)

#%% calculate class weights

class_weights = sk.utils.class_weight.compute_class_weight('balanced', np.unique(targets), 
                                                  targets[target_label[0]])
class_weights = dict(enumerate(class_weights))

#%% combine dataframes

concat_dataframe = pd.concat([scaled_features, targets], axis = 1)

#%% randomise and divive data for cross-validation

# stratified splitting for unbalanced datasets

split_ratio = 20
training_set, holdout_set = sk.model_selection.train_test_split(concat_dataframe, test_size = split_ratio,
                                              stratify = concat_dataframe[target_label])
validation_set, testing_set = sk.model_selection.train_test_split(holdout_set, test_size = int(split_ratio / 2),
                                              stratify = holdout_set[target_label])

#%% define features and targets

training_features = training_set[feature_labels]
validation_features = validation_set[feature_labels]
testing_features = testing_set[feature_labels]

training_targets = training_set[target_label]
validation_targets = validation_set[target_label]
testing_targets = testing_set[target_label]

#%% build and train neural network model

# define parameters

optimiser = 'adam'
n_epochs = 300
n_neurons = 25
n_classes = 3
batch_size = 5
l1_reg = 0.0
l2_reg = 0.1

# build model

model = k.models.Sequential()

model.add(k.layers.Dense(n_neurons, 
                         input_shape = (training_features.shape[1],),
                         kernel_regularizer = k.regularizers.l1_l2(l1 = l1_reg, l2 = l2_reg),
                         activation = 'relu'))
model.add(k.layers.Dense(n_classes, 
                         activation = 'softmax'))

model.compile(optimizer = optimiser,
              loss = 'sparse_categorical_crossentropy',
              metrics = ['categorical_accuracy'])

# train model

history = model.fit(training_features, training_targets, verbose = 0,
                    batch_size = batch_size, epochs = n_epochs, class_weight = class_weights,
                    validation_data = (validation_features, validation_targets))

#%% evaluate model performance

# calculate loss metrics

training_loss, training_accuracy = model.evaluate(training_features, training_targets)
validation_loss, validation_accuracy = model.evaluate(validation_features, validation_targets)

# make predictions

training_predictions = model.predict(training_features)
training_predictions = np.argmax(training_predictions, axis = 1)
training_predictions = pd.DataFrame(training_predictions, columns = target_label,
                                    index = training_features.index, dtype = float)

validation_predictions = model.predict(validation_features)
validation_predictions = np.argmax(validation_predictions, axis = 1)
validation_predictions = pd.DataFrame(validation_predictions, columns = target_label,
                                      index = validation_features.index, dtype = float)

# confusion matrix

cm_training = sk.metrics.confusion_matrix(training_targets, training_predictions)
cm_training = cm_training.astype('float') / cm_training.sum(axis = 1)[:, np.newaxis]

cm_validation = sk.metrics.confusion_matrix(validation_targets, validation_predictions)
cm_validation = cm_validation.astype('float') / cm_validation.sum(axis = 1)[:, np.newaxis]

# plot training performance

plot_softmax_classification_performance(history, cm_training, cm_validation)
