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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import scipy as sp
import time
import os

from plot_softmax_classification_performance import plot_softmax_classification_performance
from save_load_variables import save_load_variables

#%% define logging and data display format

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

split_ratio = 40
training_set, holdout_set = train_test_split(concat_dataframe, test_size = split_ratio,
                                             stratify = concat_dataframe[target_label])
validation_set, testing_set = train_test_split(holdout_set, test_size = int(split_ratio / 2),
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

learning_rate = 0.001
n_epochs = 700
n_neurons = 64
n_layers = 1
n_classes = 3
batch_size = 5
l1_reg = 0.0
l2_reg = 0.02
batch_norm = False
dropout = None

# build model

if 'model' in locals():
    del model

model = k.models.Sequential()

model.add(k.layers.Dense(n_neurons, 
                         input_shape = (training_features.shape[1],),
                         kernel_regularizer = k.regularizers.l1_l2(l1 = l1_reg, l2 = l2_reg),
                         activation = 'relu'))
if batch_norm is True:
    model.add(k.layers.BatchNormalization())
if dropout is not None:
    model.add(k.layers.Dropout(dropout))
    
i = 1   
while i < n_layers:
    model.add(k.layers.Dense(n_neurons,
                             kernel_regularizer = k.regularizers.l1_l2(l1 = l1_reg, l2 = l2_reg),
                             activation = 'relu'))
    if batch_norm is True:
        model.add(k.layers.BatchNormalization())
    if dropout is not None:
        model.add(k.layers.Dropout(dropout))
    i += 1
del i

model.add(k.layers.Dense(n_classes, 
                         activation = 'softmax'))

model.compile(optimizer = k.optimizers.Adamax(lr = learning_rate),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.summary()

# train model

class PrintDot(k.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
    
timestr = time.strftime('%Y%m%d-%H%M%S')

history = model.fit(training_features, training_targets, verbose = 0, callbacks = [PrintDot()],
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

cm_training = confusion_matrix(training_targets, training_predictions)
cm_training = cm_training.astype('float') / cm_training.sum(axis = 1)[:, np.newaxis]

cm_validation = confusion_matrix(validation_targets, validation_predictions)
cm_validation = cm_validation.astype('float') / cm_validation.sum(axis = 1)[:, np.newaxis]

# plot training performance

f1 = plot_softmax_classification_performance('keras', history, cm_training, cm_validation)

#%% save model

model_dir = 'Keras models\\%s_TA%d_VA%d' % (timestr, 
                                            round(training_accuracy*100), 
                                            round(validation_accuracy*100))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
f1.savefig(model_dir + '\\' + 'evaluation_metrics.pdf', dpi = 600, format = 'pdf',
                    bbox_inches = 'tight', pad_inches = 0)

variables_to_save = {'learning_rate': learning_rate,
                     'n_epochs': n_epochs,
                     'n_neurons': n_neurons,
                     'n_classes': n_classes,
                     'n_layers': n_layers,
                     'batch_size': batch_size,
                     'l1_reg': l1_reg,
                     'l2_reg': l2_reg,
                     'batch_norm': batch_norm,
                     'dropout': dropout,
                     'class_weights': class_weights,
                     'NPV_bins': NPV_bins,
                     'split_ratio': split_ratio,
                     'timestr': timestr,
                     'history': history,
                     'model_dir': model_dir,
                     'fibroid_dataframe': fibroid_dataframe,
                     'concat_dataframe': concat_dataframe,
                     'holdout_set': holdout_set,
                     'training_set': training_set,
                     'training_features': training_features,
                     'training_targets': training_targets,
                     'validation_set': validation_set,
                     'validation_features': validation_features,
                     'validation_targets': validation_targets,
                     'testing_set': testing_set,
                     'testing_features': testing_features,
                     'testing_targets': testing_targets,
                     'scaled_features': scaled_features,
                     'features': features,
                     'targets': targets,
                     'feature_labels': feature_labels,
                     'target_label': target_label}
    
save_load_variables(model_dir, variables_to_save, 'variables', 'save')

model.save(model_dir + '\\' + 'keras_model.h5')
