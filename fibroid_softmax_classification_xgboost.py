# -*- coding: utf-8 -*-
'''
Created on Wed Dec 19 13:05:13 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    November 2018
    
@description:
    
    This code is used to classify (using softmax) the treatability of uterine 
    fibroids before HIFU therapy based on their pre-treatment parameters using
    XGBoost
    
'''

#%% clear variables

%reset -f
%clear

#%% import necessary libraries

import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import scipy as sp
import time
import os

from plot_softmax_classification_performance import plot_softmax_classification_performance
from plot_feature_importance import plot_feature_importance
from save_load_variables import save_load_variables

#%% define logging and data display format

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#%% read data

fibroid_dataframe = pd.read_csv(r'C:\Users\visa\Documents\TYKS\Machine learning\Uterine fibroid\fibroid_dataframe.csv', sep = ',')

#%% calculate nan percent for each label

nan_percent = pd.DataFrame(fibroid_dataframe.isnull().mean() * 100, columns = ['% of NaN'])

#%% replace nan values

fibroid_dataframe['Height'] = fibroid_dataframe['Height'].fillna(fibroid_dataframe['Height'].mean())
fibroid_dataframe['Gravidity'] = fibroid_dataframe['Gravidity'].fillna(fibroid_dataframe['Gravidity'].mode()[0])
fibroid_dataframe['bleeding'] = fibroid_dataframe['bleeding'].fillna(fibroid_dataframe['bleeding'].mode()[0])
fibroid_dataframe['pain'] = fibroid_dataframe['pain'].fillna(fibroid_dataframe['pain'].mode()[0])
fibroid_dataframe['mass'] = fibroid_dataframe['mass'].fillna(fibroid_dataframe['mass'].mode()[0])
fibroid_dataframe['urinary'] = fibroid_dataframe['urinary'].fillna(fibroid_dataframe['urinary'].mode()[0])
fibroid_dataframe['infertility'] = fibroid_dataframe['infertility'].fillna(fibroid_dataframe['infertility'].mode()[0])
fibroid_dataframe['ADC'] = fibroid_dataframe['ADC'].fillna(fibroid_dataframe['ADC'].mean())

#%% display NPV histogram

fibroid_dataframe['NPV_percent'].hist(bins = 20)

#%% categorise NPV into classes according to bins

NPV_bins = [-1, 29.9, 80, 100]
fibroid_dataframe['NPV_class'] = fibroid_dataframe[['NPV_percent']].apply(lambda x: pd.cut(x, NPV_bins, labels = False))

#%% define feature and target labels

#feature_labels = ['Subcutaneous_fat_thickness', 'Abdominal_scars',
#                  'Fibroid_diameter', 'Fibroid_distance', 
#                  'anteverted', 'retroverted', 'vertical']

feature_labels = ['white', 'black', 'asian', 'Age', 'Weight', 'History_of_pregnancy',
                  'Live_births', 'C-section', 'esmya', 'open_myomectomy', 
                  'laprascopic_myomectomy', 'hysteroscopic_myomectomy',
                  'Subcutaneous_fat_thickness', 'Front-back_distance', 'Abdominal_scars',
                  'bleeding', 'pain', 'mass', 'urinary', 'infertility',
                  'Fibroid_diameter', 'Fibroid_distance', 'intramural', 'subserosal', 
                  'submucosal', 'anterior', 'posterior', 'lateral', 'fundus',
                  'anteverted', 'retroverted', 'Type_I', 'Type_II', 'Type_III',
                  'Fibroid_volume', 'ADC']

target_label = ['NPV_class']

#%% randomise and divive data for cross-validation

# stratified splitting for unbalanced datasets

split_ratio = 20
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

t_mean = training_features.mean()
t_std = training_features.std()

training_features = (training_features - t_mean) / t_std
validation_features = (validation_features - t_mean) / t_std
testing_features = (testing_features - t_mean) / t_std

#%% calculate class weights

class_weights = compute_sample_weight('balanced', training_targets)

#%% build and train model

param = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eta': 0.2,
        'max_depth': 3,
        'silent': 1,
        'alpha': 0.0,
        'lambda': 1,
        }

trn = xgb.DMatrix(training_features, label = training_targets, weight = class_weights)
vld = xgb.DMatrix(validation_features, label = validation_targets)

res = xgb.cv(param, trn, nfold = 4, num_boost_round = 2000, early_stopping_rounds = 50,
             stratified = True, show_stdv = True, metrics = {'merror'}, maximize = False)

num_round = 100

evals_result = {}

timestr = time.strftime('%Y%m%d-%H%M%S')

model = xgb.train(param, trn, num_round, [(trn, 'training'), (vld, 'validation')],
                  evals_result = evals_result, verbose_eval = 10)

#%% evaluate model performance

# make predictions

training_predictions = model.predict(trn)
training_predictions = np.argmax(training_predictions, axis = 1)
training_predictions = pd.DataFrame(training_predictions, columns = target_label,
                                    index = training_features.index, dtype = float)

validation_predictions = model.predict(vld)
validation_predictions = np.argmax(validation_predictions, axis = 1)
validation_predictions = pd.DataFrame(validation_predictions, columns = target_label,
                                      index = validation_features.index, dtype = float)

# calculate loss metrics

training_accuracy = accuracy_score(training_targets, training_predictions)
validation_accuracy = accuracy_score(validation_targets, validation_predictions)

# confusion matrix

cm_training = confusion_matrix(training_targets, training_predictions)
cm_training = cm_training.astype('float') / cm_training.sum(axis = 1)[:, np.newaxis]

cm_validation = confusion_matrix(validation_targets, validation_predictions)
cm_validation = cm_validation.astype('float') / cm_validation.sum(axis = 1)[:, np.newaxis]

# plot training performance

f1 = plot_softmax_classification_performance('xgboost', evals_result, cm_training, cm_validation)

# plot feature importance
    
f2 = plot_feature_importance(model, training_features)

#%% save model

model_dir = 'XGBoost models\\%s_TA%d_VA%d' % (timestr, 
                                              round(training_accuracy*100), 
                                              round(validation_accuracy*100))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
f1.savefig(model_dir + '\\' + 'evaluation_metrics.pdf', dpi = 600, format = 'pdf',
                    bbox_inches = 'tight', pad_inches = 0)
f2.savefig(model_dir + '\\' + 'feature_importance.pdf', dpi = 600, format = 'pdf',
                    bbox_inches = 'tight', pad_inches = 0)

model.save_model(model_dir + '\\' + 'xgboost.model')

