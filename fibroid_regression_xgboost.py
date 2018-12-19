# -*- coding: utf-8 -*-
'''
Created on Mon Dec 17 08:49:35 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    December 2018
    
@description:
    
    This code uses XGBoost to predict regression on uterine fibroid non-perfused
    volume (NPV)
    
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
from sklearn.metrics import mean_squared_error
import scipy as sp
import time
import os

from save_load_variables import save_load_variables
from plot_feature_importance import plot_feature_importance
from plot_regression_performance import plot_regression_performance

#%% define logging and data display format

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#%% read data

fibroid_dataframe = pd.read_csv(r'C:\Users\visa\Documents\TYKS\Machine learning\Uterine fibroid\fibroid_dataframe_combined.csv', sep = ',')

#%% calculate nan percent for each label

nan_percent = pd.DataFrame(fibroid_dataframe.isnull().mean() * 100, columns = ['% of NaN'])

#%% replace nan values

#fibroid_dataframe['Height'] = fibroid_dataframe['Height'].fillna(fibroid_dataframe['Height'].mean())
#fibroid_dataframe['Gravidity'] = fibroid_dataframe['Gravidity'].fillna(fibroid_dataframe['Gravidity'].mode()[0])
#fibroid_dataframe['bleeding'] = fibroid_dataframe['bleeding'].fillna(fibroid_dataframe['bleeding'].mode()[0])
#fibroid_dataframe['pain'] = fibroid_dataframe['pain'].fillna(fibroid_dataframe['pain'].mode()[0])
#fibroid_dataframe['mass'] = fibroid_dataframe['mass'].fillna(fibroid_dataframe['mass'].mode()[0])
#fibroid_dataframe['urinary'] = fibroid_dataframe['urinary'].fillna(fibroid_dataframe['urinary'].mode()[0])
#fibroid_dataframe['infertility'] = fibroid_dataframe['infertility'].fillna(fibroid_dataframe['infertility'].mode()[0])
#fibroid_dataframe['ADC'] = fibroid_dataframe['ADC'].fillna(fibroid_dataframe['ADC'].mean())

#%% define feature and target labels

feature_labels = ['Subcutaneous_fat_thickness', 'Abdominal_scars',
                  'Fibroid_diameter', 'Fibroid_distance', 
                  'anteverted', 'retroverted', 'vertical']

#feature_labels = ['white', 'black', 'asian', 'Age', 'Weight', 'History_of_pregnancy',
#                  'Live_births', 'C-section', 'esmya', 'open_myomectomy', 
#                  'laprascopic_myomectomy', 'hysteroscopic_myomectomy',
#                  'Subcutaneous_fat_thickness', 'Front-back_distance', 'Abdominal_scars',
#                  'bleeding', 'pain', 'mass', 'urinary', 'infertility',
#                  'Fibroid_diameter', 'Fibroid_distance', 'intramural', 'subserosal', 
#                  'submucosal', 'anterior', 'posterior', 'lateral', 'fundus',
#                  'anteverted', 'retroverted', 'Type_I', 'Type_II', 'Type_III', 'ADC',
#                  'Fibroid_volume']

target_label = ['NPV_percent']

#%% extract features and targets

features = fibroid_dataframe[feature_labels]
targets = fibroid_dataframe[target_label]

#%% scale features

scaled_features = pd.DataFrame(sp.stats.mstats.zscore(features),
                               columns = list(features), 
                               index = features.index, dtype = float)

#%% scale targets (for skewed data)

target_transform = None

if target_transform == 'log':
    
    targets = np.log1p(targets)

if target_transform == 'box-cox':
    
    lmbda = 0.15    
    
    targets = sp.special.boxcox1p(targets, lmbda)

#%% combine dataframes

concat_dataframe = pd.concat([scaled_features, targets], axis = 1)

#%% randomise and divive data for cross-validation

split_ratio = 40
training_set, holdout_set = train_test_split(concat_dataframe, test_size = split_ratio)
validation_set, testing_set = train_test_split(holdout_set, test_size = int(split_ratio / 2))

#%% define features and targets

training_features = training_set[feature_labels]
validation_features = validation_set[feature_labels]
testing_features = testing_set[feature_labels]

training_targets = training_set[target_label]
validation_targets = validation_set[target_label]
testing_targets = testing_set[target_label]

#%% build and train model

param = {
        'objective': 'reg:linear',
        'eta': 0.02,
        'eval_metric': 'rmse',
        'max_depth': 5,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'silent': 1,
        'seed': 123,
        'alpha': 0.0,
        'labmda': 0.0,
        }

trn = xgb.DMatrix(training_features, label = training_targets, weight = None)
vld = xgb.DMatrix(validation_features, label = validation_targets)

res = xgb.cv(param, trn, nfold = 4, num_boost_round = 2000, early_stopping_rounds = 50,
             show_stdv = True, metrics = {'rmse'}, maximize = False)

min_index = np.argmin(res['test-rmse-mean'])

evals_result = {}

timestr = time.strftime('%Y%m%d-%H%M%S')

model = xgb.train(param, trn, min_index, [(trn, 'training'), (vld, 'validation')],
                  evals_result = evals_result, verbose_eval = 10)

#%% evaluate model performance

# make predictions

training_predictions = model.predict(trn)
training_predictions = pd.DataFrame(training_predictions, columns = target_label,
                                    index = training_features.index, dtype = float)

validation_predictions = model.predict(vld)
validation_predictions = pd.DataFrame(validation_predictions, columns = target_label,
                                      index = validation_features.index, dtype = float)

# calculate loss metrics

training_error = np.sqrt(mean_squared_error(training_targets, training_predictions))
validation_error = np.sqrt(mean_squared_error(validation_targets, validation_predictions))

# convert log targets to linear units (for skewed data)

if target_transform == 'log':

    training_targets_lin = np.expm1(training_targets)
    validation_targets_lin = np.expm1(validation_targets)
    
    training_predictions_lin = np.expm1(training_predictions)
    validation_predictions_lin = np.expm1(validation_predictions)

# convert box-cox targets to linear units (for skewed data)
    
if target_transform == 'box-cox':

    training_targets_lin = sp.special.inv_boxcox1p(training_targets, lmbda)
    validation_targets_lin = sp.special.inv_boxcox1p(validation_targets, lmbda)
    
    training_predictions_lin = sp.special.inv_boxcox1p(training_predictions, lmbda)
    validation_predictions_lin = sp.special.inv_boxcox1p(validation_predictions, lmbda)

# plot training performance
    
if (target_transform == 'log') or (target_transform == 'box-cox'):

    f1 = plot_regression_performance('xgboost', evals_result, training_targets_lin, training_predictions_lin, 
                                     validation_targets_lin, validation_predictions_lin)
else:

    f1 = plot_regression_performance('xgboost', evals_result, training_targets, training_predictions, 
                                     validation_targets, validation_predictions)
    
# plot feature importance
    
f2 = plot_feature_importance(model, training_features)

#%% save model

model_dir = 'XGBoost models\\%s_TE%d_VE%d' % (timestr, 
                                              round(training_error), 
                                              round(validation_error))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
f1.savefig(model_dir + '\\' + 'evaluation_metrics.pdf', dpi = 600, format = 'pdf',
                    bbox_inches = 'tight', pad_inches = 0)
f2.savefig(model_dir + '\\' + 'feature_importance.pdf', dpi = 600, format = 'pdf',
                    bbox_inches = 'tight', pad_inches = 0)

