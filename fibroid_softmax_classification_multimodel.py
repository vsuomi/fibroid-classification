# -*- coding: utf-8 -*-
'''
Created on Thu Jan 10 13:05:04 2019

@author:
    
    Visa Suomi
    Turku University Hospital
    January 2019
    
@description:
    
    This code is used to train multiple classification models at the same time
    and evaluate their performance using EstimatorSelectionHelper class
    
'''

#%% clear variables

%reset -f
%clear

#%% import necessary libraries

import os
import time
import pandas as pd
import numpy as np
import scipy as sp
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from logitboost import LogitBoost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer

from EstimatorSelectionHelper import EstimatorSelectionHelper

#%% define random state

random_state = np.random.randint(0, 10000)

#%% define logging and data display format

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
pd.options.mode.chained_assignment = None                                       # disable imputation warnings

#%% read data

df = pd.read_csv('fibroid_dataframe.csv', sep = ',')

#%% check for duplicates

duplicates = any(df.duplicated())

#%% categorise NPV into classes according to bins

NPV_bins = [-1, 29.9, 80, 100]
df['NPV class'] = df[['NPV ratio']].apply(lambda x: pd.cut(x, NPV_bins, labels = False))

#%% calculate data quality

df_quality = pd.DataFrame(df.isnull().mean() * 100, columns = ['NaN ratio'])
df_quality['Mean'] = df.mean()
df_quality['Median'] = df.median()
df_quality['SD'] = df.std()
df_quality['Sum'] = df.sum()

#%% display NPV histogram

df['NPV ratio'].hist(bins = 20)

#%% define feature and target labels

feature_labels = [#'White', 
                  #'Black', 
                  #'Asian', 
                  'Age', 
                  'Weight', 
                  'Height', 
                  'Gravidity', 
                  #'Parity',
                  #'Previous pregnancies', 
                  #'Live births', 
                  #'C-section', 
                  #'Esmya', 
                  #'Open myomectomy', 
                  #'Laparoscopic myomectomy', 
                  #'Hysteroscopic myomectomy',
                  #'Embolisation', 
                  'Subcutaneous fat thickness', 
                  'Front-back distance', 
                  #'Abdominal scars', 
                  #'Bleeding', 
                  #'Pain', 
                  #'Mass', 
                  #'Urinary', 
                  #'Infertility',
                  'Fibroid diameter', 
                  'Fibroid distance', 
                  #'Intramural', 
                  #'Subserosal', 
                  #'Submucosal', 
                  #'Anterior', 
                  #'Posterior', 
                  #'Lateral', 
                  'Fundus',
                  #'Anteverted', 
                  #'Retroverted', 
                  #'Type I', 
                  #'Type II', 
                  'Type III',
                  #'ADC',
                  'Fibroid volume'
                  ]

target_label = ['NPV class']

#%% randomise and divive data for cross-validation

# stratified splitting for unbalanced datasets

split_ratio = 0.2
training_set, testing_set = train_test_split(df, test_size = split_ratio,
                                             stratify = df[target_label],
                                             random_state = random_state)

#%% impute data

impute_labels = ['Height', 
                 'Gravidity'
                 ]

impute_values = {}

for label in impute_labels:
        
    if label in {'Height', 'ADC'}:
        
        impute_values[label] = training_set[label].mean()
        
        training_set[label] = training_set[label].fillna(impute_values[label])
        testing_set[label] = testing_set[label].fillna(impute_values[label])
        
    else:
        
        impute_values[label] = training_set[label].mode()[0]
        
        training_set[label] = training_set[label].fillna(impute_values[label])
        testing_set[label] = testing_set[label].fillna(impute_values[label])
        
del label

#%% define features and targets

training_features = training_set[feature_labels]
testing_features = testing_set[feature_labels]

training_targets = training_set[target_label]
testing_targets = testing_set[target_label]

#%% discretise features

discretise = False

if discretise:
    
    disc_labels =   ['Weight',
                     'Height', 
                     'Subcutaneous fat thickness', 
                     'Front-back distance',
                     'Fibroid diameter', 
                     'Fibroid distance',
                     #'Fibroid volume',
                     #'ADC'
                     ]
    
    enc = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'uniform')
    
    bin_training = enc.fit_transform(training_features[disc_labels])
    bin_testing = enc.transform(testing_features[disc_labels])
    
    training_features[disc_labels] = bin_training
    testing_features[disc_labels] = bin_testing
    
    disc_bins = enc.n_bins_
    disc_edges = enc.bin_edges_

#%% scale features

scaling_type = 'log'

if scaling_type == 'log':
        
    training_features = np.log1p(training_features)
    testing_features = np.log1p(testing_features)
    
elif scaling_type == 'minmax':
    
    scaler = MinMaxScaler(feature_range = (0, 1)) 
    training_features = pd.DataFrame(scaler.fit_transform(training_features),
                                     columns = training_features.columns,
                                     index = training_features.index)
    testing_features = pd.DataFrame(scaler.transform(testing_features),
                                    columns = testing_features.columns,
                                    index = testing_features.index)
    
elif scaling_type == 'standard':
    
    scaler = StandardScaler() 
    training_features = pd.DataFrame(scaler.fit_transform(training_features),
                                     columns = training_features.columns,
                                     index = training_features.index)
    testing_features = pd.DataFrame(scaler.transform(testing_features),
                                    columns = testing_features.columns,
                                    index = testing_features.index)

#%% calculate class weights

class_weights = compute_class_weight('balanced', np.unique(training_targets), 
                                     training_targets[target_label[0]])
class_weights = dict(enumerate(class_weights))

#%% define models and parameters

# define models

models =    {
            'ExtraTreesClassifier': ExtraTreesClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(),
            'LogitBoost': LogitBoost(),
            'XGBClassifier': XGBClassifier(),
            'ComplementNB': ComplementNB()
            }

# define model parameters for parameter search

param_extra_trees =     {
                        'n_estimators': [10, 50, 100, 200, 300],
                        'min_samples_split': [2, 4],
                        'max_features': ['sqrt', None],
                        'random_state': [random_state],
                        'class_weight': [class_weights]
                        }

param_random_forest =   {
                        'n_estimators': [10, 50, 100, 200, 300],
                        'min_samples_split': [2, 4],
                        'max_features': ['sqrt', None],
                        'random_state': [random_state],
                        'class_weight': [class_weights]
                        }

param_adaboost =        {
                        'n_estimators': [10, 50, 100, 200, 300],
                        'learning_rate': [0.1, 0.5, 1, 5, 10],
                        'random_state': [random_state]
                        }

param_gradient_boost =  {
                        'n_estimators': [10, 50, 100, 200, 300],
                        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5, 1],
                        'subsample': [0.8, 0.9, 1],
                        'min_samples_split': [2, 4],
                        'max_features': ['sqrt', None],
                        'random_state': [random_state]
                        }

param_svc =             [
                        {
                        'kernel': ['rbf'], 
                        'C': [0.005, 0.01, 0.1, 1, 10, 100],
                        'gamma': ['auto', 'scale'],
                        'random_state': [random_state],
                        'class_weight': [class_weights]
                        },
                        {
                        'kernel': ['linear'], 
                        'C': [0.005, 0.01, 0.1, 1, 10, 100],
                        'random_state': [random_state],
                        'class_weight': [class_weights]
                        }
                        ]

param_logitboost  =     {
                        'n_estimators': [10, 50, 100, 200, 300],
                        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5, 1],
                        'random_state': [random_state]
                        }

param_xgb =             {
                        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5, 1],
                        'n_estimators': [10, 50, 100, 200, 300],
                        'subsample': [0.8, 0.9, 1],
                        'reg_alpha': [0, 1, 5, 10],
                        'reg_lambda': [0, 1, 5, 10],
                        'random_state': [random_state]
                        }

param_complementnb =    {
                        'alpha': [0.01, 0.1, 1, 10, 100],
                        'norm': [True, False]
                        }

# combine parameters

parameters =    {
                'ExtraTreesClassifier': param_extra_trees,
                'RandomForestClassifier': param_random_forest,
                'AdaBoostClassifier': param_adaboost,
                'GradientBoostingClassifier': param_gradient_boost,
                'SVC': param_svc,
                'LogitBoost': param_logitboost,
                'XGBClassifier': param_xgb,
                'ComplementNB': param_complementnb
                }

#%% perform cross-validation and parameter search

cv = 10
scoring = 'f1_micro'

clf = EstimatorSelectionHelper(models, parameters)

timestr = time.strftime('%Y%m%d-%H%M%S')
start_time = time.time()

clf.fit(training_features, training_targets, scoring = scoring, 
        n_jobs = -1, cv = cv, refit = False)

end_time = time.time()

print('Execution time: %.2f s' % (end_time - start_time))

#%% display score summary

clf_scores = clf.score_summary(sort_by = 'mean_score')
