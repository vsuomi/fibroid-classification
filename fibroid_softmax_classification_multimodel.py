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

import pandas as pd
import numpy as np
import scipy as sp
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from logitboost import LogitBoost
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import time
import os

from EstimatorSelectionHelper import EstimatorSelectionHelper

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
#                  'Fibroid_volume', 'ADC']

target_label = ['NPV_class']

#%% randomise and divive data for cross-validation

# stratified splitting for unbalanced datasets

split_ratio = 0.2
training_set, testing_set = train_test_split(fibroid_dataframe, test_size = split_ratio,
                                             stratify = fibroid_dataframe[target_label])

#%% define features and targets

training_features = training_set[feature_labels]
testing_features = testing_set[feature_labels]

training_targets = training_set[target_label]
testing_targets = testing_set[target_label]

#%% scale features

scaling_type = 'z-score'

if scaling_type == 'z-score':

    z_mean = training_features.mean()
    z_std = training_features.std()
    
    training_features = (training_features - z_mean) / z_std
    testing_features = (testing_features - z_mean) / z_std

#%% calculate class weights

class_weights = compute_class_weight('balanced', np.unique(training_targets), 
                                     training_targets[target_label[0]])
class_weights = dict(enumerate(class_weights))

#%% define random state

random_state = np.random.randint(0, 1000)

#%% define models and parameters

# define models

models =    {
            'ExtraTreesClassifier': ExtraTreesClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(),
            'LogitBoost': LogitBoost()
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
                        'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
                        'subsample': [0.8, 0.9, 1],
                        'min_samples_split': [2, 4],
                        'max_features': ['sqrt', None],
                        'random_state': [random_state]
                        }

param_svc =             [
                        {
                        'kernel': ['rbf'], 
                        'C': [0.01, 0.1, 1, 10, 100],
                        'gamma': ['auto', 'scale'],
                        'random_state': [random_state],
                        'class_weight': [class_weights]
                        },
                        {
                        'kernel': ['linear'], 
                        'C': [0.01, 0.1, 1, 10, 100],
                        'random_state': [random_state],
                        'class_weight': [class_weights]
                        }
                        ]

param_logitboost  =     {
                        'n_estimators': [10, 50, 100, 200, 300],
                        'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
                        'random_state': [random_state]
                        }
# combine parameters

parameters =    {
                'ExtraTreesClassifier': param_extra_trees,
                'RandomForestClassifier': param_random_forest,
                'AdaBoostClassifier': param_adaboost,
                'GradientBoostingClassifier': param_gradient_boost,
                'SVC': param_svc,
                'LogitBoost': param_logitboost
                }

#%% perform cross-validation and parameter search

scoring = 'f1_micro'
cv = 10

clf = EstimatorSelectionHelper(models, parameters)

timestr = time.strftime('%Y%m%d-%H%M%S')
start_time = time.time()

clf.fit(training_features, training_targets, scoring = scoring, 
        n_jobs = -1, cv = cv, refit = True)

end_time = time.time()

print('Execution time: %.2f s' % (end_time - start_time))

#%% display score summary

clf_scores = clf.score_summary(sort_by = 'mean_score')