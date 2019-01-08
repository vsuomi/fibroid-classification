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
import scipy as sp
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import time
import os

from plot_confusion_matrix import plot_confusion_matrix
from plot_feature_importance import plot_feature_importance
from save_load_variables import save_load_variables

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

z_mean = training_features.mean()
z_std = training_features.std()

training_features = (training_features - z_mean) / z_std
testing_features = (testing_features - z_mean) / z_std

#%% calculate class weights

class_weights = compute_class_weight('balanced', np.unique(training_targets), 
                                     training_targets[target_label[0]])

#%% define random state

random_state = np.random.randint(0, 100)

#%% build and train model

# define parameters for parameter search

#parameters =    {
#                'max_depth': [2, 3, 4, 5],
#                'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
#                'n_estimators': [50, 100, 150, 200],
#                'gamma': [0, 0.1, 0.2],
#                'min_child_weight': [0, 0.2, 0.4, 0.6, 0.8, 1],
#                'max_delta_step': [0],
#                'subsample': [0.7, 0.8, 0.9, 1],
#                'colsample_bytree': [1],
#                'colsample_bylevel': [1],
#                'reg_alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                'reg_lambda': [0, 1, 2, 3],
#                'base_score': [0.5]
#                }

# define parameter distributions (for randomised search only)

parameters =    {
                'max_depth': sp.stats.randint(2, 6),
                'learning_rate': sp.stats.uniform(0.05, 0.25),
                'n_estimators': sp.stats.randint(50, 201),
                'gamma': sp.stats.uniform(0, 0.2),
                'min_child_weight': sp.stats.uniform(0, 1),
                'max_delta_step': [0],
                'subsample': sp.stats.uniform(0.7, 0.3),
                'colsample_bytree': [1],
                'colsample_bylevel': [1],
                'reg_alpha': sp.stats.uniform(0, 1),
                'reg_lambda': sp.stats.uniform(0, 3),
                'base_score': [0.5]
                }

# define model

xgb_model = xgb.XGBClassifier(scale_pos_weight = class_weights, silent = True)

# define parameter search method

#clf = GridSearchCV(xgb_model, parameters, scoring = 'f1_micro', 
#                   n_jobs = -1, cv = 5, random_state = random_state)
clf = RandomizedSearchCV(xgb_model, parameters, n_iter = 3000, scoring = 'f1_micro', 
                         n_jobs = -1, cv = 5, random_state = random_state)

# train model using parameter search

timestr = time.strftime('%Y%m%d-%H%M%S')
start_time = time.time()

clf.fit(training_features, training_targets.values[:, 0])

end_time = time.time()

# summarise results

print('Best: %f using %s' % (clf.best_score_, clf.best_params_))
print('Execution time: %.2f s' % (end_time - start_time))

# obtain the best model

model = clf.best_estimator_

#%% evaluate model performance

# make predictions

training_predictions = model.predict(training_features)
training_predictions = pd.DataFrame(training_predictions, columns = target_label,
                                    index = training_features.index, dtype = float)

# calculate loss metrics

training_accuracy = accuracy_score(training_targets, training_predictions)

# confusion matrix

cm_training = confusion_matrix(training_targets, training_predictions)
cm_training = cm_training.astype('float') / cm_training.sum(axis = 1)[:, np.newaxis]

# plot training performance

f1 = plot_confusion_matrix(cm_training)

# plot feature importance
    
f2 = plot_feature_importance(model, training_features)

#%% save model

model_dir = 'XGBoost models\\%s_TA%d' % (timestr, 
                                         round(training_accuracy*100))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
f1.savefig(model_dir + '\\' + 'evaluation_metrics.pdf', dpi = 600, format = 'pdf',
                    bbox_inches = 'tight', pad_inches = 0)
f2.savefig(model_dir + '\\' + 'feature_importance.pdf', dpi = 600, format = 'pdf',
                    bbox_inches = 'tight', pad_inches = 0)

variables_to_save = {'parameters': parameters,
                     'clf': clf,
                     'random_state': random_state,
                     'class_weights': class_weights,
                     'NPV_bins': NPV_bins,
                     'split_ratio': split_ratio,
                     'timestr': timestr,
                     'scaling_type': scaling_type,
                     'z_mean': z_mean,
                     'z_std': z_std,
                     'model_dir': model_dir,
                     'fibroid_dataframe': fibroid_dataframe,
                     'training_set': training_set,
                     'training_features': training_features,
                     'training_targets': training_targets,
                     'testing_set': testing_set,
                     'testing_features': testing_features,
                     'testing_targets': testing_targets,
                     'feature_labels': feature_labels,
                     'target_label': target_label}
    
save_load_variables(model_dir, variables_to_save, 'variables', 'save')

model.save_model(model_dir + '\\' + 'xgboost.model')
