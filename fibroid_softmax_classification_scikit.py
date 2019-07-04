# -*- coding: utf-8 -*-
'''
Created on Wed Mar 20 15:45:35 2019

@author:
    
    Visa Suomi
    Turku University Hospital
    March 2019
    
@description:
    
    This code is used to classify (using softmax) the treatability of uterine 
    fibroids before HIFU therapy based on their pre-treatment parameters using
    scikit-learn
    
'''

#%% clear variables

%reset -f
%clear

#%% import necessary libraries

import os
import time
import pickle
import joblib
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, make_scorer
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.metrics import geometric_mean_score

# import classifiers

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier

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

#%% display NPV histogram

df['NPV ratio'].hist(bins = 20)

#%% categorise NPV into classes according to bins

NPV_bins = [-1, 29.9, 80, 100]
df['NPV class'] = df[['NPV ratio']].apply(lambda x: pd.cut(x, NPV_bins, labels = False))

#%% calculate data statistics

df_stats = pd.DataFrame(df.isnull().mean() * 100, columns = ['NaN ratio'])
df_stats['Mean'] = df.mean()
df_stats['Median'] = df.median()
df_stats['Min'] = df.min()
df_stats['Max'] = df.max()
df_stats['SD'] = df.std()
df_stats['Sum'] = df.sum()

#%% display NPV histogram

df['NPV ratio'].hist(bins = 20)

#%% define feature and target labels

feature_labels = [#'White', 
                  #'Black', 
                  #'Asian', 
                  #'Age', 
                  #'Weight', 
                  #'Height', 
                  #'Gravidity', 
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
                  #'Front-back distance', 
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
                  #'Fundus',
                  #'Anteverted', 
                  #'Retroverted', 
                  'Type I', 
                  'Type II', 
                  'Type III',
                  #'ADC',
                  #'Fibroid volume'
                  ]

target_label = ['NPV class']

#%% define models and parameters

# define split ratio for training and testing sets

split_ratio = 0.2

# impute features

impute_mean =   []
impute_mode =   []
impute_cons =   []

# define oversampling strategy ('random', 'smote', 'adasyn' or None)

oversample = 'adasyn'

# discretise features

#discretise = []
discretise =    [#'Age', 
                 #'Weight', 
                 #'Height',
                 'Subcutaneous fat thickness', 
                 #'Front-back distance',
                 'Fibroid diameter', 
                 'Fibroid distance'
                 ]

# define scaling type ('log', 'minmax', 'standard' or None)

scaling_type = 'log'

# define the number of cross-validations for grid search

cv = 10

# define scoring metric ('f1_*', 'balanced_accuracy' or custom scorer)

#scoring = 'f1_micro'
scoring = make_scorer(geometric_mean_score, average = 'multiclass')

#%% randomise and divive data for cross-validation

# stratified splitting for unbalanced datasets

training_set, testing_set = train_test_split(df, test_size = split_ratio,
                                             stratify = df[target_label],
                                             random_state = random_state)

#%% define features and targets

training_features = training_set[feature_labels]
testing_features = testing_set[feature_labels]

training_targets = training_set[target_label]
testing_targets = testing_set[target_label]

#%% impute features

if impute_mean:
    
    imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    
    training_features[impute_mean] = imp_mean.fit_transform(training_features[impute_mean])
    testing_features[impute_mean] = imp_mean.transform(testing_features[impute_mean])
    
if impute_mode:
    
    imp_mode = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    
    training_features[impute_mode] = imp_mode.fit_transform(training_features[impute_mode])
    testing_features[impute_mode] = imp_mode.transform(testing_features[impute_mode])
    
if impute_cons:
    
    imp_cons = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 0)
    
    training_features[impute_cons] = imp_cons.fit_transform(training_features[impute_cons])
    testing_features[impute_cons] = imp_cons.transform(testing_features[impute_cons])
    
#%% oversample imbalanced training data

if oversample == 'random':
    
    osm = RandomOverSampler(sampling_strategy = 'not majority', random_state = random_state)
    training_features, training_targets = osm.fit_resample(training_features.values, training_targets.values[:, 0])
    
    training_features = pd.DataFrame(training_features, columns = testing_features.columns)
    training_targets = pd.DataFrame(training_targets, columns = testing_targets.columns)
    
elif oversample == 'smote':
    
    osm = SMOTE(sampling_strategy = 'not majority', random_state = random_state, n_jobs = -1)
    training_features, training_targets = osm.fit_resample(training_features.values, training_targets.values[:, 0])
    
    training_features = pd.DataFrame(training_features, columns = testing_features.columns)
    training_targets = pd.DataFrame(training_targets, columns = testing_targets.columns)
    
elif oversample == 'adasyn':
    
    osm = ADASYN(sampling_strategy = 'not majority', random_state = random_state, n_jobs = -1)
    training_features, training_targets = osm.fit_resample(training_features.values, training_targets.values[:, 0])
    
    training_features = pd.DataFrame(training_features, columns = testing_features.columns)
    training_targets = pd.DataFrame(training_targets, columns = testing_targets.columns)

#%% discretise features

if discretise:
    
    enc = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'uniform')
    
    training_features[discretise] = enc.fit_transform(training_features[discretise])
    testing_features[discretise] = enc.transform(testing_features[discretise])

#%% scale features

if scaling_type == 'log':
        
    training_features = np.log1p(training_features)
    testing_features = np.log1p(testing_features)
    
elif scaling_type == 'minmax':
    
    scaler = MinMaxScaler(feature_range = (0, 1)) 
    training_features[feature_labels] = scaler.fit_transform(training_features[feature_labels])
    testing_features[feature_labels] = scaler.transform(testing_features[feature_labels])
    
elif scaling_type == 'standard':
    
    scaler = StandardScaler() 
    training_features[feature_labels] = scaler.fit_transform(training_features[feature_labels])
    testing_features[feature_labels] = scaler.transform(testing_features[feature_labels])

#%% build and train model
    
# define model and parameters for randomised search
    
# Support Vector Classifier

parameters =    {
                'kernel': ['rbf'],
                'C': sp.stats.reciprocal(1e-1, 1e4),
                'gamma': sp.stats.reciprocal(1e-2, 1e4)
                }

base_model = SVC(class_weight = 'balanced', random_state = random_state,
                 cache_size = 4000, max_iter = 200000, probability = True)

# Complement Naive-Bayes
   
#parameters =    {
#                'alpha': sp.stats.reciprocal(1e-2, 1e5),
#                'norm': [True, False]
#                }
#
#base_model = ComplementNB()

# define parameter search method

#grid = GridSearchCV(base_model, parameters, scoring = scoring, 
#                    n_jobs = -1, cv = cv, refit = True, iid = False)
grid = RandomizedSearchCV(base_model, parameters, scoring = scoring, 
                          n_jobs = -1, cv = cv, refit = True, iid = False,
                          n_iter = 10000, random_state = random_state)

# train model using parameter search

timestr = time.strftime('%Y%m%d-%H%M%S')
start_time = time.time()

grid.fit(training_features.values, training_targets.values[:, 0])

end_time = time.time()

# summarise results

print('Best score %f using parameters: %s' % (grid.best_score_, grid.best_params_))
print('Execution time: %.2f s' % (end_time - start_time))

# obtain the best model and score

best_model = grid.best_estimator_

#%% evaluate model performance

# make predictions

training_predictions = best_model.predict(training_features.values)
testing_predictions = best_model.predict(testing_features.values)

# calculate evaluation metrics

validation_score = grid.best_score_

if type(scoring) == str and scoring[:2] == 'f1':
                
    training_score = f1_score(training_targets.values[:, 0], training_predictions, average = scoring[3:])
    testing_score = f1_score(testing_targets.values[:, 0], testing_predictions, average = scoring[3:])
    
elif type(scoring) == str and scoring == 'balanced_accuracy':
    
    training_score = balanced_accuracy_score(training_targets.values[:, 0], training_predictions)
    testing_score = balanced_accuracy_score(testing_targets.values[:, 0], testing_predictions)
    
else:
    
    training_score = scoring(best_model, training_features.values, training_targets.values[:, 0])
    testing_score = scoring(best_model, testing_features.values, testing_targets.values[:, 0])

# calculate confusion matrices

cm_training = confusion_matrix(training_targets, training_predictions)
cm_training = cm_training.astype('float') / cm_training.sum(axis = 1)[:, np.newaxis]

cm_testing = confusion_matrix(testing_targets, testing_predictions)
cm_testing = cm_testing.astype('float') / cm_testing.sum(axis = 1)[:, np.newaxis]

#%% plot figures

# confusion matrices

f1 = plt.figure(figsize = (6, 4))
ax = sns.heatmap(cm_training, cmap = 'Greys', vmin = 0, vmax = 1,
                 cbar_kws = {'ticks': [0, 0.5, 1]})
ax.set_aspect(1)
#    plt.title('Training')
plt.ylabel('True class')
plt.xlabel('Predicted class')

f2 = plt.figure(figsize = (6, 4))
ax = sns.heatmap(cm_testing, cmap = 'Greys', vmin = 0, vmax = 1,
                 cbar_kws = {'ticks': [0, 0.5, 1]})
ax.set_aspect(1)
#    plt.title('Testing')
plt.ylabel('True class')
plt.xlabel('Predicted class')

#%% save data

# make directory

model_dir = os.path.join('Scikit models', 
                         ('%s_TS%d_VS%d_TS%d' % (timestr, 
                                                 round(training_score*100),
                                                 round(validation_score*100),
                                                 round(testing_score*100))))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
# save parameters into text file
    
with open(os.path.join(model_dir, 'parameters.txt'), 'w') as text_file:
    text_file.write('timestr: %s\n' % timestr)
    text_file.write('Computation time: %.1f min\n' % ((end_time - start_time) / 60))
    text_file.write('Number of samples: %d\n' % len(df))
    text_file.write('Number of features: %d\n' % len(feature_labels))
    text_file.write('Training set size: %d\n' % len(training_targets))
    text_file.write('Testing set size: %d\n' % len(testing_targets))
    text_file.write('feature_labels: %s\n' % str(feature_labels))
    text_file.write('target_label: %s\n' % str(target_label))
    text_file.write('duplicates: %s\n' % str(duplicates))
    text_file.write('oversample: %s\n' % str(oversample))
    text_file.write('discretise: %s\n' % str(discretise))
    text_file.write('impute_mean: %s\n' % str(impute_mean))
    text_file.write('impute_mode: %s\n' % str(impute_mode))
    text_file.write('impute_cons: %s\n' % str(impute_cons))
    text_file.write('scaling_type: %s\n' % str(scaling_type))
    text_file.write('scoring: %s\n' % scoring)
    text_file.write('split_ratio: %.1f\n' % split_ratio)
    text_file.write('cv: %d\n' % cv)
    
# save figures
    
for filetype in ['pdf', 'png', 'eps']:
    
    f1.savefig(os.path.join(model_dir, ('cm_training.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f2.savefig(os.path.join(model_dir, ('cm_testing.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    
# save variables
    
variable_names = %who_ls DataFrame ndarray list dict str bool int int64 float float64
variables = dict((name, eval(name)) for name in variable_names)

pickle.dump(variables, open(os.path.join(model_dir, 'variables.pkl'), 'wb'))

# save model

pickle.dump(best_model, open(os.path.join(model_dir, 'scikit_model.pkl'), 'wb'))

# save grid

joblib.dump(grid, os.path.join(model_dir, 'grid.joblib'))

# save data pre-processing functions

if impute_mean:
    
    joblib.dump(imp_mean, os.path.join(model_dir, 'imputer_mean.joblib'))
    
if impute_mode:
    
    joblib.dump(imp_mode, os.path.join(model_dir, 'imputer_mode.joblib'))
    
if impute_cons:
    
    joblib.dump(imp_cons, os.path.join(model_dir, 'imputer_cons.joblib'))
    
if oversample is not None:
    
    joblib.dump(osm, os.path.join(model_dir, 'oversampler.joblib'))
    
if discretise:
    
    joblib.dump(enc, os.path.join(model_dir, 'discretiser.joblib'))
    
if scaling_type == 'minmax' or scaling_type == 'standard':
    
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
