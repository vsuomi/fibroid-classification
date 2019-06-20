# -*- coding: utf-8 -*-
'''
Created on Tue May 21 08:15:59 2019

@author:
    
    Visa Suomi
    Turku University Hospital
    May 2019
    
@description:
    
    This code is used for comparing different machine learning models for 
    predicting the treatment outcome of high-intensity focused ultrasound
    therapy in uterine fibroids
    
'''

#%% clear variables

%reset -f
%clear

#%% import necessary libraries

import os
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('Agg')  # only for cluster use
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, balanced_accuracy_score
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

# import classifiers

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from logitboost import LogitBoost
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier

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

#%% define feature and target labels

feature_labels = ['Fibroid diameter',
                  'Fibroid distance',
                  'Front-back distance',
                  'Subcutaneous fat thickness',
                  'Weight',
                  'Age',  
                  'Height', 
                  'Gravidity',  
                  'Type III',
                  'Fundus'
                  ]

target_label = ['NPV class']

#%% define models and parameters

# define number of iterations

n_iterations = 1

# define number of features

n_features = [2, 4, 6, 8, 10]

# define split ratio for training and testing sets

split_ratio = 0.2

# impute features

impute_mean =   ['Height']
impute_mode =   ['Gravidity']
impute_cons =   []

# define oversampling strategy ('random', 'smote', 'adasyn' or None)

oversample = 'random'

# discretise features

discretise =    ['Age', 
                 'Weight', 
                 'Height',
                 'Subcutaneous fat thickness', 
                 'Front-back distance',
                 'Fibroid diameter', 
                 'Fibroid distance'
                 ]

# define scaling type ('log', 'minmax', 'standard' or None)

scaling_type = 'log'

# define the number of cross-validations for grid search

cv = 10

# define scoring metric ('f1_*' or 'balanced_accuracy')

scoring = 'f1_micro'

# initialise variables

clf_results = pd.DataFrame()

# define models

models =    {
            'ExtraTrees': ExtraTreesClassifier(),
            'RandomForest': RandomForestClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'GradientBoosting': GradientBoostingClassifier(),
            'SVC': SVC(),
            'LogitBoost': LogitBoost(),
            'XGBClassifier': XGBClassifier(),
            'ComplementNB': ComplementNB(),
            'BalancedBagging': BalancedBaggingClassifier(),
            'BalancedRandomForest': BalancedRandomForestClassifier(),
            'RUSBoost': RUSBoostClassifier(),
            'EasyEnsemble': EasyEnsembleClassifier()
            }

# define model parameters for parameter search

param_extra_trees =     {
                        'n_estimators': [5, 10, 50, 100, 200],
                        'min_samples_split': [2, 4],
                        'max_depth': [2, 3, None],
                        'max_features': ['sqrt', None],
                        'class_weight': ['balanced']
                        }

param_random_forest =   {
                        'n_estimators': [5, 10, 50, 100, 200],
                        'min_samples_split': [2, 4],
                        'max_depth': [2, 3, None],
                        'max_features': ['sqrt', None],
                        'class_weight': ['balanced']
                        }

param_adaboost =        {
                        'n_estimators': [5, 10, 50, 100, 200, 300],
                        'learning_rate': [0.001, 0.01, 0.1, 1, 10]
                        }

param_gradient_boost =  {
                        'n_estimators': [5, 10, 50, 100, 200],
                        'learning_rate': [0.001, 0.01, 0.1, 1],
                        'subsample': [0.8, 0.9, 1],
                        'min_samples_split': [2, 4],
                        'max_depth': [2, 3, None],
                        'max_features': ['sqrt', None]
                        }

param_svc =             {
                        'kernel': ['rbf'],
                        'C': [0.1, 1.0, 10, 100],
                        'gamma': [1.0, 10, 100, 1000],
                        'class_weight': ['balanced']
                        }                    

param_logitboost  =     {
                        'n_estimators': [5, 10, 50, 100, 200],
                        'learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.1]
                        }

param_xgb =             {
                        'n_estimators': [5, 10, 50, 100, 200],
                        'learning_rate': [0.001, 0.01, 0.1, 1],
                        'max_depth': [2, 3],
                        'subsample': [0.8, 0.9, 1],
                        'reg_alpha': [0, 0.1, 1, 10],
                        'reg_lambda': [0, 0.1, 1, 10]
                        }

param_complementnb =    {
                        'alpha': [0.1, 1, 10, 100, 1000, 10000],
                        'norm': [True, False]
                        }

param_balanced_bagging =            {
                                    'n_estimators': [5, 10, 50, 100, 200],
                                    'max_samples': [0.8, 0.9, 1.0],
                                    'max_features': [0.8, 0.9, 1.0],
                                    'n_jobs': [-1]
                                    }

param_balanced_random_forest =      {
                                    'n_estimators': [5, 10, 50, 100, 200],
                                    'min_samples_split': [2, 4],
                                    'max_depth': [2, 3, None],
                                    'max_features': ['sqrt', None],
                                    'class_weight': ['balanced'],
                                    'n_jobs': [-1]
                                    }

param_rusboost =                    {
                                    'n_estimators': [5, 10, 50, 100, 200, 300],
                                    'learning_rate': [0.001, 0.01, 0.1, 1, 10]
                                    }

param_easy_ensemble =               {
                                    'n_estimators': [5, 10, 50, 100, 200, 300],
                                    'n_jobs': [-1]
                                    }

# combine parameters

parameters =    {
                'ExtraTrees': param_extra_trees,
                'RandomForest': param_random_forest,
                'AdaBoost': param_adaboost,
                'GradientBoosting': param_gradient_boost,
                'SVC': param_svc,
                'LogitBoost': param_logitboost,
                'XGBClassifier': param_xgb,
                'ComplementNB': param_complementnb,
                'BalancedBagging': param_balanced_bagging,
                'BalancedRandomForest': param_balanced_random_forest,
                'RUSBoost': param_rusboost,
                'EasyEnsemble': param_easy_ensemble
                }

#%% start the iteration

timestr = time.strftime('%Y%m%d-%H%M%S')
start_time = time.time()

for iteration in range(0, n_iterations):
    
    # define random state

    random_state = np.random.randint(0, 10000)
    
    # print progress
    
    print('Iteration %d with random state %d at %.1f min' % (iteration, random_state, 
                                                             ((time.time() - start_time) / 60)))
    
    # randomise and divive data for cross-validation
    
    training_set, testing_set = train_test_split(df, test_size = split_ratio,
                                                 stratify = df[target_label],
                                                 random_state = random_state)
    
    # define features and targets
    
    training_features = training_set[feature_labels]
    testing_features = testing_set[feature_labels]
    
    training_targets = training_set[target_label]
    testing_targets = testing_set[target_label]
    
    # impute features
    
    if impute_mean:
        
        imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        
        training_features[impute_mean] = imp.fit_transform(training_features[impute_mean])
        testing_features[impute_mean] = imp.transform(testing_features[impute_mean])
        
        del imp
        
    if impute_mode:
        
        imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
        
        training_features[impute_mode] = imp.fit_transform(training_features[impute_mode])
        testing_features[impute_mode] = imp.transform(testing_features[impute_mode])
        
        del imp
        
    if impute_cons:
        
        imp = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 0)
        
        training_features[impute_cons] = imp.fit_transform(training_features[impute_cons])
        testing_features[impute_cons] = imp.transform(testing_features[impute_cons])
        
        del imp
        
    # oversample imbalanced training data
    
    if oversample == 'random':
        
        osm = RandomOverSampler(sampling_strategy = 'not majority', random_state = random_state)
        training_features, training_targets = osm.fit_resample(training_features.values, training_targets.values[:, 0])
        
        training_features = pd.DataFrame(training_features, columns = testing_features.columns)
        training_targets = pd.DataFrame(training_targets, columns = testing_targets.columns)
        
        del osm
        
    elif oversample == 'smote':
        
        osm = SMOTE(sampling_strategy = 'not majority', random_state = random_state, n_jobs = -1)
        training_features, training_targets = osm.fit_resample(training_features.values, training_targets.values[:, 0])
        
        training_features = pd.DataFrame(training_features, columns = testing_features.columns)
        training_targets = pd.DataFrame(training_targets, columns = testing_targets.columns)
        
        del osm
        
    elif oversample == 'adasyn':
        
        osm = ADASYN(sampling_strategy = 'not majority', random_state = random_state, n_jobs = -1)
        training_features, training_targets = osm.fit_resample(training_features.values, training_targets.values[:, 0])
        
        training_features = pd.DataFrame(training_features, columns = testing_features.columns)
        training_targets = pd.DataFrame(training_targets, columns = testing_targets.columns)
        
        del osm
    
    # discretise features
    
    if discretise:
    
        enc = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'uniform')
        
        training_features[discretise] = enc.fit_transform(training_features[discretise])
        testing_features[discretise] = enc.transform(testing_features[discretise])
        
        del enc
    
    # scale features
       
    if scaling_type == 'log':
        
        training_features = np.log1p(training_features)
        testing_features = np.log1p(testing_features)
        
    elif scaling_type == 'minmax':
        
        scaler = MinMaxScaler(feature_range = (0, 1)) 
        training_features[feature_labels] = scaler.fit_transform(training_features[feature_labels])
        testing_features[feature_labels] = scaler.transform(testing_features[feature_labels])
        
        del scaler
        
    elif scaling_type == 'standard':
        
        scaler = StandardScaler() 
        training_features[feature_labels] = scaler.fit_transform(training_features[feature_labels])
        testing_features[feature_labels] = scaler.transform(testing_features[feature_labels])
        
        del scaler
    
    for n in n_features:    
        for model in models:
            
            # obtain grid parameters and model
            
            clf_model = models.get(model)
            grid_param = parameters.get(model)
            
            # define parameter search method
            
            clf_grid = GridSearchCV(clf_model, grid_param, n_jobs = -1, cv = cv, 
                                    scoring = scoring, refit = True, iid = False)
            
            # fit parameter search
        
            clf_fit = clf_grid.fit(training_features[feature_labels[0:n]].values, training_targets.values[:, 0])
            
            # calculate predictions
            
            testing_predictions = clf_fit.predict(testing_features[feature_labels[0:n]].values)
            
            # calculate test score
            
            if scoring[:2] == 'f1':
                
                test_score = f1_score(testing_targets.values[:, 0], testing_predictions, average = scoring[3:])
                
            elif scoring == 'balanced_accuracy':
                
                test_score = balanced_accuracy_score(testing_targets.values[:, 0], testing_predictions)
            
            # save results
            
            res = pd.DataFrame(clf_fit.best_params_, index = [0])
            res['model'] = model
            res['validation_score'] = clf_fit.best_score_
            res['test_score'] = test_score
            res['n_features'] = n
            res['iteration'] = iteration
            res['random_state'] = random_state
            clf_results = clf_results.append(res, sort = True, ignore_index = True)
            
            del clf_model, grid_param, clf_grid, clf_fit, testing_predictions, test_score, res
                
    del n, model, random_state
    del training_set, training_features, training_targets
    del testing_set, testing_features, testing_targets
    
del iteration
        
end_time = time.time()

print('Total execution time: %.1f min' % ((end_time - start_time) / 60))

#%% calculate summaries

# summarise results

mean_vscores = clf_results.groupby(['model', 'n_features'], as_index = False)['validation_score'].mean()
mean_tscores = clf_results.groupby(['model', 'n_features'])['test_score'].mean().values

std_vscores = clf_results.groupby(['model', 'n_features'])['validation_score'].std().values
std_tscores = clf_results.groupby(['model', 'n_features'])['test_score'].std().values

clf_summary = mean_vscores.copy()
clf_summary['test_score'] = mean_tscores
clf_summary['validation_score_std'] =  std_vscores
clf_summary['test_score_std'] = std_tscores

del mean_vscores, mean_tscores, std_vscores, std_tscores

# calculate heatmaps for validation and test scores
    
heatmap_vscore_mean = clf_summary.pivot(index = 'model', columns = 'n_features', values = 'validation_score')
heatmap_vscore_mean.columns = heatmap_vscore_mean.columns.astype(int)

heatmap_tscore_mean = clf_summary.pivot(index = 'model', columns = 'n_features', values = 'test_score')
heatmap_tscore_mean.columns = heatmap_tscore_mean.columns.astype(int)

#%% plot figures

# define plotting order alphabetically

order = list(models.keys())
order.sort()

# plot validation and test scores

f1 = plt.figure(figsize = (6, 4))
ax = sns.heatmap(heatmap_vscore_mean, cmap = 'Blues', linewidths = 0.5, annot = True, fmt = '.2f')
#ax.set_aspect(1)
plt.ylabel('Classification model')
plt.xlabel('Number of features')

f2 = plt.figure(figsize = (6, 4))
ax = sns.heatmap(heatmap_tscore_mean, cmap = 'Blues', linewidths = 0.5, annot = True, fmt = '.2f')
#ax.set_aspect(1)
plt.ylabel('Classification model')
plt.xlabel('Number of features')

f3 = plt.figure(figsize = (6, 4))
ax = sns.lineplot(data = clf_summary, x = 'n_features', y = 'validation_score', 
                  label = 'Validation', ci = 95, color = 'blue')
ax = sns.lineplot(data = clf_summary, x = 'n_features', y = 'test_score', 
                  label = 'Test', ci = 95, color = 'k')
ax.grid(True)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.autoscale(enable = True, axis = 'x', tight = True)
plt.legend(loc = 'lower right')
plt.ylabel('Mean score')
plt.xlabel('Number of features')

# stripplots

f4, ax = plt.subplots(figsize = (16, 4))
#sns.despine(bottom=True, left=True)
sns.stripplot(x = 'model', y = 'validation_score', hue = 'n_features', data = clf_results, 
              order = order, dodge = True, jitter = True, alpha = .25, zorder = 1)
sns.pointplot(x = 'model', y = 'validation_score', hue = 'n_features', data = clf_results,
              order = order, dodge = .532, join = False, palette = 'dark', markers = 'd', scale = .75, ci = None)
ax.yaxis.grid()
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[5:], labels[5:], title = 'Number of features', handletextpad = 0, 
          columnspacing = 1, loc = 'lower center', ncol = 5, frameon = True)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
plt.ylabel('Score')
plt.xlabel('Classification model')

f5, ax = plt.subplots(figsize = (16, 4))
sns.stripplot(x = 'model', y = 'test_score', hue = 'n_features', data = clf_results, 
              order = order, dodge = True, jitter = True, alpha = .25, zorder = 1)
sns.pointplot(x = 'model', y = 'test_score', hue = 'n_features', data = clf_results, 
              order = order, dodge = .532, join = False, palette = 'dark', markers = 'd', scale = .75, ci = None)
ax.yaxis.grid()
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[5:], labels[5:], title = 'Number of features', handletextpad = 0, 
          columnspacing = 1, loc = 'lower center', ncol = 5, frameon = True)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
plt.ylabel('Score')
plt.xlabel('Classification model')

# boxplots

f6 = plt.figure(figsize = (16, 4))
ax = sns.boxplot(x = 'model', y = 'validation_score', hue = 'n_features', data = clf_results,
                 order = order, whis = 1.5, fliersize = 2, notch = True)
#ax = sns.swarmplot(x = 'model', y = 'validation_score', data = clf_results,
#                   order = order, size = 2, color = '.3', linewidth = 0)
ax.yaxis.grid()
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title = 'Number of features', handletextpad = 0.1, 
          columnspacing = 1, loc = 'lower center', ncol = 5, frameon = True)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
plt.ylabel('Score')
plt.xlabel('Classification model')

f7 = plt.figure(figsize = (16, 4))
ax = sns.boxplot(x = 'model', y = 'test_score', hue = 'n_features', data = clf_results,
                 order = order, whis = 1.5, fliersize = 2, notch = True)
#ax = sns.swarmplot(x = 'model', y = 'test_score', data = clf_results,
#                   order = order, size = 2, color = '.3', linewidth = 0)
ax.yaxis.grid()
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title = 'Number of features', handletextpad = 0.1, 
          columnspacing = 1, loc = 'lower center', ncol = 5, frameon = True)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
plt.ylabel('Score')
plt.xlabel('Classification model')

# violinplots

f8 = plt.figure(figsize = (16, 4))
ax = sns.violinplot(x = 'model', y = 'validation_score', hue = 'n_features', data = clf_results,
                    order = order)
ax.yaxis.grid()
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title = 'Number of features', handletextpad = 0.1, 
          columnspacing = 1, loc = 'lower center', ncol = 5, frameon = True)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
plt.ylabel('Score')
plt.xlabel('Classification model')

f9 = plt.figure(figsize = (16, 4))
ax = sns.violinplot(x = 'model', y = 'test_score', hue = 'n_features', data = clf_results,
                    order = order)
ax.yaxis.grid()
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title = 'Number of features', handletextpad = 0.1, 
          columnspacing = 1, loc = 'lower center', ncol = 5, frameon = True)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
plt.ylabel('Score')
plt.xlabel('Classification model')

#%% save data

# make directory

model_dir = os.path.join('Model selection', 
                         ('%s_NF%d_NM%d_NI%d' % (timestr, max(n_features), len(models), n_iterations)))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
# save parameters into text file
    
with open(os.path.join(model_dir, 'parameters.txt'), 'w') as text_file:
    text_file.write('timestr: %s\n' % timestr)
    text_file.write('Computation time: %.1f min\n' % ((end_time - start_time) / 60))
    text_file.write('Number of samples: %d\n' % len(df))
    text_file.write('Number of features: %d\n' % len(feature_labels))
    text_file.write('models: %s\n' % str(list(models.keys())))
    text_file.write('duplicates: %s\n' % str(duplicates))
    text_file.write('n_iterations: %d\n' % n_iterations)
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
    
    f1.savefig(os.path.join(model_dir, ('heatmap_vscore_mean.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f2.savefig(os.path.join(model_dir, ('heatmap_tscore_mean.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f3.savefig(os.path.join(model_dir, ('lineplot_scores.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f4.savefig(os.path.join(model_dir, ('stripplot_vscore.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f5.savefig(os.path.join(model_dir, ('stripplot_tscore.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f6.savefig(os.path.join(model_dir, ('boxplot_vscore.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f7.savefig(os.path.join(model_dir, ('boxplot_tscore.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f8.savefig(os.path.join(model_dir, ('violinplot_vscore.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f9.savefig(os.path.join(model_dir, ('violinplot_tscore.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)

# save variables
    
variable_names = %who_ls DataFrame ndarray list dict str bool int int64 float float64
variables = dict((name, eval(name)) for name in variable_names)
    
pickle.dump(variables, open(os.path.join(model_dir, 'variables.pkl'), 'wb'))
