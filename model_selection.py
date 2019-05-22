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
from sklearn.metrics import f1_score

# import classifiers

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from logitboost import LogitBoost
from xgboost import XGBClassifier

#%% define logging and data display format

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
pd.options.mode.chained_assignment = None                                       # disable imputation warnings

#%% read data

df = pd.read_csv(r'fibroid_dataframe.csv', sep = ',')

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

impute = True

# discretise features

discretise = True

# define scaling type ('log', 'minmax', 'standard' or None)

scaling_type = 'log'

# parameters for grid search

cv = 10
scoring = 'f1_micro'

# initialise variables

clf_results = pd.DataFrame()

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
                        'class_weight': ['balanced']
                        }

param_random_forest =   {
                        'n_estimators': [10, 50, 100, 200, 300],
                        'min_samples_split': [2, 4],
                        'max_features': ['sqrt', None],
                        'class_weight': ['balanced']
                        }

param_adaboost =        {
                        'n_estimators': [10, 50, 100, 200, 300],
                        'learning_rate': [0.1, 0.5, 1, 5, 10]
                        }

param_gradient_boost =  {
                        'n_estimators': [10, 50, 100, 200, 300],
                        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5, 1],
                        'subsample': [0.8, 0.9, 1],
                        'min_samples_split': [2, 4],
                        'max_features': ['sqrt', None]
                        }

param_svc =             {
                        'kernel': ['rbf'],
                        'C': list(np.logspace(-1, 4, 6)),
                        'gamma': list(np.logspace(-2, 4, 7)),
                        'class_weight': ['balanced']
                        }                    

param_logitboost  =     {
                        'n_estimators': [10, 50, 100, 200, 300],
                        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5, 1]
                        }

param_xgb =             {
                        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5, 1],
                        'n_estimators': [10, 50, 100, 200, 300],
                        'subsample': [0.8, 0.9, 1],
                        'reg_alpha': [0, 1, 5, 10],
                        'reg_lambda': [0, 1, 5, 10]
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
    
    if impute == True:
    
        impute_mean =   ['Height']
        
        impute_mode =   ['Gravidity']
        
#        impute_cons =   []
        
        imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        imp_mode = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
#        imp_cons = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 0)
        
        training_features[impute_mean] = imp_mean.fit_transform(training_features[impute_mean])
        testing_features[impute_mean] = imp_mean.transform(testing_features[impute_mean])
        
        training_features[impute_mode] = imp_mode.fit_transform(training_features[impute_mode])
        testing_features[impute_mode] = imp_mode.transform(testing_features[impute_mode])
        
#        training_features[impute_cons] = imp_cons.fit_transform(training_features[impute_cons])
#        testing_features[impute_cons] = imp_cons.transform(testing_features[impute_cons])
        
        del imp_mean, imp_mode, #imp_cons
    
    # discretise features
    
    if discretise == True:
    
        disc_labels =   ['Age', 
                         'Weight', 
                         'Height',
                         'Subcutaneous fat thickness', 
                         'Front-back distance',
                         'Fibroid diameter', 
                         'Fibroid distance'
                         ]
        
        enc = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'uniform')
        
        training_features[disc_labels] = enc.fit_transform(training_features[disc_labels])
        testing_features[disc_labels] = enc.transform(testing_features[disc_labels])
        
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
            test_score = f1_score(testing_targets.values[:, 0], testing_predictions, average = scoring[3:])
            
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

# define colormap

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

# plot test and validation score in boxplot

f4, ax = plt.subplots(figsize = (16, 4))
#sns.despine(bottom=True, left=True)
sns.stripplot(x = 'model', y = 'validation_score', hue = 'n_features', data = clf_results, 
              dodge = True, jitter = True, alpha = .25, zorder = 1)
sns.pointplot(x = 'model', y = 'validation_score', hue = 'n_features', data = clf_results,
              dodge = .532, join = False, palette = 'dark', markers = 'd', scale = .75, ci = None)
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
              dodge = True, jitter = True, alpha = .25, zorder = 1)
sns.pointplot(x = 'model', y = 'test_score', hue = 'n_features', data = clf_results, 
              dodge = .532, join = False, palette = 'dark', markers = 'd', scale = .75, ci = None)
ax.yaxis.grid()
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[5:], labels[5:], title = 'Number of features', handletextpad = 0, 
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
    text_file.write('discretise: %s\n' % str(discretise))
    text_file.write('impute: %s\n' % str(impute))
    text_file.write('scaling_type: %s\n' % scaling_type)
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
    f4.savefig(os.path.join(model_dir, ('swarmplot_vscore.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)
    f5.savefig(os.path.join(model_dir, ('swarmplot_tscore.' + filetype)), dpi = 600, format = filetype,
               bbox_inches = 'tight', pad_inches = 0)

# save variables
    
variable_names = %who_ls DataFrame ndarray list dict str bool int int64 float float64
variables = dict((name, eval(name)) for name in variable_names)
    
pickle.dump(variables, open(os.path.join(model_dir, 'variables.pkl'), 'wb'))
