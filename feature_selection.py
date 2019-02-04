# -*- coding: utf-8 -*-
'''
Created on Fri Jan 18 10:59:43 2019

@author:
    
    Visa Suomi
    Turku University Hospital
    January 2019
    
@description:
    
    This code is used for feature selection for different classification models
    
'''

#%% clear variables

%reset -f
%clear

#%% import necessary libraries

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import reliefF
from skfeature.function.similarity_based import trace_ratio
from skfeature.function.statistical_based import gini_index
from skfeature.function.statistical_based import chi_square                     # same as chi2
from skfeature.function.statistical_based import f_score                        # same as f_classif
#from skfeature.function.statistical_based import CFS
#from skfeature.function.statistical_based import t_score                        # only for binary
from skfeature.function.information_theoretical_based import JMI
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import DISR
from skfeature.function.information_theoretical_based import MIM
from skfeature.function.information_theoretical_based import CMIM
from skfeature.function.information_theoretical_based import ICAP
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import MIFS

from save_load_variables import save_load_variables

#%% define logging and data display format

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
pd.options.mode.chained_assignment = None                                       # disable imputation warnings

#%% read data

dataframe = pd.read_csv(r'fibroid_dataframe.csv', sep = ',')

#%% calculate nan percent for each label

nan_percent = pd.DataFrame(dataframe.isnull().mean() * 100, columns = ['% of NaN'])

#%% display NPV histogram

dataframe['NPV_percent'].hist(bins = 20)

#%% categorise NPV into classes according to bins

NPV_bins = [-1, 29.9, 80, 100]
dataframe['NPV_class'] = dataframe[['NPV_percent']].apply(lambda x: pd.cut(x, NPV_bins, labels = False))

#%% define feature and target labels

feature_labels = ['white', 'black', 'asian', 'Age', 'Weight', 'Height', 'Gravidity', 'Parity',
                  'History_of_pregnancy', 'Live_births', 'C-section', 'esmya', 
                  'open_myomectomy', 'laprascopic_myomectomy', 'hysteroscopic_myomectomy',
                  'embolisation', 'Subcutaneous_fat_thickness', 'Front-back_distance', 
                  'Abdominal_scars', 'bleeding', 'pain', 'mass', 'urinary', 'infertility',
                  'Fibroid_diameter', 'Fibroid_distance', 'intramural', 'subserosal', 
                  'submucosal', 'anterior', 'posterior', 'lateral', 'fundus',
                  'anteverted', 'retroverted', 'Type_I', 'Type_II', 'Type_III',
                  'Fibroid_volume']

target_label = ['NPV_class']

#%% define parameters for iteration

# define number of iterations

n_iterations = 100

# define split ratio for training and testing sets

split_ratio = 0.2

# define scaling type ('log', 'minmax' or None)

scaling_type = 'log'

# define number of features

n_features = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# define scorer methods

methods =   ['fisher_score', 
             'reliefF', 
             'trace_ratio',
             'gini_index', 
             'chi_square', 
             'f_score',
             'disr', 
             'cmim',
             'icap',
             'jmi',
             'cife',
             'mim',
             'mrmr',
             'mifs'
             ]

# define scorer functions

scorers = [fisher_score.fisher_score, 
           reliefF.reliefF,
           trace_ratio.trace_ratio,
           gini_index.gini_index,
           chi_square.chi_square, 
           f_score.f_score,
           DISR.disr, 
           CMIM.cmim,
           ICAP.icap,
           JMI.jmi,
           CIFE.cife,
           MIM.mim,
           MRMR.mrmr,
           MIFS.mifs
           ]

# define scorer rankers (for scikit-feature only)

rankers = [fisher_score.feature_ranking, 
           reliefF.feature_ranking,
           None,
           gini_index.feature_ranking,
           chi_square.feature_ranking,
           f_score.feature_ranking,
           None,
           None,
           None,
           None,
           None,
           None, 
           None,
           None
           ]

# define parameters for parameter search

grid_param =    {
                'kernel': ['rbf'], 
                'C': list(np.logspace(-1, 4, 6)),
                'gamma': list(np.logspace(-2, 4, 7)),
                'random_state': [None]
                }

# define data imputation values

impute_labels = ['Height', 'Gravidity', 'bleeding', 'pain', 'mass', 'urinary',
                 'infertility']

# define classification model

max_iter = 200000
class_weight = 'balanced'

clf_model = SVC(probability = True, class_weight = class_weight, cache_size = 4000,
                max_iter = max_iter)

# define parameter search method

cv = 10
scoring = 'f1_micro'
    
clf_grid = GridSearchCV(clf_model, grid_param, n_jobs = -1, cv = cv, 
                        scoring = scoring, refit = True, iid = False)

# initialise variables

clf_results = pd.DataFrame()
feature_rankings = pd.DataFrame()
k = len(feature_labels)

#%% start the iteration

timestr = time.strftime('%Y%m%d-%H%M%S')
start_time = time.time()

for iteration in range(0, n_iterations):
    
    # define random state

    random_state = np.random.randint(0, 10000)
    
    # assign random state to grid parameters
    
    grid_param['random_state'] = [random_state]
    
    # print progress
    
    print('Iteration %d with random state %d at %.1f min' % (iteration, random_state, 
                                                             ((time.time() - start_time) / 60)))
    
    # randomise and divive data for cross-validation
    
    training_set, testing_set = train_test_split(dataframe, test_size = split_ratio,
                                                 stratify = dataframe[target_label],
                                                 random_state = random_state)
    
    impute_values = {}
    
    for label in impute_labels:
        
        if label in {'Height'}:
            
            impute_values[label] = training_set[label].mean()
            
            training_set[label] = training_set[label].fillna(impute_values[label])
            testing_set[label] = testing_set[label].fillna(impute_values[label])
            
        else:
            
            impute_values[label] = training_set[label].mode()[0]
            
            training_set[label] = training_set[label].fillna(impute_values[label])
            testing_set[label] = testing_set[label].fillna(impute_values[label])
            
    del label
    
    # define features and targets
    
    training_features = training_set[feature_labels]
    testing_features = testing_set[feature_labels]
    
    training_targets = training_set[target_label]
    testing_targets = testing_set[target_label]
    
    # scale features
       
    if scaling_type == 'log':
        
        training_features = np.log1p(training_features)
        testing_features = np.log1p(testing_features)
        
    elif scaling_type == 'minmax':
        
        mms = MinMaxScaler(feature_range = (0, 1)) 
        training_features = pd.DataFrame(mms.fit_transform(training_features),
                                         columns = training_features.columns,
                                         index = training_features.index)
        testing_features = pd.DataFrame(mms.transform(testing_features),
                                         columns = testing_features.columns,
                                         index = testing_features.index)
    
    # find k best features for each feature selection method
    
    k_features = pd.DataFrame(index = range(0, k), columns = methods)
    
    for scorer, ranker, method in zip(scorers, rankers, methods):
        
        if method in ('disr', 'cmim', 'icap', 'jmi', 'cife', 'mim', 'mrmr', 'mifs', 'trace_ratio'):
            
            indices, _, _ = scorer(training_features.values, training_targets.values[:, 0], n_selected_features = k)
            k_features[method] = pd.DataFrame(training_features.columns.values[indices], columns = [method])
            
            del indices
            
        elif method in ('f_classif', 'chi2', 'mutual_info_classif'):
            
            selector = SelectKBest(scorer, k = k)
            selector.fit(training_features.values, training_targets.values[:, 0])
            k_features[method] = list(training_features.columns[selector.get_support(indices = True)])
            
            del selector
        
        else:
            
            scores = scorer(training_features.values, training_targets.values[:, 0])
            indices = ranker(scores)
            k_features[method] = pd.DataFrame(training_features.columns.values[indices[0:k]], columns = [method])
            
            del scores, indices
            
    del scorer, ranker, method
    
    # calculate feature scores
    
    k_rankings = pd.DataFrame(k_features.T.values.argsort(1),
                              columns = np.sort(k_features.iloc[:, 0].values),
                              index = k_features.columns)
    k_rankings['method'] = k_rankings.index
    k_rankings['iteration'] = iteration
    k_rankings['random_state'] = random_state
    feature_rankings = feature_rankings.append(k_rankings, sort = False, ignore_index = True)
    
    del k_rankings
    
    # train model using parameter search

    for n in n_features:
        for method in methods:
            
            # fit parameter search
        
            clf_fit = clf_grid.fit(training_features[k_features[method][0:n]].values, training_targets.values[:, 0])
            
            # calculate predictions
            
            testing_predictions = clf_fit.predict(testing_features[k_features[method][0:n]].values)
            test_score = f1_score(testing_targets.values[:, 0], testing_predictions, average = 'micro')
            
            # save results
            
            df = pd.DataFrame(clf_fit.best_params_, index = [0])
            df['method'] = method
            df['validation_score'] = clf_fit.best_score_
            df['test_score'] = test_score
            df['n_features'] = n
            df['iteration'] = iteration
            df['random_state'] = random_state
            clf_results = clf_results.append(df, sort = False, ignore_index = True)
            
            del clf_fit, testing_predictions, test_score, df
    
    del n, method
    del k_features, random_state, impute_values
    del training_set, training_features, training_targets
    del testing_set, testing_features, testing_targets
    
del iteration

end_time = time.time()

# summarise results

print('Total execution time: %.1f min' % ((end_time - start_time) / 60))

#%% calculate summaries

# summarise results

mean_validation = clf_results.groupby(['method', 'n_features'], as_index = False)['validation_score'].mean()
mean_test = clf_results.groupby(['method', 'n_features'])['test_score'].mean().values

std_validation = clf_results.groupby(['method', 'n_features'])['validation_score'].std().values
std_test = clf_results.groupby(['method', 'n_features'])['test_score'].std().values

clf_summary = mean_validation.copy()
clf_summary['test_score'] = mean_test
clf_summary['validation_std'] =  std_validation
clf_summary['test_std'] = std_test

# calculate heatmaps
    
heatmap_validation = clf_summary.pivot(index = 'method', columns = 'n_features', values = 'validation_score')
heatmap_validation.columns = heatmap_validation.columns.astype(int)

heatmap_test = clf_summary.pivot(index = 'method', columns = 'n_features', values = 'test_score')
heatmap_test.columns = heatmap_test.columns.astype(int)

heatmap_gap = heatmap_validation - heatmap_test

# calculate feature rankings

feature_boxplot = feature_rankings[feature_labels].melt(var_name = 'feature', value_name = 'ranking')
feature_order = feature_boxplot.groupby(['feature'])['ranking'].median().sort_values(ascending = True).index

heatmap_rankings = feature_rankings.groupby(['method'], as_index = False)[feature_labels].mean()
heatmap_rankings = heatmap_rankings.set_index('method')

#%% plot figures

# plot validation and test scores

f1 = plt.figure()
ax = sns.heatmap(heatmap_validation, cmap = 'Blues', linewidths = 0.5, annot = True, fmt = ".2f")
#ax.set_aspect(1)
plt.ylabel('Feature selection method')
plt.xlabel('Number of features')

f2 = plt.figure()
ax = sns.heatmap(heatmap_test, cmap = 'Blues', linewidths = 0.5, annot = True, fmt = ".2f")
#ax.set_aspect(1)
plt.ylabel('Feature selection method')
plt.xlabel('Number of features')

f3 = plt.figure()
ax = sns.heatmap(heatmap_gap, cmap = 'Blues', linewidths = 0.5, annot = True, fmt = ".2f")
#ax.set_aspect(1)
plt.ylabel('Feature selection method')
plt.xlabel('Number of features')

f4 = plt.figure()
ax = sns.lineplot(data = clf_summary, x = 'n_features', y = 'validation_score', label = 'Validation')
ax = sns.lineplot(data = clf_summary, x = 'n_features', y = 'test_score', label = 'Test')
ax.grid(True)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.autoscale(enable = True, axis = 'x', tight = True)
plt.legend(loc = 'lower right')
plt.ylabel('Mean score')
plt.xlabel('Number of features')

# plot feature rankings

f5 = plt.figure(figsize = (16, 4))
ax = sns.boxplot(x = 'feature', y = 'ranking', data = feature_boxplot, order = feature_order,
                 whis = 'range', palette = 'Blues')
#ax = sns.swarmplot(x = 'feature', y = 'ranking', data = feature_boxplot, order = feature_order, 
#                   size = 2, color = '.3', linewidth = 0)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, ha = 'right')
plt.ylabel('Ranking')
plt.xlabel('Feature')

f6 = plt.figure(figsize = (22, 4))
ax = sns.heatmap(heatmap_rankings, cmap = 'Blues', linewidths = 0.5, annot = True, fmt = ".1f")
#ax.set_aspect(1)
plt.ylabel('Feature selection method')
plt.xlabel('Feature')

# plot parameter distributions

f7 = plt.figure()
ax = clf_results.C.value_counts().plot(kind = 'bar')
plt.ylabel('Count')
plt.xlabel('C')

f8 = plt.figure()
ax = clf_results.gamma.value_counts().plot(kind = 'bar')
plt.ylabel('Count')
plt.xlabel('Gamma')

#%% save figures and variables

model_dir = 'Feature selection\\%s_NF%d_NM%d_NI%d' % (timestr, 
                                                      max(n_features), 
                                                      len(methods),
                                                      n_iterations)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
f1.savefig(model_dir + '\\' + 'heatmap_validation.pdf', dpi = 600, format = 'pdf',
           bbox_inches = 'tight', pad_inches = 0)
f2.savefig(model_dir + '\\' + 'heatmap_test.pdf', dpi = 600, format = 'pdf',
           bbox_inches = 'tight', pad_inches = 0)
f3.savefig(model_dir + '\\' + 'heatmap_gap.pdf', dpi = 600, format = 'pdf',
           bbox_inches = 'tight', pad_inches = 0)
f4.savefig(model_dir + '\\' + 'lineplot_error.pdf', dpi = 600, format = 'pdf',
           bbox_inches = 'tight', pad_inches = 0)
f5.savefig(model_dir + '\\' + 'feature_rankings.pdf', dpi = 600, format = 'pdf',
           bbox_inches = 'tight', pad_inches = 0)
f6.savefig(model_dir + '\\' + 'heatmap_rankings.pdf', dpi = 600, format = 'pdf',
           bbox_inches = 'tight', pad_inches = 0)
f7.savefig(model_dir + '\\' + 'c_count.pdf', dpi = 600, format = 'pdf',
           bbox_inches = 'tight', pad_inches = 0)
f8.savefig(model_dir + '\\' + 'gamma_count.pdf', dpi = 600, format = 'pdf',
           bbox_inches = 'tight', pad_inches = 0)

variables_to_save = {'nan_percent': nan_percent,
                     'grid_param': grid_param,
                     'impute_labels': impute_labels,
                     'max_iter': max_iter,
                     'class_weight': class_weight,
                     'k': k,
                     'cv': cv,
                     'scoring': scoring,
                     'n_features': n_features,
                     'n_iterations': n_iterations,
                     'methods': methods,
                     'clf_results': clf_results,
                     'clf_summary': clf_summary,
                     'feature_rankings': feature_rankings,
                     'feature_boxplot': feature_boxplot,
                     'feature_order': feature_order,
                     'heatmap_rankings': heatmap_rankings,
                     'heatmap_validation': heatmap_validation,
                     'heatmap_test': heatmap_test,
                     'heatmap_gap': heatmap_gap,
                     'start_time': start_time,
                     'end_time': end_time,
                     'NPV_bins': NPV_bins,
                     'split_ratio': split_ratio,
                     'timestr': timestr,
                     'scaling_type': scaling_type,
                     'model_dir': model_dir,
                     'dataframe': dataframe,
                     'feature_labels': feature_labels,
                     'target_label': target_label}
    
save_load_variables(model_dir, variables_to_save, 'variables', 'save')
