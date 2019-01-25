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
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
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

feature_labels = ['white', 'black', 'asian', 'Age', 'Weight', 'Gravidity', 'Parity',
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

n_iterations = 2

# define maximum number of features

k = 20

# define split ratio for training and testing sets

split_ratio = 0.2

# define scaling type (log or None)

scaling_type = 'log'

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

# define fitting parameters

cv = 10
scoring = 'f1_micro'
n_features = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# define parameters for parameter search

grid_param =    [
                {
                'kernel': ['rbf'], 
                'C': [0.01, 0.1, 1, 10, 100, 1000],
                'gamma': ['auto', 'scale']
                },
                {
                'kernel': ['linear'], 
                'C': [0.01, 0.1, 1, 10, 100, 1000]
                }
                ]

# define data imputation values

impute_labels = ['Height', 'Gravidity', 'bleeding', 'pain', 'mass', 'urinary',
                 'infertility']

# empty dataframe for storing results

clf_results = pd.DataFrame()

#%% start the iteration

timestr = time.strftime('%Y%m%d-%H%M%S')
start_time = time.time()

for iteration in range(0, n_iterations):
    
    # define random state

    random_state = np.random.randint(0, 10000)
    
    # print progress
    
    print('Iteration %d with random state %d at %.1f s' % (iteration, random_state, 
                                                           (time.time() - start_time)))
    
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
    
    # calculate class weights
    
    class_weights = compute_class_weight('balanced', np.unique(training_targets), 
                                         training_targets[target_label[0]])
    class_weights = dict(enumerate(class_weights))
    
    # find k best features for each method
    
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
    
    # define classification model
    
    clf_model = SVC(probability = True, random_state = random_state, class_weight = class_weights)
    
    # define parameter search method
    
    clf_grid = GridSearchCV(clf_model, grid_param, n_jobs = -1, cv = cv, 
                            scoring = scoring, refit = True, iid = False)
    
    # train model using parameter search

    for n in n_features:
        for method in methods:
            
            # fit parameter search
        
            clf_fit = clf_grid.fit(training_features[k_features[method][0:n]].values, training_targets.values[:, 0])
            
            # obtain best results
            
            best_model = clf_fit.best_estimator_
            testing_predictions = best_model.predict(testing_features[k_features[method][0:n]].values)
            test_score = f1_score(testing_targets.values[:, 0], testing_predictions,
                                  average = 'micro')
            
            # save results
            
            df = pd.DataFrame(clf_fit.best_params_, index = [0])
            df['validation_score'] = clf_fit.best_score_
            df['test_score'] = test_score
            df['method'] = method
            df['n_features'] = n
            df['iteration'] = iteration
            df['random_state'] = random_state
            clf_results = clf_results.append(df, sort = False, ignore_index = True)
            
            del clf_fit, df, best_model, testing_predictions, test_score
    
    del n, method
    del clf_model, clf_grid, k_features, class_weights, random_state, impute_values
    del training_set, training_features, training_targets
    del testing_set, testing_features, testing_targets
    
del iteration

end_time = time.time()

# summarise results

print('Execution time: %.2f s' % (end_time - start_time))

#%% plot heatmap

# summarise results

clf_summary = pd.DataFrame()

for method in methods:
    for n in n_features:
        
        validation_scores = clf_results[(clf_results['method'] == method) & 
                                        (clf_results['n_features'] == n)]['validation_score']
        test_scores = clf_results[(clf_results['method'] == method) & 
                                  (clf_results['n_features'] == n)]['test_score']
        
        df = {}
        df['method'] = method
        df['n_features'] = n
        df['mean_validation_score'] = validation_scores.mean()
        df['mean_test_score'] = test_scores.mean()
        df['std_validation_score'] = validation_scores.std()
        df['std_test_score'] = test_scores.std()
        
        clf_summary = clf_summary.append(df, sort = False, ignore_index = True)
        
        del df, validation_scores, test_scores
    
del method, n

clf_summary = clf_summary[['method', 'n_features', 'mean_validation_score',
                          'std_validation_score', 'mean_test_score',
                          'std_test_score']]

# calculate heatmaps

heatmap_validation = []
heatmap_test = []

for n in n_features:

    heatmap_validation.append(list(clf_summary.loc[clf_summary['n_features'] == n]['mean_validation_score']))
    heatmap_test.append(list(clf_summary.loc[clf_summary['n_features'] == n]['mean_test_score']))
    
del n
    
heatmap_validation = pd.DataFrame(heatmap_validation, index = n_features, columns = methods).T
heatmap_test = pd.DataFrame(heatmap_test, index = n_features, columns = methods).T

# plot heatmap

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

#%% save features

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

variables_to_save = {'nan_percent': nan_percent,
                     'grid_param': grid_param,
                     'impute_labels': impute_labels,
                     'k': k,
                     'cv': cv,
                     'scoring': scoring,
                     'n_features': n_features,
                     'n_iterations': n_iterations,
                     'methods': methods,
                     'clf_results': clf_results,
                     'clf_summary': clf_summary,
                     'heatmap_validation': heatmap_validation,
                     'heatmap_test': heatmap_test,
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
