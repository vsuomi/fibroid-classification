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

#%% define random state

random_state = np.random.randint(0, 1000)

#%% define logging and data display format

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#%% read data

fibroid_dataframe = pd.read_csv(r'fibroid_dataframe.csv', sep = ',')

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

feature_labels = ['white', 'black', 'asian', 'Age', 'Weight', 'Gravidity', 'Parity',
                  'History_of_pregnancy', 'Live_births', 'C-section', 'esmya', 
                  'open_myomectomy', 'laprascopic_myomectomy', 'hysteroscopic_myomectomy',
                  'embolisation', 'Subcutaneous_fat_thickness', 'Front-back_distance', 
                  'Abdominal_scars', 'bleeding', 'pain', 'mass', 'urinary', 'infertility',
                  'Fibroid_diameter', 'Fibroid_distance', 'intramural', 'subserosal', 
                  'submucosal', 'anterior', 'posterior', 'lateral', 'fundus',
                  'anteverted', 'retroverted', 'Type_I', 'Type_II', 'Type_III',
                  'ADC', 'Fibroid_volume']

target_label = ['NPV_class']

#%% randomise and divive data for cross-validation

# stratified splitting for unbalanced datasets

split_ratio = 0.2
training_set, testing_set = train_test_split(fibroid_dataframe, test_size = split_ratio,
                                             stratify = fibroid_dataframe[target_label],
                                             random_state = random_state)

#%% define features and targets

training_features = training_set[feature_labels]
testing_features = testing_set[feature_labels]

training_targets = training_set[target_label]
testing_targets = testing_set[target_label]

#%% scale features

scaling_type = 'log'

if scaling_type == 'z-score':

    z_mean = training_features.mean()
    z_std = training_features.std()
    
    training_features = (training_features - z_mean) / z_std
    testing_features = (testing_features - z_mean) / z_std
    
elif scaling_type == 'log':
    
    training_features = np.log1p(training_features)
    testing_features = np.log1p(testing_features)

#%% calculate class weights

class_weights = compute_class_weight('balanced', np.unique(training_targets), 
                                     training_targets[target_label[0]])
class_weights = dict(enumerate(class_weights))

#%% find best features

# number of features

k = 20

# define scorer names

names = ['fisher_score', 
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
         'mifs',
         'f_classif',
         'chi2',
         'mutual_info_classif'
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
           MIFS.mifs,
           f_classif,
           chi2,
           mutual_info_classif
           ]

# define scorer rankers (sk-feature only)

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
           None,
           None,
           None,
           None
           ]

# find k best features for each scorer

k_features = pd.DataFrame(index = range(0, k), columns = names)

for scorer, ranker, name in zip(scorers, rankers, names):
    
    if name in ('disr', 'cmim', 'icap', 'jmi', 'cife', 'mim', 'mrmr', 'mifs', 'trace_ratio'):
        
        indices, _, _ = scorer(training_features.values, training_targets.values[:, 0], n_selected_features = k)
        k_features[name] = pd.DataFrame(training_features.columns.values[indices], columns = [name])
        
    elif name in ('f_classif', 'chi2', 'mutual_info_classif'):
        
        selector = SelectKBest(scorer, k = k)
        selector.fit(training_features.values, training_targets.values[:, 0])
        k_features[name] = list(training_features.columns[selector.get_support(indices = True)])
    
    else:
        
        scores = scorer(training_features.values, training_targets.values[:, 0])
        indices = ranker(scores)
        k_features[name] = pd.DataFrame(training_features.columns.values[indices[0:k]], columns = [name])
        
del scorer, ranker, name, selector, scores, indices

#%% find best number of features for each scorer

# define fitting parameters

cv = 10
scoring = 'f1_micro'
n_features = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# define parameters for parameter search

parameters =    [
                {
                'kernel': ['rbf'], 
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'gamma': ['auto', 'scale']
                },
                {
                'kernel': ['linear'], 
                'C': [0.001, 0.01, 0.1, 1, 10, 100]
                }
                ]

# define model

svc_model = SVC(probability = True, random_state = random_state, class_weight = class_weights)

# define parameter search method

clf = GridSearchCV(svc_model, parameters, n_jobs = -1, cv = cv, 
                   scoring = scoring, refit = True)

# train model using parameter search
    
timestr = time.strftime('%Y%m%d-%H%M%S')
start_time = time.time()

clf_results = pd.DataFrame()

i = 0

for n in n_features:
    for name in names:
    
        clf.fit(training_features[k_features[name][0:n]].values, training_targets.values[:, 0])
        
        model = clf.best_estimator_
        testing_predictions = model.predict(testing_features[k_features[name][0:n]].values)
        test_score = f1_score(testing_targets.values[:, 0], testing_predictions,
                              average = 'micro')
        
        df = pd.DataFrame(clf.best_params_, index = [i])
        df['validation_score'] = clf.best_score_
        df['test_score'] = test_score
        df['method'] = name
        df['n_features'] = n
        clf_results = clf_results.append(df)
        
        del df, model, testing_predictions, test_score
        
        i += 1

del n, i, name

end_time = time.time()

# summarise results

print('Execution time: %.2f s' % (end_time - start_time))

#%% plot heatmap

# collect best scores for each n features

heatmap_validation = []
heatmap_test = []

for n in n_features:
    
    heatmap_validation.append(list(clf_results.loc[clf_results['n_features'] == n]['validation_score']))
    heatmap_test.append(list(clf_results.loc[clf_results['n_features'] == n]['test_score']))
    
del n
    
heatmap_validation = pd.DataFrame(heatmap_validation, index = n_features, columns = names).T
heatmap_test = pd.DataFrame(heatmap_test, index = n_features, columns = names).T

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

model_dir = 'Feature selection\\%s_%s_NF%d_NM%d' % (timestr, scoring, 
                                                    max(n_features), len(names))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
f1.savefig(model_dir + '\\' + 'heatmap_validation.pdf', dpi = 600, format = 'pdf',
           bbox_inches = 'tight', pad_inches = 0)
f2.savefig(model_dir + '\\' + 'heatmap_test.pdf', dpi = 600, format = 'pdf',
           bbox_inches = 'tight', pad_inches = 0)

variables_to_save = {'nan_percent': nan_percent,
                     'parameters': parameters,
                     'k': k,
                     'cv': cv,
                     'scoring': scoring,
                     'n_features': n_features,
                     'names': names,
                     'k_features': k_features,
                     'clf': clf,
                     'clf_results': clf_results,
                     'heatmap_validation': heatmap_validation,
                     'heatmap_test': heatmap_test,
                     'start_time': start_time,
                     'end_time': end_time,
                     'random_state': random_state,
                     'class_weights': class_weights,
                     'NPV_bins': NPV_bins,
                     'split_ratio': split_ratio,
                     'timestr': timestr,
                     'scaling_type': scaling_type,
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
