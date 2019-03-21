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
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
                  #'Fibroid volume'
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

#%% build and train model

# define parameters for grid search

#parameters =    {
#                'kernel': ['rbf'], 
#                'C': list(np.logspace(-1, 4, 6)),
#                'gamma': list(np.logspace(-2, 4, 7))
#                }

# define parameters for randomised search

parameters =    {
                'kernel': ['rbf'],
                'C': sp.stats.reciprocal(1e-1, 1e4),
                'gamma': sp.stats.reciprocal(1e-2, 1e4)
                }

# define model

max_iter = 200000
cache_size = 4000
class_weight = 'balanced'

base_model = SVC(class_weight = class_weight, random_state = random_state,
                 cache_size = cache_size, max_iter = max_iter, probability = True)

# define parameter search method

cv = 10
scoring = 'f1_micro'

#grid = GridSearchCV(base_model, parameters, scoring = scoring, 
#                    n_jobs = -1, cv = cv, refit = True, iid = False)
grid = RandomizedSearchCV(base_model, parameters, scoring = scoring, 
                          n_jobs = -1, cv = cv, refit = True, iid = False,
                          n_iter = 1000, random_state = random_state)

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
training_predictions = pd.DataFrame(training_predictions, columns = target_label,
                                    index = training_features.index, dtype = float)

testing_predictions = best_model.predict(testing_features.values)
testing_predictions = pd.DataFrame(testing_predictions, columns = target_label,
                                   index = testing_features.index, dtype = float)

# calculate evaluation metrics

training_accuracy = f1_score(training_targets, training_predictions, average = scoring[3:])
validation_accuracy = grid.best_score_
testing_accuracy = f1_score(testing_targets, testing_predictions, average = scoring[3:])

# confusion matrix

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
                         ('%s_TA%d_VA%d_TA%d' % (timestr, 
                                                 round(training_accuracy*100),
                                                 round(validation_accuracy*100),
                                                 round(testing_accuracy*100))))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
# save parameters into text file
    
with open(os.path.join(model_dir, 'parameters.txt'), 'w') as text_file:
    text_file.write('timestr: %s\n' % timestr)
    text_file.write('estimator: %s\n' % type(base_model).__name__)
    text_file.write('Computation time: %.1f min\n' % ((end_time - start_time) / 60))
    text_file.write('Total samples: %d\n' % len(df))
    text_file.write('Training samples: %d\n' % len(training_set))
    text_file.write('Testing samples: %d\n' % len(testing_set))
    text_file.write('Total features: %d\n' % len(feature_labels))
    text_file.write('feature_labels: %s\n' % str(feature_labels))
    text_file.write('target_label: %s\n' % str(target_label))
    text_file.write('duplicates: %s\n' % str(duplicates))
    text_file.write('scaling_type: %s\n' % scaling_type)
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

# save model
    
pickle.dump(variables, open(os.path.join(model_dir, 'variables.pkl'), 'wb'))
pickle.dump(best_model, open(os.path.join(model_dir, 'scikit_model.pkl'), 'wb'))
