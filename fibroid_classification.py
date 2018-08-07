# -*- coding: utf-8 -*-
"""
Created on Fri May 25 09:31:49 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    May 2018
    
@description:
    
    This code is used to classify the treatability of uterine fibroids before
    HIFU therapy based on their pre-treatment MR parameters
    
"""

#%% import necessary libraries

#import math

from IPython import display
#from matplotlib import cm
#from matplotlib import gridspec
#from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
#from sklearn import metrics
import tensorflow as tf
#from tensorflow.python.data import Dataset

from train_linear_classification_model import train_linear_classification_model
from train_neural_network_classification_model import train_neural_network_classification_model
from scale_features import scale_features

#%% define logging and data display format

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#%% read data

fibroid_dataframe = pd.read_csv(r"C:\Users\visa\Documents\TYKS\Machine learning\Uterine fibroid\test_data.csv", sep=",")

#%% add new feature for logistic regression

fibroid_dataframe["NPV_is_high"] = (fibroid_dataframe["NPV_percent"] > 75).astype(float) # NPV above 75%

#%% format data

# randomise and scale the data

fibroid_dataframe = fibroid_dataframe.reindex(np.random.permutation(fibroid_dataframe.index))

# examine data

print("\nFirst five entries of the data:\n")
display.display(fibroid_dataframe.head())
print("\nSummary of the data:\n")
display.display(fibroid_dataframe.describe())

#%% divide data into training and validation sets

num_training = round(0.7*len(fibroid_dataframe))
num_validation = len(fibroid_dataframe) - num_training

training_set = fibroid_dataframe.head(num_training)
validation_set = fibroid_dataframe.tail(num_validation)

#%% display correlation matrix to help select suitable features

print("\nCorrelation matrix:\n")
display.display(training_set.corr())

#%% select features and targets

training_features = training_set[["Age", "Weight", "Subcutaneous_fat_thickness",
                                  "Front-back_distance", "Fibroid_size", "Fibroid_distance",
                                  "Fibroid_volume"]]
training_targets = training_set[["NPV_is_high"]]

validation_features = validation_set[["Age", "Weight", "Subcutaneous_fat_thickness",
                                  "Front-back_distance", "Fibroid_size", "Fibroid_distance",
                                  "Fibroid_volume"]]
validation_targets = validation_set[["NPV_is_high"]]

#%% scale features

scaled_training_features = scale_features(training_features, "z-score")
scaled_validation_features = scale_features(validation_features, "z-score")

#%% train using linear classification model function

linear_classifier, training_probabilities, validation_probabilities = train_linear_classification_model(
    learning_rate = 0.001,
    steps = 1200,
    batch_size = 10,
    optimiser = "Ftrl",
    training_features = scaled_training_features,
    training_targets = training_targets,
    validation_features = scaled_validation_features,
    validation_targets = validation_targets)

#%% train using neural network classification model function

dnn_classifier, training_probabilities, validation_probabilities = train_neural_network_classification_model(
    learning_rate = 0.001,
    steps = 1200,
    batch_size = 10,
    hidden_units = [10, 10],
    optimiser = "Adam",
    training_features = scaled_training_features,
    training_targets = training_targets,
    validation_features = scaled_validation_features,
    validation_targets = validation_targets)
