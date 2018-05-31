# -*- coding: utf-8 -*-
"""
Created on Thu May 31 09:09:30 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    May 2018
    
@description:
    
    This code is used to predict the HIFU therapy outcome (non-perfused volume)
    for uterine fibroids based on their pre-treatment MR parameters
    
"""

#%% import necessary libraries

#import math

from IPython import display
#from matplotlib import cm
#from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
#from sklearn import metrics
import tensorflow as tf
#from tensorflow.python.data import Dataset

from train_linear_regression_model import train_linear_regression_model

#%% define logging and data display format

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#%% read data

fibroid_dataframe = pd.read_csv(r"C:\Users\visa\Documents\TYKS\Machine learning\Uterine fibroid\Test_data.csv", sep=",")

#%% format data

# randomise and scale the data

fibroid_dataframe = fibroid_dataframe.reindex(np.random.permutation(fibroid_dataframe.index))

# examine data

print("\nFirst five entries of the data:\n")
display.display(fibroid_dataframe.head())
print("\nSummary of the data:\n")
display.display(fibroid_dataframe.describe())

#%% divide data into training and validation sets

training_set = fibroid_dataframe.head(30)
validation_set = fibroid_dataframe.tail(13)

#%% display correlation matrix to help select suitable features

print("\nCorrelation matrix:\n")
display.display(training_set.corr())

#%% display histogram and scatter plot of a selected feature

plot_feature = "ADC" # select the feature to plot

plt.figure(figsize=(13, 4))

# histogram

plt.subplot(1, 2, 1)
plt.xlabel("%s" % plot_feature)
plt.ylabel('Number of instances')
plt.title("Feature value distribution")
training_set[plot_feature].hist()

# scatter plot

plt.subplot(1, 2, 2)
plt.xlabel("%s" % plot_feature)
plt.ylabel("NPV")
plt.title("Correlation of variables")
plt.scatter(training_set[plot_feature], training_set["NPV"])

#%% select features and targets

training_features = training_set[["ADC", "T2"]]
training_targets = training_set[["NPV"]]

validation_features = validation_set[["ADC", "T2"]]
validation_targets = validation_set[["NPV"]]

#%% plot training and validation set scatter plot

plt.figure(figsize=(13, 4))

# training set

plt.subplot(1, 2, 1)
plt.xlabel("ADC")
plt.ylabel("T2")
plt.title("Training data")
plt.scatter(training_features["ADC"],
            training_features["T2"],
            cmap="coolwarm",
            c=training_targets["NPV"] / training_targets["NPV"].max())

# validation set

plt.subplot(1,2,2)
plt.xlabel("ADC")
plt.ylabel("T2")
plt.title("Validation data")
plt.scatter(validation_features["ADC"],
            validation_features["T2"],
            cmap="coolwarm",
            c=validation_targets["NPV"] / validation_targets["NPV"].max())

#%% train using linear regression model function

linear_regressor = train_linear_regression_model(
    learning_rate=0.00002,
    steps=800,
    batch_size=5,
    training_features=training_features,
    training_targets=training_targets,
    validation_features=validation_features,
    validation_targets=validation_targets)
