# -*- coding: utf-8 -*-
"""
Created on Fri May 25 09:31:49 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    May 2018
    
@description:
    
    This code is used to classify uterine fibroids based on their MR parameters
    
"""

#%% import necessary libraries

import math

#from IPython import display
#from matplotlib import cm
#from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

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

fibroid_dataframe.head() # first five entries
fibroid_dataframe.describe() # statistics

# define features

selected_features = fibroid_dataframe[["ADC"]] # select which features to use
feature_columns = [tf.feature_column.numeric_column("ADC")] # define as numeric

# define labels

targets = fibroid_dataframe["NPV"]

#%% configure linear regressor model

my_optimiser = tf.train.GradientDescentOptimizer(learning_rate = 0.0000001) # define optimiser type
my_optimiser = tf.contrib.estimator.clip_gradients_by_norm(my_optimiser, 5.0) # enable gradient clipping

linear_regressor = tf.estimator.LinearRegressor(
        feature_columns = feature_columns,
        optimizer = my_optimiser
        )

#%% define input function

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    
    """ Trains linear regressor model with given features
    
    Args:
        features: pandas Dataframe of features
        targets: pandas Dataframe of targets
        batch_size: number of examples to calculate the gradiet
        shuffle: boolean to shuffle the data
        num_epochs: number of iterations, None = repeat indefinitely
    Returns:
        (features, labels) for next data batch
    """
    
    # convert pandas data into a dict of np arrays
    
    features = {key:np.array(value) for key, value in dict(features).items()}
    
    # construct a dataset and configure batching/repeating
    
    ds = Dataset.from_tensor_slices((features, targets)) # 2 GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # shuffle data if selected
    
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
        
    # return the next batch of data
    
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

#%% train the model
    
_ = linear_regressor.train(
        input_fn = lambda:my_input_fn(selected_features, targets),
        steps=100
        )

#%% evaluate the model

prediction_input_fn = lambda:my_input_fn(selected_features, targets, num_epochs=1, shuffle=False)

# call predict() on the linear regressor to make predictions

predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# format predictions as a NumPy array so that error metrics can be calculated

predictions = np.array([item['predictions'][0] for item in predictions])

# print MSE and RMSE

mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)

print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

# compare RMSE with min and max values of the targets

min_npv = fibroid_dataframe["NPV"].min()
max_npv = fibroid_dataframe["NPV"].max()
min_max_difference = max_npv - min_npv

print("Min. Non-perfused volume: %0.3f" % min_npv)
print("Max. Non-perfused volume: %0.3f" % max_npv)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

# compare predictions with targets

calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
calibration_data.describe()

#%% plot linear regression model fitted to data

# obtain sample dataset

sample = fibroid_dataframe.sample(n=len(fibroid_dataframe))

# get the min and max of features

x_0 = sample["ADC"].min()
x_1 = sample["ADC"].max()

# retrieve the final weight and bias generated during training

weight = linear_regressor.get_variable_value('linear/linear_model/ADC/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# get the predicted target values for the min and max selected feature values

y_0 = weight * x_0 + bias 
y_1 = weight * x_1 + bias

# plot regression line from (x_0, y_0) to (x_1, y_1)

plt.plot([x_0, x_1], [y_0, y_1], c='r')

# label the graph axes

plt.ylabel("NPV")
plt.xlabel("ADC")

# plot a scatter plot from data sample set

plt.scatter(sample["ADC"], sample["NPV"])

# display graph

plt.show()

#%% train using linear regression model function

train_linear_regression_model(input_dataframe=fibroid_dataframe,
                              learning_rate=0.00002,
                              steps=800,
                              batch_size=5)
