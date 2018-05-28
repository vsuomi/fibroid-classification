# -*- coding: utf-8 -*-
"""
Created on Fri May 25 15:14:38 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    May 2018
    
@description:
    
    This code defines a linear regression model which is called to train on
    selected features
    
"""

#%% import necessary packages

import math

from IPython import display
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from my_input_fn import my_input_fn

#%% define function

def train_linear_regression_model(
        input_dataframe,
        learning_rate, 
        steps, 
        batch_size, 
        feature_labels="ADC",
        target_labels="NPV"
        ):
    
    """
    Args:
        learning rate: the learning rate (float)
        steps: total number of training steps (int)
        batch_size: batch size to used to calculate the gradient (int)
        feature_labels: features used for training (string)
        target_labels: targets used for training (string)
    """
    
    # define periods
    
    periods = 10
    steps_per_period = steps / periods
    
    # define features
    
    features = input_dataframe[[feature_labels]]
    feature_columns = [tf.feature_column.numeric_column(feature_labels)] # define as numeric
    
    # define targets

    targets = input_dataframe[target_labels]
    
    # define input functions
    
    training_input_fn = lambda:my_input_fn(features, targets, batch_size=batch_size)
    prediction_input_fn = lambda:my_input_fn(features, targets, num_epochs=1, shuffle=False)
    
    # create linear regressor object
    
    my_optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimiser = tf.contrib.estimator.clip_gradients_by_norm(my_optimiser, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
            feature_columns=feature_columns,
            optimizer=my_optimiser)
    
    # plot regression line each period
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned regression line in each period")
    plt.ylabel(target_labels)
    plt.xlabel(feature_labels)
    sample = input_dataframe.sample(n=len(input_dataframe))
    plt.scatter(sample[feature_labels], sample[target_labels])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]
    
    # print training progress
    
    print("Model training started")
    print("RMSE on training data:")
    root_mean_squared_errors = []

    for period in range (0, periods):
               
        # train the model
               
        linear_regressor.train(
                input_fn=training_input_fn,
                steps=steps_per_period
                )
                
        # compute predictions
        
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])
        
        # calculate loss
        
        root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(predictions, targets))
        
        # print the current loss
        
        print("Period %02d: %0.2f" % (period, root_mean_squared_error))
        
        # add loss metrics to the list
        
        root_mean_squared_errors.append(root_mean_squared_error)
        
        # track the weights and biases over time
        
        y_extents = np.array([0, sample[target_labels].max()])
                
        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % feature_labels)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
    
        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,
                                          sample[feature_labels].max()),
                               sample[feature_labels].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])
        
    print("Model training finished")
    
    # print loss metrics over periods
    
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)
    
    # output a table with calibration data
    
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())
    
    print("Final RMSE on training data: %0.2f" % root_mean_squared_error)
    
    return calibration_data
