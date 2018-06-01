# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:56:05 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    June 2018
    
@description:
    
    This code defines a neural network regression model which is called 
    to train on selected features
    
"""

#%% import necessary packages

import math

from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import tensorflow as tf
from my_input_fn import my_input_fn
from construct_feature_columns import construct_feature_columns

#%% define function

def train_neural_network_regression_model(
        learning_rate, 
        steps, 
        batch_size,
        hidden_units,
        training_features,
        training_targets,
        validation_features,
        validation_targets
        ):
    
    """
    Args:
        learning rate: the learning rate (float)
        steps: total number of training steps (int)
        batch_size: batch size to used to calculate the gradient (int)
        hidden_units: number of neurons in each layrs (list)
        training_features: one or more columns of training features (DataFrame)
        training_targets: a single column of training targets (DataFrame)
        calidation_features: one or more columns of validation features (DataFrame)
        validation_targets: a single column of validation targets (DataFrame)
        
    Returns:
        A `DNNRegressor` object trained on the training data
    """
    
    # define periods
    
    periods = 10
    steps_per_period = steps / periods
    
    # create linear regressor object
    
    my_optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #my_optimiser = tf.train.AdagradOptimizer(learning_rate=learning_rate) # for convex problems
    #my_optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate) # for non-convex problems
    my_optimiser = tf.contrib.estimator.clip_gradients_by_norm(my_optimiser, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
            feature_columns=construct_feature_columns(training_features),
            hidden_units=hidden_units,
            optimizer=my_optimiser)
    
    # define input functions
    
    training_input_fn = lambda: my_input_fn(
      training_features, 
      training_targets["NPV"], 
      batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(
      training_features, 
      training_targets["NPV"], 
      num_epochs=1, 
      shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(
      validation_features, 
      validation_targets["NPV"], 
      num_epochs=1, 
      shuffle=False)
    
    # print training progress
    
    print("Model training started")
    print("RMSE on training data:")
    
    training_rmse = []
    validation_rmse = []

    for period in range (0, periods):
               
        # train the model
               
        dnn_regressor.train(
                input_fn=training_input_fn,
                steps=steps_per_period
                )
                
        # compute predictions
        
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        
        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
        
        # calculate losses
        
        training_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(validation_predictions, validation_targets))
        
        # print the current loss
        
        print("Period %02d: %0.2f" % (period, training_root_mean_squared_error))
        
        # add loss metrics to the list
        
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
        
    print("Model training finished")
    
    # plot loss metrics over periods
    
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.grid()
    plt.plot(training_rmse, label="Training")
    plt.plot(validation_rmse, label="Validation")
    plt.legend()
    
    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)
    
    return dnn_regressor
