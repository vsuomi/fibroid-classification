# -*- coding: utf-8 -*-
"""
Created on Thu May 31 09:16:18 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    May 2018
    
@description:
    
    This code defines a linear classification model which is called to train on
    selected features
    
"""

#%% import necessary packages

from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import tensorflow as tf
from my_input_fn import my_input_fn
from construct_feature_columns import construct_feature_columns

#%% define function

def train_linear_classification_model(
        learning_rate, 
        steps, 
        batch_size, 
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
        training_features: one or more columns of training features (DataFrame)
        training_targets: a single column of training targets (DataFrame)
        calidation_features: one or more columns of validation features (DataFrame)
        validation_targets: a single column of validation targets (DataFrame)
        
    Returns:
        A `LinearClassifier` object trained on the training data
    """
    
    # define periods
    
    periods = 10
    steps_per_period = steps / periods
    
    # create linear regressor object
    
    my_optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #my_optimiser = tf.train.FtrlOptimizer(learning_rate=learning_rate) # for high-dimensional linear models
    #my_optimiser = tf.train.FtrlOptimizer(learning_rate=learning_rate, 
    #                                      l1_regularization_strength=0.0) # for L1 regularisation change the regularisation strength
    my_optimiser = tf.contrib.estimator.clip_gradients_by_norm(my_optimiser, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(
            feature_columns=construct_feature_columns(training_features),
            optimizer=my_optimiser)
    
    # define input functions
    
    training_input_fn = lambda: my_input_fn(
      training_features, 
      training_targets["NPV_is_high"], 
      batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(
      training_features, 
      training_targets["NPV_is_high"], 
      num_epochs=1, 
      shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(
      validation_features, 
      validation_targets["NPV_is_high"], 
      num_epochs=1, 
      shuffle=False)
    
    # print training progress
    
    print("Model training started")
    print("LogLoss on training data:")
    
    training_log_losses = []
    validation_log_losses = []

    for period in range (0, periods):
               
        # train the model
               
        linear_classifier.train(
                input_fn=training_input_fn,
                steps=steps_per_period
                )
                
        # compute predictions
        
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'][0] for item in training_probabilities])
        
        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'][0] for item in validation_probabilities])
        
        # calculate losses
        
        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        
        # print the current loss
        
        print("Period %02d: %0.2f" % (period, training_log_loss))
        
        # add loss metrics to the list
        
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
        
    print("Model training finished")
    
    # plot loss metrics over periods
    
    plt.ylabel('LogLoss')
    plt.xlabel('Periods')
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="Training")
    plt.plot(validation_log_losses, label="Validation")
    plt.legend()
    
    print("Final LogLoss (on training data):   %0.2f" % training_log_loss)
    print("Final LogLoss (on validation data): %0.2f" % validation_log_loss)
    
    return linear_classifier
