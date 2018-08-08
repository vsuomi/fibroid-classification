# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 09:31:40 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    August 2018
    
@description:
    
    This code defines a neural network classification model which is called to 
    train on selected features
    
"""

#%% import necessary packages

from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import tensorflow as tf
from my_input_fn import my_input_fn
from construct_feature_columns import construct_feature_columns
import pandas as pd

#%% define function

def train_neural_network_classification_model(
        learning_rate, 
        steps, 
        batch_size, 
        hidden_units,
        optimiser,
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
        optimiser: type of the optimiser (GradientDescent, Adagrad, Adam)
        training_features: one or more columns of training features (DataFrame)
        training_targets: a single column of training targets (DataFrame)
        calidation_features: one or more columns of validation features (DataFrame)
        validation_targets: a single column of validation targets (DataFrame)
        
    Returns:
        A `DNNClassifier` object trained on the training data
    """
    
    # define periods
    
    periods = 10
    steps_per_period = steps / periods
    
    # create neural network classifier object
    
    if optimiser == "GradientDescent":
        my_optimiser = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    elif optimiser == "ProximalGradientDescent":
        my_optimiser = tf.train.ProximalGradientDescentOptimizer(learning_rate = learning_rate)
        #my_optimiser = tf.train.ProximalGradientDescentOptimizer(learning_rate = learning_rate, l1_regularization_strength = 0.0) # for L1 regularisation
        #my_optimiser = tf.train.ProximalGradientDescentOptimizer(learning_rate = learning_rate, l2_regularization_strength = 0.0) # for L2 regularisation
    elif optimiser == "Adagrad":
        my_optimiser = tf.train.AdagradOptimizer(learning_rate = learning_rate) # for convex problems
    elif optimiser == "ProximalAdagrad":
        my_optimiser = tf.train.ProximalAdagradOptimizer(learning_rate = learning_rate) # for convex problems
        #my_optimiser = tf.train.ProximalAdagradOptimizer(learning_rate = learning_rate, l1_regularization_strength = 0.0) # for L1 regularisation
        #my_optimiser = tf.train.ProximalAdagradOptimizer(learning_rate = learning_rate, l2_regularization_strength = 0.0) # for L2 regularisation
    elif optimiser == "Adam":
        my_optimiser = tf.train.AdamOptimizer(learning_rate = learning_rate) # for non-convex problems
    else:
        print("Unknown optimiser type")
    my_optimiser = tf.contrib.estimator.clip_gradients_by_norm(my_optimiser, 5.0)
    dnn_classifier = tf.estimator.DNNClassifier(
            feature_columns = construct_feature_columns(training_features),
            n_classes = 2,
            hidden_units = hidden_units,
            optimizer = my_optimiser)
    
    # define input functions
    
    training_input_fn = lambda: my_input_fn(
      training_features, 
      training_targets, 
      batch_size = batch_size)
    predict_training_input_fn = lambda: my_input_fn(
      training_features, 
      training_targets, 
      num_epochs = 1, 
      shuffle = False)
    predict_validation_input_fn = lambda: my_input_fn(
      validation_features, 
      validation_targets, 
      num_epochs = 1, 
      shuffle = False)
    
    # print training progress
    
    print("Model training started")
    print("LogLoss on training data:")
    
    training_log_losses = []
    validation_log_losses = []

    for period in range (0, periods):
               
        # train the model
               
        dnn_classifier.train(
                input_fn = training_input_fn,
                steps = steps_per_period
                )
                
        # compute predictions
        
        training_probabilities = dnn_classifier.predict(input_fn = predict_training_input_fn)
        training_probabilities = np.array([item["probabilities"] for item in training_probabilities])
        
        validation_probabilities = dnn_classifier.predict(input_fn = predict_validation_input_fn)
        validation_probabilities = np.array([item["probabilities"] for item in validation_probabilities])
        
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
    
    plt.figure(figsize = (12, 4))
    
    plt.subplot(1, 2, 1)
    plt.xlabel("Periods")
    plt.ylabel("LogLoss")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.grid()
    plt.plot(training_log_losses, label = "Training")
    plt.plot(validation_log_losses, label = "Validation")
    plt.legend()
    
    # get just the probabilities for the positive class
    
    training_probabilities = training_probabilities[:, 1]
    validation_probabilities = validation_probabilities[:, 1]
    
    # calculate and plot ROC curves
    
    training_false_positive_rate, training_true_positive_rate, training_thresholds = metrics.roc_curve(
            training_targets, training_probabilities)

    validation_false_positive_rate, validation_true_positive_rate, validation_thresholds = metrics.roc_curve(
            validation_targets, validation_probabilities)
    
    plt.subplot(1, 2, 2)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC")
    plt.tight_layout()
    plt.grid()
    plt.plot(training_false_positive_rate, training_true_positive_rate, label = "Training")
    plt.plot(validation_false_positive_rate, validation_true_positive_rate, label = "Validation")
    plt.plot([0, 1], [0, 1], color = "k")
    plt.legend()
    
    # display final errors
    
    print("Final LogLoss (on training data):   %0.2f" % training_log_loss)
    print("Final LogLoss (on validation data): %0.2f" % validation_log_loss)
    
    # calculate and print evaluation metrics
    
    training_evaluation_metrics = dnn_classifier.evaluate(input_fn = predict_training_input_fn)
    validation_evaluation_metrics = dnn_classifier.evaluate(input_fn = predict_validation_input_fn)

    print("AUC (on training data): %0.2f" % training_evaluation_metrics["auc"])
    print("AUC (on validation data): %0.2f" % validation_evaluation_metrics["auc"])
    
    # convert outputs to pandas DataFrame
    
    training_probabilities = pd.DataFrame(training_probabilities, columns = ["Probability"], 
                                          index = training_targets.index, dtype = float)
    validation_probabilities = pd.DataFrame(validation_probabilities, columns = ["Probability"], 
                                            index = validation_targets.index, dtype = float)
    
    return dnn_classifier, training_probabilities, validation_probabilities
