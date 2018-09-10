# -*- coding: utf-8 -*-
'''
Created on Mon Sep 10 09:16:51 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    September 2018
    
@description:
    
    This function is used to test the pretrained neural network classification
    model. Note: the arguments have to be the same as in the pretrained model
    stored in model_dir.
    
'''

#%% import necessary packages

from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import tensorflow as tf
from my_input_fn import my_input_fn
from construct_feature_columns import construct_feature_columns
import pandas as pd

#%% define function

def test_neural_network_classification_model(
        learning_rate, 
        steps, 
        batch_size, 
        hidden_units,
        weight_column,
        dropout,
        batch_norm,
        optimiser,
        model_dir,
        testing_features,
        testing_targets
        ):
    
    '''
    Args:
        learning rate: the learning rate (float)
        steps: total number of training steps (int)
        batch_size: batch size to used to calculate the gradient (int)
        hidden_units: number of neurons in each layrs (list)
        weight_column: down weight or boost examples during training for unbalanced sets
        dropout: the probability to drop out a node output (for regularisation)
        batch_norm: to use batch normalization after each hidden layer (True/False)
        optimiser: type of the optimiser (GradientDescent, ProximalGradientDescent, Adagrad, ProximalAdagrad, Adam)
        model_dir: directory where the pretrained model is saved
        testing_features: one or more columns of testing features (DataFrame)
        testing_targets: a single column of testing targets (DataFrame)
        
    Returns:
        A `DNNClassifier` object trained on the training data
    '''
    
    # create neural network classifier object
    
    if optimiser == 'GradientDescent':
        my_optimiser = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    elif optimiser == 'ProximalGradientDescent':
        my_optimiser = tf.train.ProximalGradientDescentOptimizer(learning_rate = learning_rate)
        #my_optimiser = tf.train.ProximalGradientDescentOptimizer(learning_rate = learning_rate, l1_regularization_strength = 0.1) # for L1 regularisation
        #my_optimiser = tf.train.ProximalGradientDescentOptimizer(learning_rate = learning_rate, l2_regularization_strength = 0.1) # for L2 regularisation
    elif optimiser == 'Adagrad':
        my_optimiser = tf.train.AdagradOptimizer(learning_rate = learning_rate) # for convex problems
    elif optimiser == 'ProximalAdagrad':
        my_optimiser = tf.train.ProximalAdagradOptimizer(learning_rate = learning_rate) # for convex problems
        #my_optimiser = tf.train.ProximalAdagradOptimizer(learning_rate = learning_rate, l1_regularization_strength = 0.1) # for L1 regularisation
        #my_optimiser = tf.train.ProximalAdagradOptimizer(learning_rate = learning_rate, l2_regularization_strength = 0.1) # for L2 regularisation
    elif optimiser == 'Adam':
        my_optimiser = tf.train.AdamOptimizer(learning_rate = learning_rate) # for non-convex problems
    else:
        print('Unknown optimiser type')
    my_optimiser = tf.contrib.estimator.clip_gradients_by_norm(my_optimiser, 5.0)
    dnn_classifier = tf.estimator.DNNClassifier(
            feature_columns = construct_feature_columns(testing_features),
            model_dir = model_dir,
            n_classes = 2,
            hidden_units = hidden_units,
            weight_column = weight_column,
            optimizer = my_optimiser,
            activation_fn = tf.nn.relu,
            dropout = dropout,
            batch_norm = batch_norm)
    
    # define input function
    
    predict_testing_input_fn = lambda: my_input_fn(
      testing_features, 
      testing_targets, 
      num_epochs = 1, 
      shuffle = False)
    
    # calculate testing probabilities
            
    testing_probabilities = dnn_classifier.predict(input_fn = predict_testing_input_fn)
    testing_probabilities = np.array([item['probabilities'] for item in testing_probabilities])
    
    # calculate loss
    
    testing_log_loss = metrics.log_loss(testing_targets, testing_probabilities)
    
    # get just the probabilities for the positive class
    
    testing_probabilities = testing_probabilities[:, 1]
    
    # calculate and plot ROC curves
    
    testing_false_positive_rate, testing_true_positive_rate, testing_thresholds = metrics.roc_curve(
            testing_targets, testing_probabilities)
    
    plt.subplot(1, 2, 2)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC')
    plt.tight_layout()
    plt.grid()
    plt.plot(testing_false_positive_rate, testing_true_positive_rate, label = 'Testing')
    plt.plot([0, 1], [0, 1], color = 'k')
    plt.legend()
    
    # display final errors
    
    print('LogLoss (on testing data): %0.2f' % testing_log_loss)
    
    # calculate and print evaluation metrics
    
    testing_evaluation_metrics = dnn_classifier.evaluate(input_fn = predict_testing_input_fn)

    print('AUC (on testing data): %0.2f' % testing_evaluation_metrics['auc'])
    
    # convert outputs to pandas DataFrame
    
    testing_probabilities = pd.DataFrame(testing_probabilities, columns = ['Probability'], 
                                            index = testing_targets.index, dtype = float)
    
    return dnn_classifier, testing_probabilities
