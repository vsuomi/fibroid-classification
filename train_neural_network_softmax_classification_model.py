# -*- coding: utf-8 -*-
'''
Created on Tue Sep 18 09:33:40 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    September 2018
    
@description:
    
    This code defines a multi-class neural network classification model 
    which is called to train on selected features
    
'''

#%% import necessary packages

from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import tensorflow as tf
from my_input_fn import my_input_fn
from construct_feature_columns import construct_feature_columns
import pandas as pd
import seaborn as sns

#%% define function

def train_neural_network_softmax_classification_model(
        learning_rate, 
        steps, 
        batch_size, 
        hidden_units,
        n_classes,
        weight_column,
        dropout,
        batch_norm,
        optimiser,
        model_dir,
        training_features,
        training_targets,
        validation_features,
        validation_targets
        ):
    
    '''
    Args:
        learning rate: the learning rate (float)
        steps: total number of training steps (int)
        batch_size: batch size to used to calculate the gradient (int)
        hidden_units: number of neurons in each layrs (list)
        n_classes: number of classes (int)
        weight_column: down weight or boost examples during training for unbalanced sets
        dropout: the probability to drop out a node output (for regularisation)
        batch_norm: to use batch normalization after each hidden layer (True/False)
        optimiser: type of the optimiser (GradientDescent, ProximalGradientDescent, Adagrad, ProximalAdagrad, Adam)
        model_dir: directory to save the checkpoint ('None' if no saving)
        training_features: one or more columns of training features (DataFrame)
        training_targets: a single column of training targets (DataFrame)
        validation_features: one or more columns of validation features (DataFrame)
        validation_targets: a single column of validation targets (DataFrame)
        
    Returns:
        A `DNNClassifier` object trained on the training data
    '''
    
    # define periods
    
    periods = 10
    steps_per_period = steps / periods
    
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
            feature_columns = construct_feature_columns(training_features),
            model_dir = model_dir,
            n_classes = n_classes,
            hidden_units = hidden_units,
            weight_column = weight_column,
            optimizer = my_optimiser,
            activation_fn = tf.nn.relu,
            dropout = dropout,
            batch_norm = batch_norm)
    
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
    
    print('Model training started')
    print('LogLoss on training data:')
    
    training_log_losses = []
    validation_log_losses = []

    for period in range (0, periods):
               
        # train the model
               
        dnn_classifier.train(
                input_fn = training_input_fn,
                steps = steps_per_period
                )
                
        # compute predictions
        
        training_predictions = list(dnn_classifier.predict(input_fn = predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, n_classes)
        
        validation_predictions = list(dnn_classifier.predict(input_fn = predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])    
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, n_classes)  
        
        # calculate losses
        
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        
        # print the current loss
        
        print('Period %02d: %0.2f' % (period, training_log_loss))
        
        # add loss metrics to the list
        
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
        
    print('Model training finished')
    
    # Calculate final predictions (not probabilities, as above)
    
    final_training_predictions = dnn_classifier.predict(input_fn = predict_training_input_fn)
    final_training_predictions = np.array([item['class_ids'][0] for item in final_training_predictions])
    
    final_validation_predictions = dnn_classifier.predict(input_fn = predict_validation_input_fn)
    final_validation_predictions = np.array([item['class_ids'][0] for item in final_validation_predictions])
    
    # calculate accuracy
    
    training_accuracy = metrics.accuracy_score(training_targets, final_training_predictions)
    validation_accuracy = metrics.accuracy_score(validation_targets, final_validation_predictions)
    
    print('Final accuracy (on training data): %0.2f' % training_accuracy)
    print('Final accuracy (on validation data): %0.2f' % validation_accuracy)
    
    # plot and save loss metrics over periods
    
    plt.figure(figsize = (6, 4))
    
    plt.xlabel('Periods')
    plt.ylabel('LogLoss')
    #plt.title('LogLoss vs. Periods')
    plt.tight_layout()
    plt.grid()
    plt.plot(training_log_losses, label = 'Training')
    plt.plot(validation_log_losses, label = 'Validation')
    plt.legend()
    
    if model_dir is not None:
        plt.savefig(model_dir + '\\' + 'LogLoss.eps', dpi = 600, format = 'eps',
                    bbox_inches = 'tight', pad_inches = 0)
        plt.savefig(model_dir + '\\' + 'LogLoss.pdf', dpi = 600, format = 'pdf',
                    bbox_inches = 'tight', pad_inches = 0)
        plt.savefig(model_dir + '\\' + 'LogLoss.png', dpi = 600, format = 'png',
                    bbox_inches = 'tight', pad_inches = 0)
    
    # plot and save confusion matrix (training)
    
    plt.figure(figsize = (6, 4))
    
    cm = metrics.confusion_matrix(training_targets, final_training_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap = 'bone_r')
    ax.set_aspect(1)
    #plt.title('Confusion matrix (training)')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if model_dir is not None:
        plt.savefig(model_dir + '\\' + 'confm_training.eps', dpi = 600, format = 'eps',
                    bbox_inches = 'tight', pad_inches = 0)
        plt.savefig(model_dir + '\\' + 'confm_training.pdf', dpi = 600, format = 'pdf',
                    bbox_inches = 'tight', pad_inches = 0)
        plt.savefig(model_dir + '\\' + 'confm_training.png', dpi = 600, format = 'png',
                    bbox_inches = 'tight', pad_inches = 0)
        
    # plot and save confusion matrix (validation)
    
    plt.figure(figsize = (6, 4))
    
    cm = metrics.confusion_matrix(validation_targets, final_validation_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap = 'bone_r')
    ax.set_aspect(1)
    #plt.title('Confusion matrix (validation)')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if model_dir is not None:
        plt.savefig(model_dir + '\\' + 'confm_validation.eps', dpi = 600, format = 'eps',
                    bbox_inches = 'tight', pad_inches = 0)
        plt.savefig(model_dir + '\\' + 'confm_validation.pdf', dpi = 600, format = 'pdf',
                    bbox_inches = 'tight', pad_inches = 0)
        plt.savefig(model_dir + '\\' + 'confm_validation.png', dpi = 600, format = 'png',
                    bbox_inches = 'tight', pad_inches = 0)
    
    # display final errors
    
    print('Final LogLoss (on training data):   %0.2f' % training_log_loss)
    print('Final LogLoss (on validation data): %0.2f' % validation_log_loss)
    
    # convert outputs to pandas DataFrame
    
    final_training_predictions = pd.DataFrame(final_training_predictions, columns = ['Class'], 
                                          index = training_targets.index, dtype = float)
    final_validation_predictions = pd.DataFrame(final_validation_predictions, columns = ['Class'], 
                                            index = validation_targets.index, dtype = float)
    
    return dnn_classifier, final_training_predictions, final_validation_predictions
