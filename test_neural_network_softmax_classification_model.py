# -*- coding: utf-8 -*-
'''
Created on Tue Oct 16 15:38:22 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    October 2018
    
@description:
    
    This function is used to test the pretrained neural network softmax classification
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
import seaborn as sns

#%% define function

def test_neural_network_softmax_classification_model(
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
        testing_features,
        testing_targets
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
            n_classes = n_classes,
            hidden_units = hidden_units,
            weight_column = weight_column,
            optimizer = my_optimiser,
            activation_fn = tf.nn.relu,
            dropout = dropout,
            batch_norm = batch_norm)
    
    # define input functions
    
    predict_testing_input_fn = lambda: my_input_fn(
      testing_features, 
      testing_targets, 
      num_epochs = 1, 
      shuffle = False)

    # calculate testing predictions
    
    testing_predictions = list(dnn_classifier.predict(input_fn = predict_testing_input_fn))
    testing_probabilities = np.array([item['probabilities'] for item in testing_predictions])    
    testing_pred_class_id = np.array([item['class_ids'][0] for item in testing_predictions])
    testing_pred_one_hot = tf.keras.utils.to_categorical(testing_pred_class_id, n_classes)  
    
    # calculate loss
    
    testing_log_loss = metrics.log_loss(testing_targets, testing_pred_one_hot)
        
    # Calculate final predictions (not probabilities, as above)
    
    final_testing_predictions = dnn_classifier.predict(input_fn = predict_testing_input_fn)
    final_testing_predictions = np.array([item['class_ids'][0] for item in final_testing_predictions])
    
    # calculate accuracy
    
    testing_accuracy = metrics.accuracy_score(testing_targets, final_testing_predictions)
    
    print('Final accuracy (on testing data): %0.2f' % testing_accuracy)
        
    # plot and save confusion matrix (testing)
    
    plt.figure(figsize = (6, 4))
    
    cm = metrics.confusion_matrix(testing_targets, final_testing_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap = 'bone_r')
    ax.set_aspect(1)
    #plt.title('Confusion matrix (testing)')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if model_dir is not None:
        plt.savefig(model_dir + '\\' + 'confm_testing.eps', dpi = 600, format = 'eps',
                    bbox_inches = 'tight', pad_inches = 0)
        plt.savefig(model_dir + '\\' + 'confm_testing.pdf', dpi = 600, format = 'pdf',
                    bbox_inches = 'tight', pad_inches = 0)
        plt.savefig(model_dir + '\\' + 'confm_testing.png', dpi = 600, format = 'png',
                    bbox_inches = 'tight', pad_inches = 0)
    
    # display final errors
    
    print('Final LogLoss (on testing data): %0.2f' % testing_log_loss)
    
    # convert outputs to pandas DataFrame
    
    final_testing_predictions = pd.DataFrame(final_testing_predictions, columns = ['Class'], 
                                            index = testing_targets.index, dtype = float)
    
    return dnn_classifier, final_testing_predictions