# -*- coding: utf-8 -*-
'''
Created on Thu Jan 10 13:44:06 2019

@author:
    
    Visa Suomi
    Turku University Hospital
    January 2019
    
@description:
    
    This function is used to build a Keras model using the input parameters
    
'''

#%% import necessary packages

import keras as k

#%% define function

def build_keras_model(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], optimiser = 'adam', 
                      learning_rate = 0.001, n_neurons = 30, n_layers = 1, n_classes = 3,
                      l1_reg = 0.001, l2_reg = 0.001, batch_norm = False, dropout = None, 
                      input_shape = (8,)):

    model = k.models.Sequential()
    
    model.add(k.layers.Dense(n_neurons, 
                             input_shape = input_shape,
                             kernel_regularizer = k.regularizers.l1_l2(l1 = l1_reg, l2 = l2_reg),
                             activation = 'relu'))
    if batch_norm is True:
        model.add(k.layers.BatchNormalization())
    if dropout is not None:
        model.add(k.layers.Dropout(dropout))
        
    i = 1   
    while i < n_layers:
        model.add(k.layers.Dense(n_neurons,
                                 kernel_regularizer = k.regularizers.l1_l2(l1 = l1_reg, l2 = l2_reg),
                                 activation = 'relu'))
        if batch_norm is True:
            model.add(k.layers.BatchNormalization())
        if dropout is not None:
            model.add(k.layers.Dropout(dropout))
        i += 1
    del i
    
    model.add(k.layers.Dense(n_classes, activation = 'softmax'))
    
    if optimiser == 'adam':
        koptimiser = k.optimizers.Adam(lr = learning_rate)
    elif optimiser == 'adamax':
        koptimiser = k.optimizers.Adamax(lr = learning_rate)
    elif optimiser == 'nadam':
        koptimiser = k.optimizers.Nadam(lr = learning_rate)
    else:
        print('Unknown optimiser type')
    
    model.compile(optimizer = koptimiser, loss = loss, metrics = metrics)
    
    model.summary()
    
    return model
