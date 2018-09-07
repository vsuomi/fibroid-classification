# -*- coding: utf-8 -*-
'''
Created on Fri Sep  7 09:30:00 2018

@author:
    
    Visa Suomi
    Turku University Hospital
    September 2018
    
@description:
    
    This function is used to save variables from workspace into a pickle file
    
'''

#%% import necessary packages

import pickle

#%% save or load variables

def save_load_variables(directory, variables, opt):
    
    '''
    Args:
        directory: directory to save the model
        variables: variables to save
        opt: whether save ('save') or load ('load') variables
        
    Returns:
        variables: saved variables
    '''
    
    if opt == 'save':
        
        pickle_out = open(directory + '\\' + 'variables.pickle', 'wb')
        pickle.dump(variables, pickle_out)
        pickle_out.close()
        
    elif opt == 'load':
        
        pickle_in = open(directory + '\\' + 'variables.pickle', 'rb')
        variables = pickle.load(pickle_in)
        return variables
        
    else:
        
        print('Invalid option')