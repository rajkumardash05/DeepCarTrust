# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:11:17 2020

@author: rajku
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.constraints import maxnorm
from sklearn.model_selection import GridSearchCV
import pdb

class Deep_Models:
    def __init__(self, name):
        self.name = name
        self.model = Sequential()
        self.estimators = []
        self.hidden_layers = [100, 100, 100]
        self.input_dm = 1
        self.activation_fn = ['relu']
        self.k_fold = 10
        self.loss = 'binary_crossentropy'
        self.optimizer = 'adam'
        self.model_type = 'classifier'
        self.no_of_output = 1
        self.metrics = ['accuracy']
        self.sample_weight = None
        self.dropout_spec = [0.2, 0.2, 0.2]

    def get_name(self):
        return self.name        
    
    def get_activation_fn(self, activation_fn, pos):
        if len(activation_fn) - 2 < pos:
            return activation_fn[-2]
        else:
            return activation_fn[pos]
        
    def update_parameters(self, param):
        self.hidden_layers = param['hidden_layers']
        self.input_dm = param['input_dm']
        self.activation_fn = param['activation_fn']
        self.k_fold = param['k_fold']
        self.loss = param['loss']
        self.optimizer = param['optimizer']
        self.model_type = param['model_type']
        self.no_of_output = param['no_of_output']
        self.metrics = param['metrics']
        self.sample_weight = param['sample_weight']
        self.dropout_spec = param['dropout_spec']
    
    def update_parameters_class_Weight(self, param):
        self.hidden_layers = param['hidden_layers']
        self.input_dm = param['input_dm']
        self.activation_fn = param['activation_fn']
        self.k_fold = param['k_fold']
        self.loss = param['loss']
        self.optimizer = param['optimizer']
        self.model_type = param['model_type']
        self.no_of_output = param['no_of_output']
        self.metrics = param['metrics']
        self.sample_weight = param['sample_weight']
        self.dropout_spec = param['dropout_spec']
        self.class_weight = param['class_weight']

            
    def build_model(self):
        
        for hl in self.hidden_layers:
            if self.hidden_layers.index(hl) == 0: # adding the very first hidden layer
                self.model.add(Dense(hl, input_dim=self.input_dm, kernel_initializer='normal', activation =self.activation_fn[0]))
            else:
                self.model.add(Dense(hl, kernel_initializer='normal', activation = self.get_activation_fn(self.activation_fn, self.hidden_layers.index(hl))))
                self.model.add(Dropout(self.dropout_spec[self.hidden_layers.index(hl)]))
                
        self.model.add(Dense(self.no_of_output, kernel_initializer='normal', activation = self.activation_fn[-1]))          
        
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics = self.metrics)             
        
        return self.model    
    
    def tune_model_Dropout(self, dropout_rate=0.0, weight_constraint=0):
        for hl in self.hidden_layers:
            if self.hidden_layers.index(hl) == 0: # adding the very first hidden layer
                self.model.add(Dense(hl, input_dim=self.input_dm, kernel_initializer='normal', activation =self.activation_fn[0], kernel_constraint=maxnorm(weight_constraint)))
                self.model.add(Dropout(dropout_rate))
            else:
                self.model.add(Dense(hl, kernel_initializer='normal', activation = self.get_activation_fn(self.activation_fn, self.hidden_layers.index(hl))))                
                self.model.add(Dropout(dropout_rate))
                
        self.model.add(Dense(self.no_of_output, kernel_initializer='normal', activation = self.activation_fn[-1]))          
        
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics = self.metrics)             
        
        return self.model    
        

    def build_estimator(self):
        self.estimators.append(('standardize', StandardScaler()))
        
        if self.model_type == 'regressor':
            self.estimators.append(('mlp', KerasRegressor(build_fn=self.build_model, epochs=50, batch_size=5, verbose=0)))
        elif self.model_type == 'classifier':
            self.estimators.append(('mlp', KerasClassifier(build_fn=self.build_model, epochs=200, batch_size=5, verbose=0)))
                    
        self.pipeline = Pipeline(self.estimators)
        
    def kfold_CVS(self, X, Y):
        kfold = KFold(n_splits=10)
        results = cross_val_score(self.pipeline, X, Y, cv=kfold)
        print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))    
    
    def DNN_Models(self, X, Y, **param):

        self.update_parameters(param)
        self.build_estimator()
       # pdb.set_trace()
        if self.k_fold > 0:
            self.kfold_CVS(X, Y)
        print(self.sample_weight)
        self.pipeline.fit(X, Y, **{'mlp__sample_weight': self.sample_weight})
        
        return self.pipeline
    
    def DNN_Models_Class_Weight(self, X, Y, **param):

        self.update_parameters(param)
        self.build_estimator()
       # pdb.set_trace()
        if self.k_fold > 0:
            self.kfold_CVS(X, Y)
        print(self.sample_weight)
        self.pipeline.fit(X, Y, **{'mlp__class_weight': self.sample_weight})
        
        return self.pipeline
        
    def DNN_Models_Tuner(self, X, Y, weight_constraint, dropout_rate, **param):
       self.update_parameters(param)
       model = KerasClassifier(build_fn=self.tune_model_Dropout, epochs=100, batch_size=10, verbose=0)
       # define the grid search parameter
       weight_constraint = weight_constraint
       dropout_rate = dropout_rate
       #pdb.set_trace()
       param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
       
       grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
       
       grid_result = grid.fit(X, Y)
    # summarize results
       print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
       means = grid_result.cv_results_['mean_test_score']
       stds = grid_result.cv_results_['std_test_score']
       params = grid_result.cv_results_['params']
       for mean, stdev, param in zip(means, stds, params):
           print("%f (%f) with: %r" % (mean, stdev, param))
        