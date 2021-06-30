# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:46:43 2021

@author: Shashank Dwivedi
"""
import numpy as np
import pandas as pd
class LinearRegression(object):
    
    """
    This function is for Hardcoded Linear Regression algorithm
    using Gradient descent
    
    """
    
    def __init__(self):
        self.object = None
        self.accuracy = None
        self.weight = None
        
    def fit(self,x,y):
        data_set = x
        output = y
        cost = []
        num_instances , num_features = list(data_set.shape)
        self.weight = [np.random.randint(10) for _ in range(num_features + 1)]
        for instance , result in zip(data_set , output):
            arr = np.ndarray(instance)
            prediction = sum([arr[i]*weight[i]] for i in range(len(arr))) + weights[-1]
            error = result  - prediction
            for i in range(len(arr)):
                weight[i] = weight[i] - alpha * error * arr[i]
            weight[-1] = weight[-1] - alpha * error
            cost.append(error)
        
        print("model Trained")
    
    def predict(self,x):
        arr = np.ndarray(x)
        prediction = sum([arr[i]*weight[i] for i in range(len(arr))])
        return prediction
    
    def accuracy(self,x):
        error = []
        for instance, result in zip(data_set, output):
            y_hat = predict(instance)
            squared_error = (result - y_hat)**2
            error.append(squared_error)
        rmse_error = np.sqrt(sum(error)/len(error))
        self.accuracy = rmse_error
        return self.accuracy
            
            
        
        
        
        
        
        