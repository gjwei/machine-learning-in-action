#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/17/17
  
"""
import numpy as np


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta:float
        Learning rate (between 0.0 and 1.0)
    n_iter:int
        Passes over the training dataset.

    Attributes
    -------------
    w_: 1d-array
        Weights after fitting.
    errors_: list
        Numebr of misclassifications in every epoch.

    """
    def __init__(self, eta, n_iter):
        self.eta = self.eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        """Train data
        
        Parameters:
            @:param X: input data (n_sample, n_features)
            @:param y: target (n_sample, 1)
        
        Returns
            self
        """
        self.w_ = np.zeros(X.shape[1] + 1)
        self.errors = []
        
        for _ in range(self.n_iter):
            for x, target in zip(X, y):
            
        
    
    def predict(self, x):
        """Predict result"""
        return
        
        