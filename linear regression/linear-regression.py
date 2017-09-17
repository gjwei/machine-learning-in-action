#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/16/17
  
"""
import numpy as np

class LR(object):
    
    def __init__(self, alpha, n_iter):
        self.alpha = alpha
        self.errors = []
        self.n_iter = n_iter
        self.weight = [0]
    
    
    def fit(self, X, y):
        """Fitting the data
        Parameters:
            @:param X: the input data, shape (m_samples, n_features)
            @:param y: the label of input data, shape (m_sample, )
            
        return:
            self
            
        """
        self.weight = np.zeros((X.shape[1] + 1, 1))
        
        for iter in range(self.n_iter):
            error = 0
            for i in range(len(X)):
                error += (self.lost_function(X[i], y[i]))
                lost = (self.output(X[i]) - y[i]) * X[i]  # 损失函数的导数值(1, n+1)
                self.weight -= (self.alpha * lost)  # 没经过一个数据都要更新一次weight
            self.errors.append(error)
            print("{i}the iteration the lost is {lost}".format(i=iter, lost=self.errors[-1]))
        return self
        
    def lost_function(self, x, y):
        """Calculate lost function"""
        return (self.output(x) - y) ** 2 / 2.0
        
        
    def output(self, x):
        """Calculate the h(x)"""
        return np.dot(x, self.weight[1:]) + self.weight[0]
    
    def predcit(self, x):
        return np.where(self.output(x) > 0, 1, -1)
    