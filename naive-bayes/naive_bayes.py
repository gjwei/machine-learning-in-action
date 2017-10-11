#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/18/17
  
"""
"""基于贝叶斯定理和特征条件独立假设分类方法"""

import numpy as np
from base.base import BaseEstimator
from metrics.activiations import softmax


class NaiveBayesClassifier(BaseEstimator):
    """
    Gaussian Naive Bayes
    Binary problem
    """
    n_class = 2

    def fit(self, X, y=None):
        """fit 我们的数据，得到先验概率，X的按照feature的平均值，方差"""
        self._process_input(X, y)
        
        self._means = np.zeros((self.n_class, self.n_features), dtype=np.float64)
        self._vars = np.zeros((self.n_class, self.n_features), dtype=np.float64)
        self._priors = np.zeros(self.n_class, dtype=np.float64)
        
        for c in range(self.n_class):
            X_c = X[y == c]
            
            self._means[c, :] = np.mean(X_c, axis=0)
            self._vars[c, :] = np.var(X_c, axis=0)
            # with laplacian correction
            self._priors[c] = (len(X_c) + 1) / (float(len(X)) + self.n_class)
            
    def _predict(self, X=None):
        """根据fit的结果，预测x的情况"""
        predictions = np.array([self._predict_x(x) for x in X])
        return softmax(predictions)
    
    def _predict_x(self, x):
        """预测一个数据的情况"""
        out = []
        
        for y in range(self.n_class):
            prior = np.log(self._priors[y])
            posterior = np.log(self.guassian(x, y)).sum()
            
            prediction = prior + posterior
            
            out.append(prediction)
        
        return out
    
    def guassian(self, x, c):
        """得到相应的高斯方程"""
        mean = self._means[c]
        var = self._vars[c]
        
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        
        return numerator / denominator