#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/17/17
  
"""
import numpy as np


class BaseEstimator(object):
    """实现出机器学习的算法框架
    其中包括了：
    1. 对数据的预处理
    2. fit
    3. predict
    """
    
    def __init__(self):
        self.X = None
        self.y = None
        self.y_required = True  # 是够需要y label
        self.fit_required = True    # 模型是否需要fit数据
        
    def _process_input(self, X, y=None):
        """对数据进行预处理过程"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.size == 0:
            raise ValueError("Number of features must be > 0")
        
        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])
            
        self.X = X
        
        if self.y_required:
            if y is None:
                raise ValueError("Missed required argument y")
            
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            
            if y.size == 0:
                raise ValueError("number of target must be > 0")
            
        self.y = y
        
    def fit(self, X, y=None):
        """Fit the data"""
        self._process_input(X, y)
        
    def predict(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError("You must call 'fit' before 'predict'")
        
    def _predict(self, X=None):
        raise NotImplementedError()
