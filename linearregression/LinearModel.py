#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/17/17
  
"""
import logging

import autograd.numpy as np
from autograd import grad
# import numpy as np

from base import BaseEstimator
from metrics.metrics import mean_squared_error, binary_crossentropy

np.random.seed(1000)


class BasicRegression(BaseEstimator):
    
    def __init__(self, lr=0.001, penalty=None, C=0.01, max_iters=1000, tolerance=1e-4):
        """Basic class for implementing continuous regression estimators which
        are trained with gradient descent optimization on their particular loss
        function.

        Parameters
        ----------
        lr : float, default 0.001
            Learning rate.
        penalty: str, {"l1", "l2"}, default None
            Regularization function name
        
        C: float
            the regularization coefficient
        
        max_iters : int, default 10000
            The maximum number of iterations.
        """
        super(BasicRegression, self).__init__()
        self.lr = lr
        self.penalty = penalty
        self.C = C
        self.max_iters = max_iters
        self.errors = []
        self.theta = []
        self.n_samples, self.n_features = None, None
        self.cost_fun = None
        self.tolerance = tolerance
        
    def _loss(self, w):
        raise NotImplementedError()
    
    def init_cost(self):
        """指定好loss function"""
        raise NotImplementedError()
    
    def _cost(self, X, y, theta):
        """计算损失"""
        prediction = X.dot(theta)
        error = self.cost_fun(y, prediction)
        
        return error
    
    def fit(self, X, y=None):
        self._process_input(X, y)
        self.init_cost()
        self.n_samples, self.n_features = X.shape[0], X.shape[1]
        
        # Initialize weights + bias term
        self.theta = np.random.normal(size=(self.n_features + 1), scale=0.5)
        
        # add a intercept column
        self.X = self._add_intercept(self.X)
        
        self._train()
    
    @staticmethod
    def _add_intercept(X):
        b = np.ones([X.shape[0], 1])
        # print(b.shape, X.shape)
        
        return np.concatenate([b, X], axis=1)
    
    def _add_penalty(self, loss, w):
        """Apply regulation to the loss"""
        if self.penalty == 'l1':
            loss += self.C * np.abs(w[:-1]).sum()
        elif self.penalty == 'l2':
            loss += (0.5 * self.C) * (w[:-1] ** 2).mean()
        return loss
        
    def _train(self):
        self.theta,  self.errors = self._gradient_descent()
        # print("Theta: %s" % self.theta.flatten())
        
    def _predict(self, X=None):
        """计算输出"""
        X = self._add_intercept(X)
        return X.dot(self.theta)
    
    def _gradient_descent(self):
        """求解模型的梯度下降，使用了grad这个函数进行求解"""
        theta = self.theta
        errors = [self._cost(self.X, self.y, theta)]
        
        for i in range(1, self.max_iters + 1):
            # get the derivative of the loss function
            cost_d = grad(self._loss)
            # calculate gradient and update theta
            delta = cost_d(theta)
            # print(delta)
            
            theta -= self.lr * delta
            
            errors.append(self._cost(self.X, self.y, theta))
            print("Iteration %s, error %s" % (i, errors[i]))
            
            error_diff = np.linalg.norm(errors[i - 1] - errors[i])
            
            # if abs(error_diff) < self.tolerance:
            #     print('Convergence has reached.')
            #     break
                
        return theta, errors
    
    
class LogisticRegression(BasicRegression):
    """Binary logistic regression with gradient descent optimizer"""
    def init_cost(self):
        self.cost_fun = binary_crossentropy
        
    def _loss(self, w):
        loss = self.cost_fun(self.y,
                             self.sigmoid(np.dot(self.X, w)))
        # print(self.sigmoid(np.dot(self.X, w)))
        return loss
    
    @staticmethod
    def sigmoid(x):
        return 0.5 * (np.tanh(x) + 1)
    
    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))
    
    def predict(self, X=None):
        X = self._add_intercept(X)
        return self.sigmoid(X.dot(self.theta))
    

class LinearRegression(BasicRegression):
    """Linear regression"""
    def init_cost(self):
        self.cost_fun = mean_squared_error
    
    def _loss(self, w):
        loss = self.cost_fun(self.y, np.dot(self.X, w))
        return loss
    
    
        
    
            
        
    
        
        
    


