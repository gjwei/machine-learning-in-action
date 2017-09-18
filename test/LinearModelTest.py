#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/18/17
  
"""
from datasets import datasets
from linearregression import LogisticRegression, LinearRegression

(X_train, X_test, y_train, y_test) = datasets.load_mnist()
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1)


linear_regression = LinearRegression(lr=1e-4, penalty='l2', C=1e-5, max_iters=1000)
linear_regression.fit(X_train[:1000], y_train[:1000])

# logistic_regression = LogisticRegression(lr=1e-4, penalty='l1', C=1e-4, max_iters=5000)
# logistic_regression.fit(X_train[:1000], y_train[:1000])

# print(logistic_regression.)


