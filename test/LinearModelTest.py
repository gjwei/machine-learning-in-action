#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/18/17
  
"""
from datasets import datasets
from linearregression import LogisticRegression, LinearRegression

(X_train, X_test, y_train, y_test) = datasets.load_mnist()

logistic_regression = LogisticRegression(lr=0.01, penalty='l1')
logistic_regression.fit(X_train, y_train)

# print(logistic_regression.)


