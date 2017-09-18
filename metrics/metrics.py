#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/17/17
  
"""

import autograd.numpy as np
# import numpy as np

EPS = 1e-15


def squared_error(actual, predicted):
    return (actual - predicted) ** 2


def mean_squared_error(actual, predicted):
    return np.mean(squared_error(actual, predicted))


def binary_crossentropy(actual, predicted):
    predicted = np.clip(predicted, EPS, 1 - EPS)
    return np.mean(
            -np.sum(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
    )
    

# aliases
mse = mean_squared_error
# rmse = root_mean_squared_error
# mae = mean_absolute_error


def get_metric(name):
    """Return metric function by name"""
    try:
        return globals()[name]
    except:
        raise ValueError('Invalid metric function.')
