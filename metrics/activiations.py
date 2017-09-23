#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/19/17
  
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    """Avoid overflow by removing max"""
    e = np.exp(x - np.amax(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

