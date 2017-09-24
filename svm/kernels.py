#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/24/17
  
"""
import numpy as np
import scipy.spatial.distance as dist


class Linear(object):
    def __init__(self):
        pass
    
    def __call__(self, x, y, *args, **kwargs):
        return np.dot(x, y.T)
    
    def __repr__(self):
        return "Linear kernel"
    
    
class Poly(object):
    def __init__(self, degree=3):
        self.degree = degree
        
    def __call__(self, x, y, *args, **kwargs):
        return np.dot(x, y.T) ** self.degree
    
    def __repr__(self):
        return "Poly kernel"
    
    
class RBF(object):
    def __init__(self, gamma):
        self.gamma = gamma
    
    def __call__(self, x, y, *args, **kwargs):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        
        return np.exp(-self.gamma * dist.cdist(x, y) ** 2).flatten()
    
    def __repr__(self):
        return "RBF kernel"