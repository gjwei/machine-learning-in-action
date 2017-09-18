#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/18/17
  
"""
import numpy as np


class LinearClassifier(object):
    
    def __init__(self):
        self.W = None
        
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """Train the linear classification use SDG
        
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
        
        outputs:
        A list containing the value of the loss function at each training iteration
        """
        

