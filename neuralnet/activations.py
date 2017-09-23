#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/22/17
  
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def identity(z):
    return z


def binary_step(z):
    return np.where(z < 0, 0, 1)
    

def logistic(z):
    return 1.0 / (1 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def arctan(z):
    return np.arctan(z)


def softsign(z):
    return z / (1.0 + np.abs(z))


def relu(z):
    return np.maximum(1, z)


def leakyRelu(z):
    # return np.where(z < 0, 0.01 * z, z)
    return np.maximum(0.01 * z, z)
    

def pRelu(z, alpha):
    return np.where(z < 0, alpha * z, z)


def RLeakyRelu(z, alpha):
    return np.where(z < 0, alpha * z, z)


def exponential_linear_unit(z, alpha):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)


def scaled_exponential_Lu(z):
    alpha = 1.67326
    lambda_ = 1.0507
    return np.where(z < 0, lambda_ * alpha * (np.exp(z) - 1), lambda_ * z)


def adaptive_piecewise_linear(z):
    pass


def softPlus(x):
    return np.log1p(x)


def bent_identity(x):
    return ((np.sqrt(x ** 2 + 1) - 1) / 2) + x


def softExp(x, a):
    return np.where(a < 0, -1 * (np.log(1 - a * (x + a))) / a, np.where(a == 0, x, np.exp(a * x) - 1) / a) + a


def sinusoid(x):
    return np.sin(x)


def sinc(x):
    return np.where(x == 0, 1, np.sin(x) / x)


def gaussian(x):
    return np.exp(- x ** 2)


    
    





