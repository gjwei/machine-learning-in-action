#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/18/17
  
"""
import numpy as np


def distance(x1, x2, p):
    return np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1.0 / p)


def euclidean_distance(x1, x2):
    return distance(x1, x2, 2)


def manhattan_distance(x1, x2):
    return distance(x1, x2, 1)


