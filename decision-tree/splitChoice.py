#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/24/17
  
"""
import numpy as np
from scipy import stats


def f_entropy(p):
    """
    计算信息熵
    :param p:
    :return:
    """
    p = np.bincount(p) / float(len(p))
    
    ep = 0
    for p_value in p:
        if p_value == 0:
            continue
        ep -= p_value * np.log(p_value)
    return ep


def information_gain(y, splits):
    """
    获取按照某一个属性a的某个值Value分割之后的数据的信息增益
    :param y: 训练集中的label
    :param splits: 将原有的label按照某个属性的属性值切分之后的结果
    :return: 信息增益，越大说明属性a进行划分的增益越大
    """
    splits_entropy = np.sum(f_entropy(split) * len(split) / float(len(y)) for split in splits)
    return f_entropy(y) - splits_entropy


def mse_criterion(y, splits):
    """
    对于回归树，要用平方误差最小化准则
    :param y:
    :param splits:
    :return:
    """
    y_mean = np.mean(y)
    return -sum([np.sum((split - y_mean) ** 2) * (float(split.shape[0]) / y.shape[0]) for split in splits])


def xgb_criterion(y, left, right, loss):
    left = loss.gain(left['actual'], left['y_pred'])
    right = loss.gain(right['actual'], right['y_pred'])
    initial = loss.gain(y['actual'], y['y_pred'])
    gain = left + right - initial
    return gain




