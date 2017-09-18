#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/18/17
  
"""
import numpy as np
import base.base
from metrics.distance import euclidean_distance
from collections import Counter
# from scipy.spatial.distance import


class KNNbase(base.BaseEstimator):
    """k 近邻方法
    利用训练数据集，对特征向量进行空间划分，并作为分类模型。
    样本点最近的K个训练集决定了该样本点的类型
    三个要素：
        k值
        距离度量
        分类决策规则
    """
    def __init__(self, k=5, distance_fun=euclidean_distance):
        """
        Base class for NN classifier and regression
        :param k: int, default 5
                The number of neighbors to take into account.
                if 0, all the training example are used
                
        :param distance_fun: default euclidean distance
            A distance function taking two arguments. Any function from
            scipy.spatial.distance will do.
        """
        super(KNNbase, self).__init__()
        self.k = None if k == 0 else k  # l[:None] returns the whole list :)
        self.distance_fun = distance_fun
        
    def fit(self, X, y=None):
        self._process_input(X, y)
        
    def _predict(self, X=None):
        """对输入的X，计算他的最近k个邻接点"""
        return np.array([self._predict_label(x) for x in X])
        
    def _predict_label(self, x):
        """对单个点x, 计算最近k个邻接点"""
        distances = self.distance_fun(self.X, x)
        labels = self.y[np.argmax(distances)[0:self.k]]
        return labels
    
    def aggregate(self, labels):
        raise NotImplementedError()
    

class KnnClassifier(KNNbase):
    """
    Nearest neighbors classifier.

    Note: if there is a tie for the most common label among the neighbors, then
    the predicted label is arbitrary.
    """
    def aggregate(self, labels):
        """Return the most common target label."""
        return Counter(labels).most_common(1)[0][0]
    

class KnnRegeression(KNNbase):
    """
    Nearest neighbors regressor.
    """
    def aggregate(self, labels):
        """
        Return the mean of all targets.
        :param labels:
        :return:
        """
        return np.mean(labels)