#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/24/17
  
"""
import numpy as np
from base.base import BaseEstimator
from kernels import Linear

class SVM(BaseEstimator):
    def __init__(self, C=1.0, kernel=None, tolerance=10e-3, max_iters=100):
        # super.__init__(SVM, self).__init__()
        self.C = C
        self.kernel = Linear if kernel is None else kernel
        self.tolerance = tolerance
        self.max_iters = max_iters
        
        self.b = 0
        self.alpha = None
        self.kernel_matrix = None

    def fit(self, X, y=None):
        self._process_input(X, y)
        self.K = np.zeros((self.n_samples, self.n_samples))
    
        # for i in range(self.n_samples):
        #     self.K[:, i] = self.kernel(self.X, self.X[i, :])
        #
        self.kernel_matrix = self.kernel(self.X, self.X)
        
        self.alpha = np.zeros(self.n_samples)  # 拉格朗日算子
        self.suppert_vector_index = np.arange(0, self.n_samples)  # suppert_vector_index
        
        return self._train()
    
    def _train(self):
        """
        训练SVM，使用SMO算法进行训练，详情见：
       ` http://bitjoy.net/2016/05/02/svm-smo-algorithm/`
       https://zh.wikipedia.org/wiki/%E5%BA%8F%E5%88%97%E6%9C%80%E5%B0%8F%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95
       
        
        """
        iters = 0
        while iters < self.max_iters:
            iters += 1
            alpha_prev = np.copy(self.alpha)
        
            for j in range(self.n_samples):
                # Pick random i, 用于SMO的坐标梯度上升
                i = self.random_index(j)
            
                eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                if eta >= 0:
                    continue
                L, H = self._find_bounds(i, j)
            
                # Error for current examples
                e_i, e_j = self._error(i), self._error(j)
            
                # Save old alphas
                alpha_io, alpha_jo = self.alpha[i], self.alpha[j]
            
                # Update alpha
                self.alpha[j] -= (self.y[j] * (e_i - e_j)) / eta
                self.alpha[j] = self.clip(self.alpha[j], H, L)
            
                self.alpha[i] = self.alpha[i] + self.y[i] * self.y[j] * (alpha_jo - self.alpha[j])
            
                # Find intercept
                b1 = self.b - e_i - self.y[i] * (self.alpha[i] - alpha_jo) * self.K[i, i] - \
                     self.y[j] * (self.alpha[j] - alpha_jo) * self.K[i, j]
                b2 = self.b - e_j - self.y[j] * (self.alpha[j] - alpha_jo) * self.K[j, j] - \
                     self.y[i] * (self.alpha[i] - alpha_io) * self.K[i, j]
                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = 0.5 * (b1 + b2)
        
            # Check convergence
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tolerance:
                break
        print('Convergence has reached after %s.' % iters)
    
        # Save support vectors index
        self.sv_idx = np.where(self.alpha > 0)[0]
        
    def random_index(self, j):
        i = j
        while i == j:
            i = np.random.randint(0, self.n_samples - 1)
        return i
    
    def _error(self, i):
        return self._predict_row(self.X[i]) - self.y[i]
        
    def _predict_row(self, x):
        """对单个数据进行预测
        f(x) = alpha* y * k(x, X) + b
        详情见西瓜书P127
        """
        k_v = self.kernel(self.X[self.sv_idx], x)
        return np.dot((self.alpha[self.sv_idx] * self.y[self.sv_idx]).T, k_v.T) + self.b
        
    def _find_bounds(self, i, j):
        """Find L and H such that L <= alpha <= H.
        Also, alpha must satisfy the constraint 0 <= αlpha <= C.
        """
        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C - self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        return L, H
    
    def clip(self, alpha, H, L):
        """
        alpha需要满足0 < a < C的要求
        :param alpha:
        :param H:
        :param L:
        :return:
        """
        if alpha > H:
            return H
        elif alpha < L:
            return L
        return alpha
        