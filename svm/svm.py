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
        self.alphas = None
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
        iters = 0
        while iters < self.max_iters:
            iters += 1
            alpha_prev = np.copy(self.alpha)
        
            for j in range(self.n_samples):
                # Pick random i
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
            if diff < self.tol:
                break
        logging.info('Convergence has reached after %s.' % iters)
    
        # Save support vectors index
        self.sv_idx = np.where(self.alpha > 0)[0]