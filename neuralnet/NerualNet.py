#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/21/17
  
"""
import numpy as np


class NeuralNet(object):
    """对神经网络模型进行设计"""
    
    def __init__(self, sizes):
        """
        初始化
        :param sizes: list类型： 存储每层神经元数目
                      sizes = [2, 3, 2] 表示输入层有两个神经元、
                      隐藏层有3个神经元以及输出层有2个神经元
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        self._initialization()
        
    def _initialization(self):
        """
        对神经网络的权值进行初始化
        a = w * x + b
        :return:
        """
        self.biases = [np.random.randn(neuron_size, 1) for neuron_size in self.sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(self.sizes[1:], self.sizes[:-1])]
        
    def feedforward(self, a):
        """
        前向传输
        :param a: 上个神经元的输出
        :return: 神经元的sigmoid值
        """
        self.activitions = [a]
        self.zs = []
        for w, b in zip(self.weights, self.biases):
            z = w * a + b
            a = self.sigmoid(z)
            self.activitions.append(a)
            self.zs.append(z)
        
        return a
    
    def SGD(self, train_data, learn_rate, epochs, batch_size=32, test_data=None):
        """
        随机梯度下降方法，用于更新权值
        :param train_data: 训练数据，(x, y)
        :param learn_rate: 学习速率
        :param epochs: 迭代周期数
        :param batch_size: batch size
        :param test_data: 测试数据
        :return:
        """
            
        n_train = len(train_data)
          
        for epoch in range(epochs):
            np.random.shuffle(train_data)
            
            batchs = [train_data[index, index+batch_size]
                      for index in range(0, n_train, batch_size)]
            
            for batch in batchs:
                self.update_weights(batch, learn_rate)
                
            if test_data:
                n_test = len(test_data)
                print("Epoch {0}: {1} / {2}".format(
                    epoch, self.evaluate(test_data), n_test
                ))
            else:
                print("Epoch {0} complete!".format(epoch))
                
    def update_weights(self, batch, learn_rate):
        """
        使用后向传播，进行
        :param batch:
        :param learn_rate:
        :return:
        """
        nable_bias = [np.zeros(b.shape) for b in self.biases]
        nable_weights = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in batch:
            nable_b, nable_w = self.backprop(x, y)
            nable_bias += nable_b
            nable_weights += nable_w
        
        self.weights -= learn_rate / len(batch) * nable_weights
        self.biases -= learn_rate / len(batch) * nable_bias
        
    def backprop(self, X, y):
        """
        实现后向传播
        :param X: 输入的训练数据
        :param y: 输出的label
        :return: 返回一个列表，列表中包含每一层中的cost函数对权值的偏导
        """
        n_bias = [np.zeros(b.shape) for b in self.biases]
        n_weights = [np.zeros(w.shape) for w in self.weights]
        
        # 前向传播，求出所有的神经元输入和输出,已经得到所有的中间结果，
        # 存储在activitions和zs中
        # 使用后向传播算法
        delta = self.cost_derivate(self.activitions[-1], y) * sigmoid_derivate(self.zs[-1])
        n_bias[-1] = delta
        n_weights[-1] = np.dot(delta, self.activitions[-2].T)
        
        for layer in range(2, self.num_layers):
            z = self.zs[-layer]
            sp = sigmoid_derivate(z)
            
            delta = np.dot(self.weights[-layer + 1].T, delta) * sp
            n_bias[-layer] = delta
            n_weights[-layer] = np.dot(delta, self.activitions[-layer - 1].T)
            
        return n_bias, n_weights
        
    def evaluate(self, X):
        """
        对测试数据进行评测
        :param X: 测试的数据(x_test, y_test)
        :return:
        """
        x_test, y_test = X
        test_output = np.argmax(self.feedforward(x_test))
        return np.sum(test_output == y_test)
        
    def cost_derivate(self, output, y):
        """
        计算代价函数的对上一层神经元输入的偏导
        :param output: NN网络的输出
        :param y: labels
        :return: 导数值
        """
        return output - y
        
        
        
        
        
        
        
        
                
            
        
def sigmoid(x):
    return 1.0 / (1 + np.exp(x))

def sigmoid_derivate(x):
    t = sigmoid(x)
    return t * (1 - t)


