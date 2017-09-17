#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/17/17
  
"""
import os
import numpy as np
import struct
from array import array as pyarray
from numpy import array, int8, uint8, zeros


def get_filename(name):
    """得到指定文件的路径"""
    if os.path.exists(os.path.join(os.path.dirname(__path__), name)):
        os.system('/data/get_mnist.sh')
    return os.path.join(os.path.dirname(__path__), name)

def load_mnist():
    
    def load(dataset='train', digits=np.arange(10)):
        if dataset == 'train':
            fname_img = get_filename('data/mnist/train-images-idx3-ubyte')
            fname_lbl = get_filename('data/mnist/train-labels-idx1-ubyte')
        elif dataset == "test":
            fname_img = get_filename('data/mnist/t10k-images-idx3-ubyte')
            fname_lbl = get_filename('data/mnist/t10k-labels-idx1-ubyte')
        
        with open(fname_lbl, 'rb') as flbl:
            magic_nr, size = struct.unpack(">II", flbl.read(8))
            lbl = pyarray("b", flbl.read())
        
        with open(fname_img, 'rb') as fimg:
            magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = pyarray("B", fimg.read())

        ind = [k for k in range(size) if lbl[k] in digits]
        N = len(ind)

        images = zeros((N, rows, cols), dtype=uint8)
        labels = zeros((N, 1), dtype=int8)
        for i in range(len(ind)):
            images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels

    X_train, y_train = load('train')
    X_test, y_test = load('test')

    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype(np.float32)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype(np.float32)

    return X_train, X_test, y_train, y_test

  
def load_nietzsche():
    text = open(get_filename('data/nietzsche.txt')).read().decode('utf-8').lower()
    chars = set(list(text))
    char_indices = {ch: i for i, ch in enumerate(chars)}
    indices_char = {i: ch for i, ch in enumerate(chars)}

    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
        
    return X, y, text, chars, char_indices, indices_char


