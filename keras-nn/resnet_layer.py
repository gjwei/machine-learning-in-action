#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/9/28
  
"""
from keras.layers import Input, Conv2D
import keras


x = Input(shape=(3, 256, 256))
y = Conv2D(64, (3, 3), 1, padding='valid')

z = keras.layers.add([x, y])

