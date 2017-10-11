#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/9/28
  
"""
from keras.layer import Input, Dense
from keras.models import Model


inputs = Input(shape=(784, ))

x = Dense(128, activitions='relu', name='layer_1')(inputs)
x = Dense(256, activitions='relu', name='layer_2')(x)

predictions = Dense(10, activitions='softmax', name='output')(x)

model = Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='categroy_crossentropy',
              metrics=['accuracy'],
              )



