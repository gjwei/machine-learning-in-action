#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/9/28
  
"""
import keras
from keras.layers import (Dense, Input,
                          Conv2D, Flatten,
                          MaxPool2D)
from keras.models import Model


inputs = Input(shape=(1, 27, 27))
x = Conv2D(64, (3, 3), activation='relu')(inputs)
x = Conv2D(64, (3, 3), activation='relu')(x)

out = Flatten()(x)

vision_model = Model(inputs=inputs, outputs=out)

digit_a = Input(shape=(1, 27, 27))
digit_b = Input(shape=(1, 27, 27))

out_a = vision_model(inputs=digit_a)
out_b = vision_model(inputs=digit_b)

concatenated = keras.layers.concatenate([out_a, out_b])

out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model(inputs=[digit_a, digit_b], outputs=out)





