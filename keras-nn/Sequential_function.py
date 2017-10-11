#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/9/28
  
"""
from pprint import pprint
import json
from keras.utils import plot_model
from keras.datasets import mnist

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, LocallyConnected2D

from keras.models import Sequential

model = Sequential()
model.add(LocallyConnected2D(64, (3, 3), activation='relu', padding='valid', input_shape=(28, 28, 1), name='conv_1'))
model.add(Conv2D(128, (3, 3),activation='relu', padding='same', name='conv_2'))
model.add(MaxPool2D((2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_3'))
model.add(Conv2D(512, (3, 3), activation='relu', name='conv_4'))

model.add(GlobalAveragePooling2D())

# model.add(Flatten())

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'], )
model.summary()
# print("layers", model.layers)
# for layer in model.layers:
    # print(layer.name)
weights = model.get_weights()
# for i in range(len(weights)):
#     print(weights[i])
json_string = model.to_json()
# print(model.get_layer(name='conv_1'))

plot_model(model, to_file='model.png ')
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
#
# model.fit(x=x_train, y=y_train, batch_size=batch_size,
#           epochs=epochs, verbose=1, callbacks=None, validation_split=0.1,
#           shuffle=True, )