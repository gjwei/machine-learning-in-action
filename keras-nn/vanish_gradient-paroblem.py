#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/22/17
  
"""
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model


def nn_model(X, y, X_test, y_test,  batch_size, epochs):
    inputs = Input(shape=(784, ))
    
    x_1 = Dense(128, activation='relu')(inputs)
    x_2 = Dense(128, activation='relu')(x_1)
    x_3 = Dense(128, activation='relu')(x_2)
    x_4 = Dense(128, activation='relu')(x_3)
    x_5 = Dense(62, activation='relu')(x_4)
    x_6 = Dense(62, activation='relu')(x_5)
    x_6 = Dense(62, activation='relu')(x_6)
    x_6 = Dense(62, activation='relu')(x_6)
    x_6 = Dense(62, activation='relu')(x_6)
    x_6 = Dense(62, activation='relu')(x_6)
    x_6 = Dense(62, activation='relu')(x_6)
    x_6 = Dense(62, activation='relu')(x_6)
    x_6 = Dense(62, activation='relu')(x_6)
    x_6 = Dense(62, activation='relu')(x_6)
    
    predictions = Dense(10, activation='softmax')(x_6)
    
    model = Model(inputs=inputs, outputs=predictions)
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X, y, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)
    
if __name__ == '__main__':
    batch_size = 128
    num_classes = 10
    epochs = 20
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    nn_model(x_train, y_train, x_test, y_test, batch_size, epochs)
