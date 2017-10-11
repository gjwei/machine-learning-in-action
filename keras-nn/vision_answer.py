#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/9/28
  
"""
import keras
import numpy as np
from keras.layers import (
                        Conv2D, MaxPool2D, Flatten, Input, LSTM,
                        Embedding, Dense
                        )
from keras.models import Model, Sequential


# define a sequential model
vision_mode = Sequential()
vision_mode.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(3, 224, 224)))
vision_mode.add(Conv2D(64, (3, 3), activation='relu'))
vision_mode.add(MaxPool2D((2, 2)))

vision_mode.add(Conv2D(128, (3, 3), activation='relu'))
vision_mode.add(Conv2D(128, (3, 3), activation='relu'))
vision_mode.add(MaxPool2D((2, 2)))

vision_mode.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_mode.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_mode.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_mode.add(MaxPool2D((2,2 )))
vision_mode.add(Flatten())

image_input = Input((3, 224, 224))
encoded_image = vision_mode(image_input)

# define a language model to encode the question into a vector
question_input = Input(shape=(100,), dtype=np.int32)
embeded_question = Embedding(input_dim=1000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embeded_question)

merged = keras.layers.concatenate([encoded_image, encoded_question])

output = Dense(1000, activation='softmax')(merged)

model = Model(inputs=[image_input, question_input], outputs=output)





