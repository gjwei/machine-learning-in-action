#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/9/28
  
"""

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, MaxPool2D


class VGG16(object):
    def __init__(self, img_width=128, img_height=128, ):
        self.img_width = img_width
        self.img_height = img_height
        
    def build(self):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), batch_input_shape=(1, self.img_width, self.img_height, 3)))
        first_layer = model.layers[-1]
        
        input_img = first_layer.input
        
        blocks = 5
        block_layers = [2, 2, 3, 3, 3]
        block_kernels = [64, 128, 256, 512, 521]
        
        for block in range(blocks):
            for layer in range(block_layers[block]):
                if block == 0 and layer == 0:
                    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
                else:
                    model.add(Conv2D(block_kernels[block], (3, 3), padding='same', activation='relu',
                                 name='conv{0}_{1}'.format(block + 1, layer + 1)))
            model.add(MaxPool2D((2, 2), strides=(2, 2), name='maxpool{0}'.format(block)))
            
        return model
    
    
if __name__ == '__main__':
    model = VGG16().build()
    # model.summary()
    
    layer_dict = dict([layer.name, layer] for layer in model.layers)
    
