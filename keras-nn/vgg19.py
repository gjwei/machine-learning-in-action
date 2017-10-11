#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/9/28
  
"""
from __future__ import print_function
from __future__ import absolute_import

import keras

from keras.models import Model
from keras.layers import (Conv2D, MaxPool2D, Dense, Input,
                        GlobalAveragePooling2D, GlobalMaxPool2D,
                        Flatten)
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape


def VGG19(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    """
    
    :param include_top:
    :param weight:
    :param input_tensor:
    :param input_shape:
    :param pooling:
    :param classes:
    :return:
    """
    if weights not in {'imgenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weight)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
            
    x = Conv2D(64, (3, 3), padding='same', activition='relu', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv2')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv2')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv4')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv4')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv4')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg19')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
    
        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')
        
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model

    
    
    
    
    

