#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create fast_scnn models
"""
#from functools import partial
from tensorflow.keras.layers import Reshape, Softmax, Input
from tensorflow.keras.models import Model

from fast_scnn.models.fast_scnn import FastSCNN

#
# A map of model type to construction function for FastSCNN
#
fast_scnn_model_map = {
    'fast_scnn': FastSCNN,
}

def get_fast_scnn_model(model_type, num_classes, model_input_shape, weights_path=None, training=True):
    # check if model type is valid
    if model_type not in fast_scnn_model_map.keys():
        raise ValueError('This model type is not supported now')

    model_function = fast_scnn_model_map[model_type]

    input_tensor = Input(shape=model_input_shape + (3,), name='image_input')
    model = model_function(num_classes, input_tensor=input_tensor,
                           input_shape=model_input_shape + (3,),
                           weights=None,
                           training=training)

    # for training model, we need to flatten mask to calculate loss
    #if training:
        #x = Reshape((model_input_shape[0]*model_input_shape[1], num_classes)) (base_model.output)
    #else:
        #x = base_model.output

    #x = Softmax(name='pred_mask')(x)
    #model = Model(base_model.input, x, name=model_type)

    if weights_path:
        model.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    return model

