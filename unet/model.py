#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create unet models
"""
from functools import partial
from tensorflow.keras.layers import Reshape, Softmax, Input
from tensorflow.keras.models import Model

from unet.models.unet import UNetStandard, UNetLite, UNetSimple

#
# A map of model type to construction function for UNet
#
unet_model_map = {
    'unet_standard': UNetStandard,
    'unet_lite': UNetLite,
    'unet_simple': UNetSimple,
}

def get_unet_model(model_type, num_classes, model_input_shape, freeze_level=0, weights_path=None, training=True):
    # check if model type is valid
    if model_type not in unet_model_map.keys():
        raise ValueError('This model type is not supported now')

    model_function = unet_model_map[model_type]

    input_tensor = Input(shape=model_input_shape + (3,), name='image_input')
    base_model = model_function(num_classes, input_tensor=input_tensor,
                           input_shape=model_input_shape + (3,),
                           weights=None)

    #base_model = Model(model.input, model.layers[-5].output)
    #print('backbone layers number: {}'.format(backbone_len))


    # for training model, we need to flatten mask to calculate loss
    if training:
        x = Reshape((model_input_shape[0]*model_input_shape[1], num_classes)) (base_model.output)
    else:
        x = base_model.output

    x = Softmax(name='pred_mask')(x)
    model = Model(base_model.input, x, name=model_type)

    if weights_path:
        model.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    #if freeze_level in [1, 2]:
        ## Freeze the backbone part or freeze all but final feature map & input layers.
        #num = (backbone_len, len(base_model.layers))[freeze_level-1]
        #for i in range(num): model.layers[i].trainable = False
        #print('Freeze the first {} layers of total {} layers.'.format(num, len(model.layers)))
    #elif freeze_level == 0:
        ## Unfreeze all layers.
        #for i in range(len(model.layers)):
            #model.layers[i].trainable= True
        #print('Unfreeze all of the layers.')

    return model

