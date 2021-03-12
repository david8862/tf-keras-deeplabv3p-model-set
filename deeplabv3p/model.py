#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create deeplabv3p models
"""
from functools import partial
from tensorflow.keras.layers import Conv2D, Reshape, Activation, Softmax, Lambda, Input
from tensorflow.keras.models import Model

from deeplabv3p.models.deeplabv3p_xception import Deeplabv3pXception
from deeplabv3p.models.deeplabv3p_mobilenetv2 import Deeplabv3pMobileNetV2, Deeplabv3pLiteMobileNetV2
from deeplabv3p.models.deeplabv3p_mobilenetv3 import Deeplabv3pMobileNetV3Large, Deeplabv3pLiteMobileNetV3Large, Deeplabv3pMobileNetV3Small, Deeplabv3pLiteMobileNetV3Small
from deeplabv3p.models.deeplabv3p_resnet50 import Deeplabv3pResNet50
from deeplabv3p.models.layers import DeeplabConv2D, Subpixel, img_resize

#
# A map of model type to construction function for DeepLabv3+
#
deeplab_model_map = {
    'mobilenetv2': partial(Deeplabv3pMobileNetV2, alpha=1.0),
    'mobilenetv2_lite': partial(Deeplabv3pLiteMobileNetV2, alpha=1.0),

    'mobilenetv3large': partial(Deeplabv3pMobileNetV3Large, alpha=1.0),
    'mobilenetv3large_lite': partial(Deeplabv3pLiteMobileNetV3Large, alpha=1.0),

    'mobilenetv3small': partial(Deeplabv3pMobileNetV3Small, alpha=1.0),
    'mobilenetv3small_lite': partial(Deeplabv3pLiteMobileNetV3Small, alpha=1.0),

    'xception': Deeplabv3pXception,
    'resnet50': Deeplabv3pResNet50,
}


def get_deeplabv3p_model(model_type, num_classes, model_input_shape, output_stride, freeze_level=0, weights_path=None, training=True, use_subpixel=False):
    # check if model type is valid
    if model_type not in deeplab_model_map.keys():
        raise ValueError('This model type is not supported now')

    model_function = deeplab_model_map[model_type]

    input_tensor = Input(shape=model_input_shape + (3,), name='image_input')
    model, backbone_len = model_function(input_tensor=input_tensor,
                                         input_shape=model_input_shape + (3,),
                                         weights=None,
                                         num_classes=21,
                                         OS=output_stride)

    base_model = Model(model.input, model.layers[-5].output)
    print('backbone layers number: {}'.format(backbone_len))

    if use_subpixel:
        if model_type == 'xception':
            scale = 4
        else:
            scale = 8
        x = Subpixel(num_classes, 1, scale, padding='same')(base_model.output)
    else:
        x = DeeplabConv2D(num_classes, (1, 1), padding='same', name='conv_upsample')(base_model.output)
        x = Lambda(img_resize, arguments={'size': (model_input_shape[0], model_input_shape[1])}, name='pred_resize')(x)

    # for training model, we need to flatten mask to calculate loss
    if training:
        x = Reshape((model_input_shape[0]*model_input_shape[1], num_classes)) (x)

    x = Softmax(name='pred_mask')(x)
    model = Model(base_model.input, x, name='deeplabv3p_'+model_type)

    #if use_subpixel:
        # Do ICNR
        #for layer in model.layers:
            #if type(layer) == Subpixel:
                #c, b = layer.get_weights()
                #w = icnr_weights(scale=scale, shape=c.shape)
                #layer.set_weights([w, b])

    if weights_path:
        model.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    if freeze_level in [1, 2]:
        # Freeze the backbone part or freeze all but final feature map & input layers.
        num = (backbone_len, len(base_model.layers))[freeze_level-1]
        for i in range(num): model.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model.layers)))
    elif freeze_level == 0:
        # Unfreeze all layers.
        for i in range(len(model.layers)):
            model.layers[i].trainable= True
        print('Unfreeze all of the layers.')

    return model

