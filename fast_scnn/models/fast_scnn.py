#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Fast-SCNN model for Keras. Ported from
https://github.com/kshitizrimal/Fast-SCNN

# Reference Paper:
- [Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/abs/1902.04502)
"""
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D, UpSampling2D, ZeroPadding2D, Lambda, AveragePooling2D, Input, Concatenate, Add, Reshape, BatchNormalization, Dropout, ReLU, Softmax
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras import backend as K

from deeplabv3p.models.layers import normalize, img_resize, DeeplabConv2D, DeeplabDepthwiseConv2D, DeeplabSeparableConv2D, CustomBatchNormalization


def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True):
    """
    # Model Architecture
    #### Custom function for conv2d: conv_block
    """
    if(conv_type == 'ds'):
      x = DeeplabSeparableConv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)
    else:
      x = DeeplabConv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)

    x = CustomBatchNormalization()(x)

    if (relu):
      x = ReLU()(x)

    return x


def _res_bottleneck(inputs, filters, kernel, t, s, r=False):
    """
    #### residual custom method
    """
    tchannel = K.int_shape(inputs)[-1] * t

    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

    x = DeeplabDepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = CustomBatchNormalization()(x)
    x = ReLU()(x)

    x = conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)

    if r:
        x = Add()([x, inputs])
    return x


def bottleneck_block(inputs, filters, kernel, t, strides, n):
    """
    #### Bottleneck custom method
    """
    x = _res_bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
      x = _res_bottleneck(x, filters, kernel, t, 1, True)

    return x


def pyramid_pooling_block(input_tensor, bin_sizes):
    """
    #### PPM Method
    """
    concat_list = [input_tensor]
    w = input_tensor.shape[1]
    h = input_tensor.shape[2]

    for bin_size in bin_sizes:
      x = AveragePooling2D(pool_size=(w//bin_size, h//bin_size), strides=(w//bin_size, h//bin_size))(input_tensor)
      x = DeeplabConv2D(128, 3, 2, padding='same')(x)
      x = Lambda(img_resize, arguments={'size': (w, h), 'mode': 'bilinear'})(x)
      #x = UpSampling2D((w//bin_size, h//bin_size))(x)

      concat_list.append(x)

    return Concatenate(axis=-1)(concat_list)


def FastSCNN(num_classes,
             input_shape=(2048, 1024, 3),
             input_tensor=None,
             weights=None,
             training=True,
             **kwargs):

    if input_tensor is None:
        inputs = Input(shape=input_shape, name='image_input')
    else:
        inputs = input_tensor

    # normalize input image
    #inputs_norm= Lambda(normalize, name='input_normalize')(inputs)

    """## Step 1: Learning to DownSample"""
    lds_layer = conv_block(inputs, 'conv', 32, (3, 3), strides = (2, 2))
    lds_layer = conv_block(lds_layer, 'ds', 48, (3, 3), strides = (2, 2))
    lds_layer = conv_block(lds_layer, 'ds', 64, (3, 3), strides = (2, 2))


    """## Step 2: Global Feature Extractor"""
    """#### Assembling all the methods"""
    gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)
    gfe_layer = pyramid_pooling_block(gfe_layer, [2,4,6,8])

    """## Step 3: Feature Fusion"""
    ff_layer1 = conv_block(lds_layer, 'conv', 128, (1,1), padding='same', strides= (1,1), relu=False)
    ff_layer2 = UpSampling2D((4, 4))(gfe_layer)
    ff_layer2 = DeeplabSeparableConv2D(128, (3, 3), padding='same', strides = (1, 1), activation=None, dilation_rate=(4, 4))(ff_layer2)

    # old approach with DepthWiseConv2d
    #ff_layer2 = DeeplabDepthwiseConv2D((3,3), strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)

    ff_layer2 = CustomBatchNormalization()(ff_layer2)
    ff_layer2 = ReLU()(ff_layer2)
    ff_layer2 = DeeplabConv2D(128, 1, 1, padding='same', activation=None)(ff_layer2)

    ff_final = Add()([ff_layer1, ff_layer2])
    ff_final = CustomBatchNormalization()(ff_final)
    ff_final = ReLU()(ff_final)

    """## Step 4: Classifier"""
    classifier = DeeplabSeparableConv2D(128, (3, 3), padding='same', strides = (1, 1), name = 'DSConv1_classifier')(ff_final)
    classifier = CustomBatchNormalization()(classifier)
    classifier = ReLU()(classifier)

    classifier = DeeplabSeparableConv2D(128, (3, 3), padding='same', strides = (1, 1), name = 'DSConv2_classifier')(classifier)
    classifier = CustomBatchNormalization()(classifier)
    classifier = ReLU()(classifier)

    classifier = conv_block(classifier, 'conv', num_classes, (1, 1), strides=(1, 1), padding='same', relu=False)

    classifier = Dropout(0.3)(classifier)

    classifier = UpSampling2D((8, 8))(classifier)

    # for training model, we need to flatten mask to calculate loss
    if training:
        classifier = Reshape((inputs.shape[1]*inputs.shape[2], num_classes))(classifier)

    classifier = Softmax(name='pred_mask')(classifier)

    model = Model(inputs=inputs, outputs=classifier, name='Fast_SCNN')

    return model


if __name__ == '__main__':
    # try to use legecy optimizer if possible
    try:
        from tensorflow.keras.optimizers.legacy import SGD
    except:
        from tensorflow.keras.optimizers import SGD

    #input_tensor = Input(shape=(2048, 1024, 3), name='image_input')
    input_tensor = Input(shape=(512, 512, 3), name='image_input')
    model = FastSCNN(input_tensor=input_tensor,
                                      weights=None,
                                      num_classes=21)

    optimizer = SGD(momentum=0.9, lr=0.045)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()
