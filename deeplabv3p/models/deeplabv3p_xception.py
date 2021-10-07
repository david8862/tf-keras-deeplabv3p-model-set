#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Deeplabv3+ Xception model for Keras.
On Pascal VOC, original model gets to 84.56% mIOU

Reference Paper:
- [Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Xception: Deep Learning with Depthwise Separable Convolutions]
    (https://arxiv.org/abs/1610.02357)
"""
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ZeroPadding2D, Lambda, AveragePooling2D, Input, Concatenate, Add, Reshape, BatchNormalization, Dropout, ReLU, Softmax, add
from tensorflow.keras.utils import get_source_inputs, get_file
#from tensorflow.keras import backend as K

from deeplabv3p.models.layers import DeeplabConv2D, DeeplabDepthwiseConv2D, CustomBatchNormalization, SepConv_BN, ASPP_block, Decoder_block, normalize, img_resize

WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return DeeplabConv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return DeeplabConv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = CustomBatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def Xception_body(input_tensor, OS):
    """
    Modified Alighed Xception feature extractor body
    with specified output stride and skip level feature
    """
    if OS == 8:
        origin_os16_stride = 1
        origin_os16_block_rate = 2
        origin_os32_stride = 1
        origin_os32_block_rate = 4
    elif OS == 16:
        origin_os16_stride = 2
        origin_os16_block_rate = 1
        origin_os32_stride = 1
        origin_os32_block_rate = 2
    elif OS == 32:
        origin_os16_stride = 2
        origin_os16_block_rate = 1
        origin_os32_stride = 2
        origin_os32_block_rate = 1
    else:
        raise ValueError('invalid output stride', OS)

    x = DeeplabConv2D(32, (3, 3), strides=(2, 2),
               name='entry_flow_conv1_1', use_bias=False, padding='same')(input_tensor)

    x = CustomBatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = ReLU()(x)

    x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    x = CustomBatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = ReLU()(x)

    x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                        skip_connection_type='conv', stride=2,
                        depth_activation=False)
    # skip level feature, with output stride = 4
    x, skip = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                               skip_connection_type='conv', stride=2,
                               depth_activation=False, return_skip=True)

    # original output stride changes to 16 from here, so we start to control block stride and dilation rate
    x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                        skip_connection_type='conv', stride=origin_os16_stride,
                        depth_activation=False)
    for i in range(16):
        x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type='sum', stride=1, rate=origin_os16_block_rate,
                            depth_activation=False)

    # original output stride changes to 32 from here
    x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                        skip_connection_type='conv', stride=origin_os32_stride, rate=origin_os16_block_rate,
                        depth_activation=False)
    x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                        skip_connection_type='none', stride=1, rate=origin_os32_block_rate,
                        depth_activation=True)
    # end of feature extractor

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    #else:
        #inputs = img_input

    backbone_len = len(Model(inputs, x).layers)
    return x, skip, backbone_len



def Deeplabv3pXception(input_shape=(512, 512, 3),
                       weights='pascalvoc',
                       input_tensor=None,
                       num_classes=21,
                       OS=16):
    """ Instantiates the Deeplabv3+ architecture
    Optionally loads weights pre-trained
    on PASCAL VOC. This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    data format `(width, height, channels)`.
    # Arguments
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images
        weights: pretrained weights type
                - pascalvoc : pre-trained on PASCAL VOC
                - None : random initialization
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        num_classes: number of desired classes.
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16,32}
    # Returns
        A Keras model instance.
    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`
    """

    if not (weights in {'pascalvoc', None}):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `pascalvoc` '
                         '(pre-trained on PASCAL VOC)')

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='image_input')
    else:
        img_input = input_tensor

    # normalize input image
    #img_norm = Lambda(normalize, name='input_normalize')(img_input)

    # backbone body for feature extract
    x, skip_feature, backbone_len = Xception_body(img_input, OS)

    # ASPP block
    x = ASPP_block(x, OS)

    # Deeplabv3+ decoder for feature projection
    x = Decoder_block(x, skip_feature)

    # Final prediction conv block
    x = DeeplabConv2D(num_classes, (1, 1), padding='same', name='logits_semantic')(x)
    x = Lambda(img_resize, arguments={'size': (input_shape[0],input_shape[1]), 'mode': 'bilinear'}, name='pred_resize')(x)
    x = Reshape((input_shape[0]*input_shape[1], num_classes)) (x)
    x = Softmax(name='Predictions/Softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    #if input_tensor is not None:
        #inputs = get_source_inputs(input_tensor)
    #else:
        #inputs = img_input

    model = Model(img_input, x, name='deeplabv3p_xception')

    # load weights
    if weights == 'pascalvoc':
        weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_PATH_X,
                                cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
    return model, backbone_len

