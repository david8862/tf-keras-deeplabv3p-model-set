#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division

from functools import wraps

#import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ZeroPadding2D, Lambda, AveragePooling2D, Concatenate, BatchNormalization, Dropout, ReLU
from tensorflow.keras.regularizers import l2
import tensorflow as tf


@wraps(Conv2D)
def DeeplabConv2D(*args, **kwargs):
    """Wrapper to set Deeplab parameters for Conv2D."""
    deeplab_conv_kwargs = {'kernel_regularizer': l2(2e-5)}
    #deeplab_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    deeplab_conv_kwargs.update(kwargs)
    return Conv2D(*args, **deeplab_conv_kwargs)


@wraps(DepthwiseConv2D)
def DeeplabDepthwiseConv2D(*args, **kwargs):
    """Wrapper to set Deeplab parameters for DepthwiseConv2D."""
    deeplab_conv_kwargs = {'kernel_regularizer': l2(2e-5)}
    #deeplab_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    deeplab_conv_kwargs.update(kwargs)
    return DepthwiseConv2D(*args, **deeplab_conv_kwargs)


def normalize(x):
    return x/127.5 - 1


def img_resize(x, size, mode='bilinear'):
    if mode == 'bilinear':
        return tf.image.resize(x, size=size, method='bilinear')
    elif mode == 'nearest':
        return tf.image.resize(x, size=size, method='nearest')
    else:
        raise ValueError('output model file is not specified')


def CustomBatchNormalization(*args, **kwargs):
    if tf.__version__ >= '2.2':
        from tensorflow.keras.layers.experimental import SyncBatchNormalization
        BatchNorm = SyncBatchNormalization
    else:
        BatchNorm = BatchNormalization

    return BatchNorm(*args, **kwargs)



def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = ReLU()(x)
    x = DeeplabDepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = CustomBatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = ReLU()(x)
    x = DeeplabConv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = CustomBatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = ReLU()(x)

    return x


def ASPP_block(x, OS):
    """
    branching for Atrous Spatial Pyramid Pooling
    """
    if OS == 8:
        atrous_rates = (12, 24, 36)
    elif OS == 16:
        atrous_rates = (6, 12, 18)
    elif OS == 32:
        # unofficial hyperparameters, just have a try
        atrous_rates = (3, 6, 9)
    else:
        raise ValueError('invalid output stride', OS)

    # feature map shape, (batch, height, width, channel)
    feature_shape = x.shape.as_list()

    # Image Feature branch
    b4 = AveragePooling2D(pool_size=(feature_shape[1], feature_shape[2]))(x)

    b4 = DeeplabConv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = CustomBatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = ReLU()(b4)
    b4 = Lambda(img_resize, arguments={'size': (feature_shape[1], feature_shape[2]), 'mode': 'bilinear'}, name='aspp_resize')(b4)

    # simple 1x1
    b0 = DeeplabConv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = CustomBatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = ReLU(name='aspp0_activation')(b0)

    # rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = DeeplabConv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = CustomBatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    return x


def ASPP_Lite_block(x):
    """
    a simplified version of Deeplab ASPP block, which
    only have global pooling & simple 1x1 conv branch
    """
    # feature map shape, (batch, height, width, channel)
    feature_shape = x.shape.as_list()

    # Image Feature branch
    b4 = AveragePooling2D(pool_size=(feature_shape[1], feature_shape[2]))(x)

    b4 = DeeplabConv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = CustomBatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = ReLU()(b4)
    b4 = Lambda(img_resize, arguments={'size': (feature_shape[1], feature_shape[2]), 'mode': 'bilinear'}, name='aspp_resize')(b4)

    # simple 1x1 conv
    b0 = DeeplabConv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = CustomBatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = ReLU(name='aspp0_activation')(b0)

    # only 2 branches
    x = Concatenate()([b4, b0])
    x = DeeplabConv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = CustomBatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    return x


def Decoder_block(x, skip_feature):
    """
    DeepLab v.3+ decoder
    Feature projection x4 (x2) block
    """
    # skip feature shape, (batch, height, width, channel)
    skip_shape = skip_feature.shape.as_list()

    x = Lambda(img_resize, arguments={'size': (skip_shape[1], skip_shape[2]), 'mode': 'bilinear'}, name='decoder_resize')(x)

    skip_feature = DeeplabConv2D(48, (1, 1), padding='same',
                       use_bias=False, name='feature_projection0')(skip_feature)
    skip_feature = CustomBatchNormalization(
        name='feature_projection0_BN', epsilon=1e-5)(skip_feature)
    skip_feature = ReLU()(skip_feature)
    x = Concatenate()([x, skip_feature])
    x = SepConv_BN(x, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)
    return x



#def icnr_weights(init = tf.glorot_normal_initializer(), scale=2, shape=[3,3,32,4], dtype = tf.float32):
    #sess = tf.Session()
    #return sess.run(ICNR(init, scale=scale)(shape=shape, dtype=dtype))

class ICNR:
    """ICNR initializer for checkerboard artifact free sub pixel convolution
    Ref:
     [1] Andrew Aitken et al. Checkerboard artifact free sub-pixel convolution
     https://arxiv.org/pdf/1707.02937.pdf)
    Args:
    initializer: initializer used for sub kernels (orthogonal, glorot uniform, etc.)
    scale: scale factor of sub pixel convolution
    """

    def __init__(self, initializer, scale=1):
        self.scale = scale
        self.initializer = initializer

    def __call__(self, shape, dtype, partition_info=None):
        shape = list(shape)
        if self.scale == 1:
            return self.initializer(shape)

        new_shape = shape[:3] + [shape[3] // (self.scale ** 2)]
        x = self.initializer(new_shape, dtype, partition_info)
        x = tf.transpose(x, perm=[2, 0, 1, 3])
        x = tf.image.resize_nearest_neighbor(x, size=(shape[0] * self.scale, shape[1] * self.scale))
        x = tf.space_to_depth(x, block_size=self.scale)
        x = tf.transpose(x, perm=[1, 2, 0, 3])

        return x

class Subpixel(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding='valid',
                 data_format=None,
                 strides=(1,1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            filters=r*r*filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c = I.get_shape().as_list()
        bsize = K.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = K.reshape(I, [bsize, a, b, int(c/(r*r)),r, r]) # bsize, a, b, c/(r*r), r, r
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
        #Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:,i,:,:,:,:] for i in range(a)] # a, [bsize, b, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:,i,:,:,:] for i in range(b)] # b, [bsize, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r*unshifted[1], self.r*unshifted[2], int(unshifted[3]/(self.r*self.r)))

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.pop('rank')
        config.pop('dilation_rate')
        config['filters']= int(config['filters'] / self.r*self.r)
        config['r'] = self.r
        return config
