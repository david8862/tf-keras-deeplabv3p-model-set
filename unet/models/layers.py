#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division

from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.regularizers import l2
import tensorflow as tf


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

