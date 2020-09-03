#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Deeplabv3+ ResNet50 model for Keras.

# Reference:
- [Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
"""
import os
import warnings
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ZeroPadding2D, Lambda, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Dense, Concatenate, add, Reshape, BatchNormalization, Dropout, ReLU, Softmax
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras import backend as K

from deeplabv3p.models.layers import DeeplabConv2D, DeeplabDepthwiseConv2D, ASPP_block, ASPP_Lite_block, Decoder_block, normalize, img_resize


WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')


def identity_block(input_tensor, kernel_size, filters, stage, block, rate=1):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      dilation_rate=(rate, rate),
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = ReLU()(x)

    x = Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      dilation_rate=(rate, rate),
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = ReLU()(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      dilation_rate=(rate, rate),
                      name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = ReLU()(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               rate=1):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      dilation_rate=(rate, rate),
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = ReLU()(x)

    x = Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      dilation_rate=(rate, rate),
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = ReLU()(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      dilation_rate=(rate, rate),
                      name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             dilation_rate=(rate, rate),
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = ReLU()(x)
    return x


def ResNet50(include_top=True,
             OS=8,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    """
    Modified ResNet50 feature extractor body
    with specified output stride and skip level feature
    """
    if OS == 8:
        origin_os16_stride = (1, 1)
        origin_os16_block_rate = 2
        origin_os32_stride = (1, 1)
        origin_os32_block_rate = 4
    elif OS == 16:
        origin_os16_stride = (2, 2)
        origin_os16_block_rate = 1
        origin_os32_stride = (1, 1)
        origin_os32_block_rate = 2
    elif OS == 32:
        origin_os16_stride = (2, 2)
        origin_os16_block_rate = 1
        origin_os32_stride = (2, 2)
        origin_os32_block_rate = 1
    else:
        raise ValueError('invalid output stride', OS)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        #if not backend.is_keras_tensor(input_tensor):
            #img_input = Input(tensor=input_tensor, shape=input_shape)
        #else:
            #img_input = input_tensor
        img_input = input_tensor

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = ReLU()(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # skip level feature, with output stride = 4
    skip = x

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # original output stride changes to 16 from here, so we start to control block stride and dilation rate
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=origin_os16_stride) # origin: stride=(2, 2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', rate=origin_os16_block_rate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', rate=origin_os16_block_rate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', rate=origin_os16_block_rate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', rate=origin_os16_block_rate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', rate=origin_os16_block_rate)

    # original output stride changes to 32 from here
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=origin_os32_stride, rate=origin_os16_block_rate) # origin: stride=(2, 2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', rate=origin_os32_block_rate)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', rate=origin_os32_block_rate)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    backbone_len = len(model.layers)
    # need to return feature map and skip connection,
    # not the whole "no top" model
    return x, skip, backbone_len
    #return model



def Deeplabv3pResNet50(input_shape=(512, 512, 3),
                          weights=None,
                          input_tensor=None,
                          classes=21,
                          OS=8,
                          **kwargs):
    """ Instantiates the Deeplabv3+ MobileNetV3Large architecture
    # Arguments
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images
        weights: one of 'pascal_voc' (pre-trained on pascal voc)
            or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        classes: number of desired classes. If classes != 21,
            last layer is initialized randomly
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.

    # Returns
        A Keras model instance.
    """
    if not (weights in {'pascal_voc', None}):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `pascal_voc` '
                         '(pre-trained on PASCAL VOC)')

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='image_input')
    else:
        img_input = input_tensor

    # normalize input image
    img_norm = Lambda(normalize, name='input_normalize')(img_input)

    # backbone body for feature extract
    x, skip_feature, backbone_len = ResNet50(include_top=False, input_tensor=img_norm, weights='imagenet', OS=OS)

    # ASPP block
    x = ASPP_block(x, OS)

    # Deeplabv3+ decoder for feature projection
    x = Decoder_block(x, skip_feature)

    # Final prediction conv block
    x = DeeplabConv2D(classes, (1, 1), padding='same', name='logits_semantic')(x)
    x = Lambda(img_resize, arguments={'size': (input_shape[0],input_shape[1]), 'mode': 'bilinear'}, name='pred_resize')(x)
    x = Reshape((input_shape[0]*input_shape[1], classes)) (x)
    x = Softmax(name='Predictions/Softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    #if input_tensor is not None:
        #inputs = get_source_inputs(input_tensor)
    #else:
        #inputs = img_input
    model = Model(img_input, x, name='deeplabv3p_resnet50')

    return model, backbone_len




if __name__ == '__main__':
    input_tensor = Input(shape=(224, 224, 3), name='image_input')
    #model = ResNet50(include_top=False, input_shape=(512, 512, 3), weights='imagenet')
    model = ResNet50(include_top=True, input_tensor=input_tensor, weights='imagenet')
    model.summary()

    import numpy as np
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    from keras_preprocessing import image

    img = image.load_img('../../examples/dog.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
