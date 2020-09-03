#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Deeplabv3+ MobileNetV2 model for Keras.

# Reference Paper:
- [Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for
    Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)
"""
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ZeroPadding2D, Lambda, AveragePooling2D, Input, Concatenate, Add, Reshape, BatchNormalization, Dropout, ReLU, Softmax
from tensorflow.keras.utils import get_source_inputs, get_file
#from tensorflow.keras import backend as K

from deeplabv3plus.models.layers import DeeplabConv2D, DeeplabDepthwiseConv2D, ASPP_block, ASPP_Lite_block, Decoder_block, normalize, img_resize

BACKBONE_WEIGHT_PATH = ('https://github.com/JonathanCMitchell/mobilenet_v2_keras/'
                    'releases/download/v1.1/')

WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    #in_channels = inputs._keras_shape[-1]
    in_channels = inputs.shape.as_list()[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand
        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = ReLU(max_value=6.)(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)
    x = ReLU(max_value=6., name=prefix + 'depthwise_relu')(x)

    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])
    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


def MobileNetV2_body(input_tensor, OS, alpha, weights='imagenet'):
    """
    Modified MobileNetV2 feature extractor body
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

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2), padding='same',
               use_bias=False, name='Conv')(input_tensor)
    x = BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = ReLU(6.)(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)
    # skip level feature, with output stride = 4
    skip = x

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)

    # original output stride changes to 16 from here, so we start to control block stride and dilation rate
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=origin_os16_stride,  # origin: stride=2!
                            expansion=6, block_id=6, skip_connection=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=origin_os16_block_rate,
                            expansion=6, block_id=7, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=origin_os16_block_rate,
                            expansion=6, block_id=8, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=origin_os16_block_rate,
                            expansion=6, block_id=9, skip_connection=True)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=origin_os16_block_rate,
                            expansion=6, block_id=10, skip_connection=False)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=origin_os16_block_rate,
                            expansion=6, block_id=11, skip_connection=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=origin_os16_block_rate,
                            expansion=6, block_id=12, skip_connection=True)

    # original output stride changes to 32 from here
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=origin_os32_stride, rate=origin_os16_block_rate,  # origin: stride=2!
                            expansion=6, block_id=13, skip_connection=False)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=origin_os32_block_rate,
                            expansion=6, block_id=14, skip_connection=True)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=origin_os32_block_rate,
                            expansion=6, block_id=15, skip_connection=True)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=origin_os32_block_rate,
                            expansion=6, block_id=16, skip_connection=False)
    # end of feature extractor

    # expand the model structure to MobileNetV2 no top, so
    # that we can load official imagenet pretrained weights

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    y = Conv2D(last_block_filters,
                      kernel_size=1,
                      use_bias=False,
                      name='Conv_1')(x)
    y = BatchNormalization(epsilon=1e-3,
                           momentum=0.999,
                           name='Conv_1_bn')(y)
    y = ReLU(6., name='out_relu')(y)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    #else:
        #inputs = img_input

    # hardcode row=224
    rows = 224

    model = Model(inputs, y, name='mobilenetv2_%0.2f_%s' % (alpha, rows))
    # Load weights.
    if weights == 'imagenet':
        model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +
                      str(alpha) + '_' + str(rows) + '_no_top' + '.h5')
        weight_path = BACKBONE_WEIGHT_PATH + model_name
        weights_path = get_file(
            model_name, weight_path, cache_subdir='models')

        model.load_weights(weights_path)

    backbone_len = len(model.layers) - 3
    # need to return feature map and skip connection,
    # not the whole "no top" model
    return x, skip, backbone_len


def Deeplabv3pMobileNetV2(input_shape=(512, 512, 3),
                          alpha=1.0,
                          weights=None,
                          input_tensor=None,
                          classes=21,
                          OS=8,
                          **kwargs):
    """ Instantiates the Deeplabv3+ MobileNetV2 architecture
    # Arguments
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone
        weights: one of 'pascal_voc' (pre-trained on pascal voc)
            or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        classes: number of desired classes. If classes != 21,
            last layer is initialized randomly
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.

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
    x, skip_feature, backbone_len = MobileNetV2_body(img_norm, OS, alpha)

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

    model = Model(img_input, x, name='deeplabv3p_mobilenetv2')

    # load weights
    #if weights == 'pascal_voc':
        #weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                #WEIGHTS_PATH_MOBILE,
                                #cache_subdir='models')
        #model.load_weights(weights_path, by_name=True)
    return model, backbone_len


def Deeplabv3pLiteMobileNetV2(input_shape=(512, 512, 3),
                          alpha=1.0,
                          weights='pascal_voc',
                          input_tensor=None,
                          classes=21,
                          OS=8,
                          **kwargs):
    """ Instantiates the Deeplabv3+ MobileNetV2Lite architecture
    # Arguments
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone
        weights: one of 'pascal_voc' (pre-trained on pascal voc)
            or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        classes: number of desired classes. If classes != 21,
            last layer is initialized randomly
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.

    # Returns
        A Keras model instance.
    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`
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
    x, _, backbone_len = MobileNetV2_body(img_norm, OS, alpha)

    # use ASPP Lite block & no decode block
    x = ASPP_Lite_block(x)

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

    model = Model(img_input, x, name='deeplabv3p_mobilenetv2')

    # load weights
    if weights == 'pascal_voc':
        weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_PATH_MOBILE,
                                cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
    return model, backbone_len

