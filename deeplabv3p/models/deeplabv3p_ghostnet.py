#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Deeplabv3+ GhostNet model for Keras.

# Reference Paper:
- [Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907)
"""
import os, sys
import warnings
import math

from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Dense, Flatten, ReLU, Reshape, Activation
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Dropout, Add, Multiply, Lambda, Softmax
from tensorflow.keras import backend as K

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from deeplabv3p.models.layers import DeeplabConv2D, DeeplabDepthwiseConv2D, CustomBatchNormalization, ASPP_block, ASPP_Lite_block, Decoder_block, normalize, img_resize


BASE_WEIGHT_PATH = (
    'https://github.com/david8862/tf-keras-image-classifier/'
    'releases/download/v1.0.0/')


def preprocess_input(x):
    """
    "mode" option description in preprocess_input
    mode: One of "caffe", "tf" or "torch".
        - caffe: will convert the images from RGB to BGR,
            then will zero-center each color channel with
            respect to the ImageNet dataset,
            without scaling.
        - tf: will scale pixels between -1 and 1,
            sample-wise.
        - torch: will scale pixels between 0 and 1 and then
            will normalize each channel with respect to the
            ImageNet dataset.
    """
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2]

    return x


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x):
    return ReLU(6.0)(x + 3.0) / 6.0


def primary_conv(x, output_filters, kernel_size, strides=(1,1), padding='same', act=True, use_bias=False, name=None):
    x = DeeplabConv2D(filters=output_filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name + '_0')(x)
    x = CustomBatchNormalization(name=name+'_1')(x)
    x = ReLU(name=name+'_relu')(x) if act else x
    return x


def cheap_operations(x, output_filters, kernel_size, strides=(1,1), padding='same', act=True, use_bias=False, name=None):
    x = DeeplabDepthwiseConv2D(kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        use_bias=use_bias,
                        name=name+'_0')(x)
    x = CustomBatchNormalization(name=name+'_1')(x)
    x = ReLU(name=name+'_relu')(x) if act else x
    return x


def SqueezeExcite(input_x, se_ratio=0.25, reduced_base_chs=None, divisor=4, name=None):
    reduce_chs =_make_divisible((reduced_base_chs or int(input_x.shape[-1]))*se_ratio, divisor)

    x = GlobalAveragePooling2D(name=name+'_avg_pool2d')(input_x)
    if K.image_data_format() == 'channels_first':
        x = Reshape((int(input_x.shape[-1]), 1, 1))(x)
    else:
        x = Reshape((1, 1, int(input_x.shape[-1])))(x)

    x = DeeplabConv2D(filters=reduce_chs, kernel_size=1, use_bias=True, name=name+'_conv_reduce')(x)
    x = ReLU(name=name+'_act')(x)
    x = DeeplabConv2D(filters=int(input_x.shape[-1]), kernel_size=1, use_bias=True, name=name+'_conv_expand')(x)

    x = Activation(hard_sigmoid, name=name+'_hard_sigmoid')(x)
    x = Multiply()([input_x, x])

    return x


def ConvBnAct(input_x, out_chs, kernel_size, stride=(1,1), name=None):
    x = DeeplabConv2D(filters=out_chs,
               kernel_size=kernel_size,
               strides=stride,
               padding='valid',
               use_bias=False,
               name=name+'_conv')(input_x)
    x = CustomBatchNormalization(name=name+'_bn1')(x)
    x = ReLU(name=name+'_relu')(x)
    return x


def GhostModule(input_x, output_chs, kernel_size=1, ratio=2, dw_size=3, stride=(1,1), act=True, name=None):
    init_channels = int(math.ceil(output_chs / ratio))
    new_channels = int(init_channels * (ratio - 1))
    x1 = primary_conv(input_x,
                      init_channels,
                      kernel_size=kernel_size,
                      strides=stride,
                      padding='valid',
                      act=act,
                      name = name + '_primary_conv')
    x2 = cheap_operations(x1,
                          new_channels,
                          kernel_size=dw_size,
                          strides=(1,1),
                          padding= 'same',
                          act=act,
                          name = name + '_cheap_operation')
    x = Concatenate(axis=3,name=name+'_concat')([x1,x2])
    return x


def GhostBottleneck(input_x, mid_chs, out_chs, dw_kernel_size=3, stride=(1,1), rate=1, keep=False, se_ratio=0., name=None):
    '''ghostnet bottleneck w/optional se'''
    has_se = se_ratio is not None and se_ratio > 0.

    #1st ghost bottleneck
    x = GhostModule(input_x, mid_chs, act=True, name=name+'_ghost1')

    #depth_with convolution
    if stride[0] > 1 or keep:
        x = DeeplabDepthwiseConv2D(kernel_size=dw_kernel_size,
                            strides=stride,
                            padding='same',
                            dilation_rate=(rate, rate),
                            use_bias=False,
                            name=name+'_conv_dw')(x)
        x = CustomBatchNormalization(name=name+'_bn_dw')(x)

    #Squeeze_and_excitation
    if has_se:
        x = SqueezeExcite(x, se_ratio=se_ratio, name=name+'_se')

    #2nd ghost bottleneck
    x = GhostModule(x, out_chs, act=False, name=name+'_ghost2')

    #short cut
    if (input_x.shape[-1] == out_chs and stride[0] == 1):
        sc = input_x
    else:
        name1 = name + '_shortcut'
        sc = DeeplabDepthwiseConv2D(kernel_size=dw_kernel_size,
                             strides=stride,
                             padding='same',
                             dilation_rate=(rate, rate),
                             use_bias=False,
                             name=name1+'_0')(input_x)
        sc = CustomBatchNormalization(name=name1+'_1')(sc)
        sc = DeeplabConv2D(filters=out_chs,
                    kernel_size=1,
                    strides=(1,1),
                    padding='valid',
                    use_bias=False,
                    name=name1+'_2')(sc)
        sc = CustomBatchNormalization(name=name1+'_3')(sc)

    x = Add(name=name+'_add')([x, sc])
    return x


OS32_CFGS = [
        # k, t, c, SE, s, r
        # stage1
        [[3,  16,  16, 0, 1, 1]],
        # stage2
        [[3,  48,  24, 0, 2, 1]],
        [[3,  72,  24, 0, 1, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2, 1]],
        [[5, 120,  40, 0.25, 1, 1]],
        # stage4
        [[3, 240,  80, 0, 2, 1]],
        [[3, 200,  80, 0, 1, 1],
         [3, 184,  80, 0, 1, 1],
         [3, 184,  80, 0, 1, 1],
         [3, 480, 112, 0.25, 1, 1],
         [3, 672, 112, 0.25, 1, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2, 1]],
        [[5, 960, 160, 0, 1, 1],
         [5, 960, 160, 0.25, 1, 1],
         [5, 960, 160, 0, 1, 1],
         [5, 960, 160, 0.25, 1, 1]
        ]
    ]


OS16_CFGS = [
        # k, t, c, SE, s, r
        # stage1
        [[3,  16,  16, 0, 1, 1]],
        # stage2
        [[3,  48,  24, 0, 2, 1]],
        [[3,  72,  24, 0, 1, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2, 1]],
        [[5, 120,  40, 0.25, 1, 1]],
        # stage4
        [[3, 240,  80, 0, 2, 1]],
        [[3, 200,  80, 0, 1, 1],
         [3, 184,  80, 0, 1, 1],
         [3, 184,  80, 0, 1, 1],
         [3, 480, 112, 0.25, 1, 1],
         [3, 672, 112, 0.25, 1, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, -1, 1]], #origin: s=2, here s=-1 mean stride=1 but keep the downsample structure
        [[5, 960, 160, 0, 1, 2],
         [5, 960, 160, 0.25, 1, 2],
         [5, 960, 160, 0, 1, 2],
         [5, 960, 160, 0.25, 1, 2]
        ]
    ]


OS8_CFGS = [
        # k, t, c, SE, s, r
        # stage1
        [[3,  16,  16, 0, 1, 1]],
        # stage2
        [[3,  48,  24, 0, 2, 1]],
        [[3,  72,  24, 0, 1, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2, 1]],
        [[5, 120,  40, 0.25, 1, 1]],
        # stage4
        [[3, 240,  80, 0, -1, 1]], #origin: s=2, here s=-1 mean stride=1 but keep the downsample structure
        [[3, 200,  80, 0, 1, 2],
         [3, 184,  80, 0, 1, 2],
         [3, 184,  80, 0, 1, 2],
         [3, 480, 112, 0.25, 1, 2],
         [3, 672, 112, 0.25, 1, 2]
        ],
        # stage5
        [[5, 672, 160, 0.25, -1, 2]], #origin: s=2, here s=-1 mean stride=1 but keep the downsample structure
        [[5, 960, 160, 0, 1, 4],
         [5, 960, 160, 0.25, 1, 4],
         [5, 960, 160, 0, 1, 4],
         [5, 960, 160, 0.25, 1, 4]
        ]
    ]

def GhostNet(input_shape=None,
             OS=8,
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             width=1.0,
             dropout=0.2,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the GhostNet architecture.

    # Arguments
        input_shape: optional shape tuple, to be specified if you would
            like to use a model with an input img resolution that is not
            (224, 224, 3).
            It should have exactly 3 inputs channels (224, 224, 3).
            You can also omit this option if you would like
            to infer input_shape from an input_tensor.
            If you choose to include both input_tensor and input_shape then
            input_shape will be used if they match, if the shapes
            do not match then we will throw an error.
            E.g. `(160, 160, 3)` would be one valid value.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape, rows when
            weights='imagenet'
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')

    if OS == 8:
        cfgs=OS8_CFGS
    elif OS == 16:
        cfgs=OS16_CFGS
    elif OS == 32:
        cfgs=OS32_CFGS
    else:
        raise ValueError('invalid output stride', OS)

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)


    # If input_shape is None and input_tensor is None using standard shape
    if input_shape is None and input_tensor is None:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        #if not K.is_keras_tensor(input_tensor):
            #img_input = Input(tensor=input_tensor, shape=input_shape)
        #else:
            #img_input = input_tensor
        img_input = input_tensor

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # building first layer
    output_channel = int(_make_divisible(16 * width, 4))
    x = DeeplabConv2D(filters=output_channel,
               kernel_size=3,
               strides=(2, 2),
               padding='same',
               use_bias=False,
               name='conv_stem')(img_input)
    x = CustomBatchNormalization(name='bn1')(x)
    x = ReLU(name='Conv2D_1_act')(x)

    # building inverted residual blocks
    for index, cfg in enumerate(cfgs):
        sub_index = 0
        keep = False
        for k,exp_size,c,se_ratio,s,r in cfg:
            # s=-1 mean stride=1 but keep the downsample structure
            if s == -1:
                s = 1
                keep = True
            else:
                keep = False
            output_channel = int(_make_divisible(c * width, 4))
            hidden_channel = int(_make_divisible(exp_size * width, 4))
            x = GhostBottleneck(x, hidden_channel, output_channel, k, (s,s),
                                rate=r,
                                keep=keep,
                                se_ratio=se_ratio,
                                name='blocks_'+str(index)+'_'+str(sub_index))
            sub_index += 1
            # skip level feature, with output stride = 4
            if index == 2 and sub_index == 1:
                skip = x

    output_channel = _make_divisible(exp_size * width, 4)
    x = ConvBnAct(x, output_channel, kernel_size=1, name='blocks_9_0')
    # keep end of the feature extrator as final feature map
    final_feature = x

    if include_top:
        x = GlobalAveragePooling2D(name='global_avg_pooling2D')(x)
        if K.image_data_format() == 'channels_first':
            x = Reshape((output_channel, 1, 1))(x)
        else:
            x = Reshape((1, 1, output_channel))(x)

        # building last several layers
        output_channel = 1280
        x = DeeplabConv2D(filters=output_channel,
                   kernel_size=1,
                   strides=(1,1),
                   padding='valid',
                   use_bias=True,
                   name='conv_head')(x)
        x = ReLU(name='relu_head')(x)

        if dropout > 0.:
            x = Dropout(dropout, name='dropout_1')(x)
        x = Flatten()(x)
        x = Dense(units=classes, activation='softmax',
                         use_bias=True, name='classifier')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='ghostnet_%0.2f' % (width))

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            file_name = 'ghostnet_weights_tf_dim_ordering_tf_kernels_224.h5'
            weight_path = BASE_WEIGHT_PATH + file_name
        else:
            file_name = 'ghostnet_weights_tf_dim_ordering_tf_kernels_224_no_top.h5'
            weight_path = BASE_WEIGHT_PATH + file_name

        weights_path = get_file(file_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    # get backbone length
    if include_top:
        if dropout > 0.:
            backbone_len = len(model.layers) - 7
        else:
            backbone_len = len(model.layers) - 6
    elif pooling is not None:
        backbone_len = len(model.layers) - 1
    else:
        backbone_len = len(model.layers)

    return final_feature, skip, backbone_len



def Deeplabv3pGhostNet(input_shape=(512, 512, 3),
                       weights='imagenet',
                       input_tensor=None,
                       num_classes=21,
                       OS=8):
    """ Instantiates the Deeplabv3+ GhostNet architecture
    # Arguments
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images
        weights: pretrained weights type
                - imagenet: pre-trained on Imagenet
                - None : random initialization
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        num_classes: number of desired classes.
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16,32}.

    # Returns
        A Keras model instance.
    """

    if not (weights in {'imagenet', None}):
        raise ValueError('The `weights` argument should be either '
                         '`imagenet` (pre-trained on Imagenet) or '
                         '`None` (random initialization)')
    if input_tensor is None:
        img_input = Input(shape=input_shape, name='image_input')
    else:
        img_input = input_tensor

    # normalize input image
    img_norm = Lambda(normalize, name='input_normalize')(img_input)

    # backbone body for feature extract
    x, skip_feature, backbone_len = GhostNet(include_top=False, input_tensor=img_norm, weights=weights, OS=OS)

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

    model = Model(img_input, x, name='deeplabv3p_ghostnet')

    return model, backbone_len



def Deeplabv3pLiteGhostNet(input_shape=(512, 512, 3),
                          #alpha=1.0,
                          weights='imagenet',
                          input_tensor=None,
                          num_classes=21,
                          OS=8):
    """ Instantiates the Deeplabv3+ MobileNetV3LargeLite architecture
    # Arguments
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images
        weights: pretrained weights type
                - imagenet: pre-trained on Imagenet
                - None : random initialization
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        num_classes: number of desired classes.
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16,32}.

    # Returns
        A Keras model instance.
    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`
    """
    if not (weights in {'imagenet', None}):
        raise ValueError('The `weights` argument should be either '
                         '`imagenet` (pre-trained on Imagenet) or '
                         '`None` (random initialization)')

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='image_input')
    else:
        img_input = input_tensor

    # normalize input image
    img_norm = Lambda(normalize, name='input_normalize')(img_input)

    # backbone body for feature extract
    x, _, backbone_len = GhostNet(include_top=False, input_tensor=img_norm, weights=weights, OS=OS)

    # use ASPP Lite block & no decode block
    x = ASPP_Lite_block(x)

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
    model = Model(img_input, x, name='deeplabv3p_ghostnet_lite')

    return model, backbone_len



if __name__ == '__main__':
    input_tensor = Input(shape=(512, 512, 3), name='image_input')
    model, backbone_len = Deeplabv3pLiteGhostNet(input_tensor=input_tensor,
                                      weights=None,
                                      num_classes=21,
                                      OS=16)
    model.summary()
