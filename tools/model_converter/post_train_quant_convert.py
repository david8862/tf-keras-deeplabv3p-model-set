#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert Deeplab keras model to an integer quantized tflite model
using latest Post-Training Integer Quantization Toolkit released in
tensorflow 2.0.0 build
"""
import os, sys, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from deeplabv3p.data import SegmentationGenerator
from common.utils import get_data_list, get_custom_objects

#tf.enable_eager_execution()


def post_train_quant_convert(keras_model_file, dataset_path, dataset, sample_num, model_input_shape, output_file):
    #get input_shapes for converter
    input_shapes=list((1,)+model_input_shape+(3,))

    #prepare quant data generator
    data_gen = SegmentationGenerator(dataset_path, dataset,
                                            1,  #batch_size
                                            1,  #num_classes, here we don't really use it
                                            target_size=model_input_shape[::-1],
                                            weighted_type=None,
                                            is_eval=False,
                                            augment=True)


    custom_object_dict = get_custom_objects()
    model = load_model(keras_model_file, custom_objects=custom_object_dict)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    def data_generator():
        i = 0
        for n, (image_data, y_true) in enumerate(data_gen):
            i += 1
            if i > sample_num:
                break

            yield [image_data]


    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    converter.representative_dataset = tf.lite.RepresentativeDataset(data_generator)

    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()
    with open(output_file, "wb") as f:
        f.write(tflite_model)



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='TF 2.x post training integer quantization converter')

    parser.add_argument('--keras_model_file', required=True, type=str, help='path to keras model file')
    parser.add_argument('--dataset_path', required=True, type=str, help='dataset path containing images and label png file')
    parser.add_argument('--dataset_file', required=True, type=str, help='data samples txt file')
    parser.add_argument('--sample_num', type=int, help='annotation sample number to feed the converter,default=%(default)s', default=30)
    parser.add_argument('--model_input_shape', type=str, help='model image input shape as <height>x<width>, default=%(default)s', default='512x512')
    parser.add_argument('--output_file', required=True, type=str, help='output tflite model file')

    args = parser.parse_args()
    height, width = args.model_input_shape.split('x')
    model_input_shape = (int(height), int(width))

    # get dataset list
    dataset = get_data_list(args.dataset_file)

    post_train_quant_convert(args.keras_model_file, args.dataset_path, dataset, args.sample_num, model_input_shape, args.output_file)



if __name__ == '__main__':
    main()

