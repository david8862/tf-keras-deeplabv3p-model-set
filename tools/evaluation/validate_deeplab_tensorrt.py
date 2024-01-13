#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import time
import glob
import numpy as np
from PIL import Image
import tensorrt as trt
from cuda import cudart  # install with "pip install cuda-python"

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.data_utils import preprocess_image, denormalize_image, mask_resize
from common.utils import get_classes, visualize_segmentation
from deeplabv3p.metrics import mIOU
from deeplabv3p.postprocess_np import crf_postprocess


def validate_deeplab_model_tensorrt(engine, image_file, class_names, do_crf, label_file, loop_count, output_path):
    # get I/O tensor info
    num_io = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(num_io)]

    # get input tensor details
    input_tensor_names = [tensor_name for tensor_name in tensor_names if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT]
    input_tensor_shapes = [engine.get_tensor_shape(input_tensor_name) for input_tensor_name in input_tensor_names]
    num_input = len(input_tensor_names)

    # assume only 1 input tensor
    assert num_input == 1, 'invalid input tensor number.'

    # check if input layout is NHWC or NCHW
    if input_tensor_shapes[0][1] == 3:
        print("NCHW input layout")
        batch, channel, height, width = input_tensor_shapes[0]  #NCHW
    else:
        print("NHWC input layout")
        batch, height, width, channel = input_tensor_shapes[0]  #NHWC

    # get model input shape
    model_input_shape = (height, width)

    # get output tensor details
    output_tensor_names = [tensor_name for tensor_name in tensor_names if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT]
    output_tensor_shapes = [engine.get_tensor_shape(output_tensor_name) for output_tensor_name in output_tensor_names]
    num_output = len(output_tensor_names)

    # assume only 1 output tensor
    assert num_output == 1, 'invalid output tensor number.'

    # check if output layout is NHWC or NCHW, (H,W) for Deeplab
    # output tensor should be same as input tensor
    if output_tensor_shapes[0][-1] == width:
        print("NCHW output layout")
        num_classes = output_tensor_shapes[0][1]  #NCHW
    elif output_tensor_shapes[0][1] == height:
        print("NHWC output layout")
        num_classes = output_tensor_shapes[0][-1]  #NHWC
    else:
        raise ValueError('invalid output layout or shape')

    if class_names:
        # check if classes number match with model prediction
        assert num_classes == len(class_names), 'classes number mismatch with model.'

    # create engine execution context
    context = engine.create_execution_context()
    context.set_optimization_profile_async(0, 0)  # use default stream

    # prepare memory buffer on host
    buffer_host = []
    for i in range(num_io):
        buffer_host.append(np.empty(context.get_tensor_shape(tensor_names[i]), dtype=trt.nptype(engine.get_tensor_dtype(tensor_names[i]))))

    # prepare memory buffer on GPU device
    buffer_device = []
    for i in range(num_io):
        buffer_device.append(cudart.cudaMalloc(buffer_host[i].nbytes)[1])

    # set address of all input & output data in device buffer
    for i in range(num_io):
        context.set_tensor_address(tensor_names[i], int(buffer_device[i]))

    # prepare input image
    img = Image.open(image_file).convert('RGB')
    image_data = preprocess_image(img, model_input_shape)
    image = denormalize_image(image_data[0])
    #origin image shape, in (width, height) format
    origin_image_size = img.size

    if input_tensor_shapes[0][1] == 3:
        # transpose image for NCHW layout
        image_data = image_data.transpose((0,3,1,2))

    # fill image data to host buffer
    buffer_host[0] = np.ascontiguousarray(image_data)

    # copy input data from host buffer to device buffer
    for i in range(num_input):
        cudart.cudaMemcpy(buffer_device[i], buffer_host[i].ctypes.data, buffer_host[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # do inference computation
    start = time.time()
    for i in range(loop_count):
        context.execute_async_v3(0)
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    # copy output data from device buffer into host buffer
    for i in range(num_input, num_io):
        cudart.cudaMemcpy(buffer_host[i].ctypes.data, buffer_device[i], buffer_host[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    prediction = [buffer_host[1]]

    if output_tensor_shapes[0][1] == num_classes:
        # transpose predict mask for NCHW layout
        prediction = [p.transpose((0,2,3,1)) for p in prediction]

    handle_prediction(prediction, image, np.array(img), image_file, num_classes, class_names, model_input_shape, origin_image_size, do_crf, label_file, output_path)

    # free GPU memory buffer after all work
    for buffer in buffer_device:
        cudart.cudaFree(buffer)


def handle_prediction(prediction, image, origin_image, image_file, num_classes, class_names, model_input_shape, origin_image_size, do_crf, label_file, output_path):
    # generate prediction mask,
    # add CRF postprocess if need
    prediction = np.argmax(prediction, -1)[0].reshape(model_input_shape)
    if do_crf:
        prediction = crf_postprocess(image, prediction, zero_unsure=False)
    prediction = mask_resize(prediction, origin_image_size)

    title_str = None
    label = None
    gt_title_str = None
    # calculate mIOU if having label image
    if label_file:
        label = np.array(Image.open(label_file), dtype='int32')
        # treat all the invalid label value as 255
        label[label>(num_classes-1)] = 255
        title_str = 'Predict Segmentation\nmIOU: '+str(mIOU(label, prediction))
        gt_title_str = 'GT Segmentation'

    image_array = visualize_segmentation(origin_image, prediction, label, class_names=class_names, title=title_str, gt_title=gt_title_str)

    # save or show result
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, os.path.basename(image_file))
        Image.fromarray(image_array).save(output_file)
    else:
        Image.fromarray(image_array).show()
    return


def load_engine(model_path):
    # support TensorRT engine model
    if model_path.endswith('.engine'):
        # load model & create engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, mode='rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
    else:
        raise ValueError('invalid model file')

    return engine


def main():
    parser = argparse.ArgumentParser(description='validate Deeplab TensorRT model (.engine) with image')
    parser.add_argument('--model_path', help='model file to predict', type=str, required=True)
    parser.add_argument('--image_path', help='image file or directory to predict', type=str, required=True)
    #parser.add_argument('--model_input_shape', help='model image input shape as <height>x<width>, default=%(default)s', type=str, default='512x512')
    parser.add_argument('--do_crf', action="store_true", help='whether to add CRF postprocess for model output', default=False)
    parser.add_argument('--label_file', help='segmentation label image file', type=str, required=False, default=None)
    parser.add_argument('--classes_path', help='path to class name definitions', type=str, required=False)
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)
    parser.add_argument('--output_path', help='output path to save predict result, default=%(default)s', type=str, required=False, default=None)

    args = parser.parse_args()

    class_names = None
    if args.classes_path:
        # get class names
        class_names = get_classes(args.classes_path)
        assert len(class_names) < 254, 'PNG image label only support less than 254 classes.'

    # param parse
    #height, width = args.model_input_shape.split('x')
    #model_input_shape = (int(height), int(width))

    # prepare environment
    assert trt.__version__ >= '8.5', 'invalid TensorRT version'
    cudart.cudaDeviceSynchronize()

    # load TensorRT engine
    engine = load_engine(args.model_path)


    # get image file list or single image
    if os.path.isdir(args.image_path):
        image_files = glob.glob(os.path.join(args.image_path, '*'))
        assert args.output_path, 'need to specify output path if you use image directory as input.'
        assert args.label_file == None, 'label file only used for single image validation.'
    else:
        image_files = [args.image_path]

    # loop the sample list to predict on each image
    for image_file in image_files:
        validate_deeplab_model_tensorrt(engine, image_file, class_names, args.do_crf, args.label_file, args.loop_count, args.output_path)



if __name__ == '__main__':
    main()
