#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a Deeplabv3plus semantic segmentation model on test images.
"""
import colorsys
import os, sys, argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time
from timeit import default_timer as timer
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
#from tensorflow.keras.utils import multi_gpu_model
#from tensorflow_model_optimization.sparsity import keras as sparsity

from deeplabv3p.model import get_deeplabv3p_model
from deeplabv3p.postprocess_np import crf_postprocess
from common.utils import get_classes, optimize_tf_gpu, visualize_segmentation
from common.data_utils import preprocess_image, denormalize_image, mask_resize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#tf.enable_eager_execution()
optimize_tf_gpu(tf, K)

default_config = {
        "model_type": 'mobilenetv2lite',
        "classes_path": os.path.join('configs', 'voc_classes.txt'),
        "model_input_shape" : (512, 512),
        "output_stride": 16,
        "weights_path": os.path.join('weights', 'mobilenetv2_original.h5'),
        "do_crf": False,
        "pruning_model": False,
        #"gpu_num" : 1,
    }


class DeepLab(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        super(DeepLab, self).__init__()
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = get_classes(self.classes_path)
        K.set_learning_phase(0)
        self.deeplab_model = self._generate_model()

    def _generate_model(self):
        '''to generate the bounding boxes'''
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_classes = len(self.class_names)
        assert len(self.class_names) < 254, 'PNG image label only support less than 254 classes.'

        # Load model, or construct model and load weights.
        try:
            deeplab_model = get_deeplabv3p_model(self.model_type, num_classes, model_input_shape=self.model_input_shape, output_stride=self.output_stride, freeze_level=0, weights_path=weights_path, training=False)
            deeplab_model.summary()
        except Exception as e:
            print(repr(e))
        #if self.gpu_num>=2:
            #deeplab_model = multi_gpu_model(deeplab_model, gpus=self.gpu_num)

        return deeplab_model


    def segment_image(self, image):
        image_data = preprocess_image(image, self.model_input_shape)
        # origin image shape, in (height, width) format
        image_shape = tuple(reversed(image.size))

        start = time.time()
        out_mask = self.predict(image_data, image_shape)
        end = time.time()
        print("Inference time: {:.8f}s".format(end - start))

        # show segmentation result
        image_array = visualize_segmentation(np.array(image), out_mask, class_names=self.class_names)
        return Image.fromarray(image_array)


    def predict(self, image_data, image_shape):
        prediction = self.deeplab_model.predict([image_data])
        # reshape prediction to mask array
        mask = np.argmax(prediction, -1)[0].reshape(self.model_input_shape)

        # add CRF postprocess if need
        if self.do_crf:
            image = denormalize_image(image_data[0])
            mask = crf_postprocess(image, mask, zero_unsure=False)

        # resize mask back to origin image size
        mask = mask_resize(mask, image_shape[::-1])

        return mask


    def dump_model_file(self, output_model_file):
        self.deeplab_model.save(output_model_file)

    def dump_saved_model(self, saved_model_path):
        model = self.deeplab_model
        os.makedirs(saved_model_path, exist_ok=True)

        tf.keras.experimental.export_saved_model(model, saved_model_path)
        print('export inference model to %s' % str(saved_model_path))


def segment_video(deeplab, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(0 if video_path == '0' else video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    # here we encode the video to MPEG-4 for better compatibility, you can use ffmpeg later
    # to convert it to x264 to reduce file size:
    # ffmpeg -i test.mp4 -vcodec libx264 -f mp4 test_264.mp4
    #
    #video_FourCC    = cv2.VideoWriter_fourcc(*'XVID') if video_path == '0' else int(vid.get(cv2.CAP_PROP_FOURCC))
    video_FourCC    = cv2.VideoWriter_fourcc(*'XVID') if video_path == '0' else cv2.VideoWriter_fourcc(*"mp4v")
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, (5. if video_path == '0' else video_fps), video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = deeplab.segment_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release everything if job is finished
    vid.release()
    if isOutput:
        out.release()
    cv2.destroyAllWindows()


def segment_img(deeplab):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = deeplab.segment_image(image)
            r_image.show()


if __name__ == '__main__':
    # class DeepLab defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='demo or dump out Deeplab h5 model')
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_type', type=str,
        help='Deeplabv3p model type: mobilenetv2/xception, default ' + DeepLab.get_defaults("model_type")
    )

    parser.add_argument(
        '--weights_path', type=str,
        help='path to model weight file, default ' + DeepLab.get_defaults("weights_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + DeepLab.get_defaults("classes_path")
    )

    parser.add_argument(
        '--model_input_shape', type=str,
        help='model input shape as <height>x<width>, default ' +
        str(DeepLab.get_defaults("model_input_shape")[0])+'x'+str(DeepLab.get_defaults("model_input_shape")[1]),
        default=str(DeepLab.get_defaults("model_input_shape")[0])+'x'+str(DeepLab.get_defaults("model_input_shape")[1])
    )

    parser.add_argument(
        '--output_stride', type=int, choices=[8, 16, 32],
        help='model output stride, default ' + str(DeepLab.get_defaults("output_stride"))
    )

    parser.add_argument(
        '--do_crf', default=False, action="store_true",
        help='whether to add CRF postprocess for model output, default ' + str(DeepLab.get_defaults("do_crf"))
    )

    #parser.add_argument(
        #'--pruning_model', default=False, action="store_true",
        #help='Whether to be a pruning model/weights file')

    #parser.add_argument(
        #'--gpu_num', type=int,
        #help='Number of GPU to use, default ' + str(DeepLab.get_defaults("gpu_num"))
    #)
    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image inference mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    '''
    Command line positional arguments -- for model dump
    '''
    parser.add_argument(
        '--dump_model', default=False, action="store_true",
        help='Dump out training model to inference model'
    )

    parser.add_argument(
        '--output_model_file', type=str,
        help='output inference model file'
    )

    args = parser.parse_args()
    # param parse
    if args.model_input_shape:
        height, width = args.model_input_shape.split('x')
        args.model_input_shape = (int(height), int(width))

    # get wrapped inference object
    deeplab = DeepLab(**vars(args))

    if args.dump_model:
        """
        Dump out training model to inference model
        """
        if not args.output_model_file:
            raise ValueError('output model file is not specified')

        print('Dumping out training model to inference model')
        deeplab.dump_model_file(args.output_model_file)
        sys.exit()

    if args.image:
        """
        Image segmentation mode, disregard any remaining command line arguments
        """
        print("Image segmentation mode")
        if "input" in args:
            print(" Ignoring remaining command line arguments: " + args.input + "," + args.output)
        segment_img(deeplab)
    elif "input" in args:
        segment_video(deeplab, args.input, args.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
