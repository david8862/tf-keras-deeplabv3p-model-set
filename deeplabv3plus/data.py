#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, time
import random
import numpy as np
import cv2
from PIL import Image

from sklearn.utils import class_weight
from tensorflow.keras.utils import Sequence

from common.data_utils import random_horizontal_flip, random_vertical_flip, random_brightness, random_grayscale, random_chroma, random_contrast, random_sharpness, random_blur, random_zoom_rotate, random_crop


class SegmentationGenerator(Sequence):
    def __init__(self, dataset_path, data_list,
                 batch_size=1,
                 num_classes=21,
                 resize_shape=None,
                 crop_shape=(640, 320),
                 weighted_type=None,
                 is_eval=False,
                 augment=True,
                 do_ahisteq=True):

        # get real path for dataset
        dataset_realpath = os.path.realpath(dataset_path)

        self.image_path_list = [os.path.join(dataset_realpath, 'images', image_id.strip()+'.jpg') for image_id in data_list]
        self.label_path_list = [os.path.join(dataset_realpath, 'labels', image_id.strip()+'.png') for image_id in data_list]

        # initialize random seed
        np.random.seed(int(time.time()))

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.weighted_type = weighted_type
        self.augment = augment
        self.is_eval = is_eval

        # prepare "Contrast Limited Adaptive Histogram Equalization" in cv2
        self.histeq = do_ahisteq
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # Preallocate memory
        if self.crop_shape:
            self.X = np.zeros((batch_size, crop_shape[1], crop_shape[0], 3), dtype='float32')
            self.Y = np.zeros((batch_size, crop_shape[1]*crop_shape[0], 1), dtype='float32')
            self.PIXEL_WEIGHTS = np.zeros((batch_size, crop_shape[1]*crop_shape[0]), dtype='float32')
        elif self.resize_shape:
            self.X = np.zeros((batch_size, resize_shape[1], resize_shape[0], 3), dtype='float32')
            self.Y = np.zeros((batch_size, resize_shape[1]*resize_shape[0], 1), dtype='float32')
            self.PIXEL_WEIGHTS = np.zeros((batch_size, resize_shape[1]*resize_shape[0]), dtype='float32')
        else:
            raise Exception('No image dimensions specified!')

    def get_batch_image_path(self, i):
        return self.image_path_list[i*self.batch_size:(i+1)*self.batch_size]

    def get_batch_label_path(self, i):
        return self.label_path_list[i*self.batch_size:(i+1)*self.batch_size]

    def get_weighted_type(self):
        return self.weighted_type

    def __len__(self):
        return len(self.image_path_list) // self.batch_size

    def __getitem__(self, i):

        for n, (image_path, label_path) in enumerate(zip(self.image_path_list[i*self.batch_size:(i+1)*self.batch_size],
                                                        self.label_path_list[i*self.batch_size:(i+1)*self.batch_size])):

            # Load image and label array
            image = cv2.imread(image_path, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR/cv2.IMREAD_GRAYSCALE/cv2.IMREAD_UNCHANGED
            label = np.array(Image.open(label_path))

            # we reset all the invalid label value as 0(background) in training,
            # but as 255(invalid) in eval
            if self.is_eval:
                label[label>(self.num_classes-1)] = 255
            else:
                label[label>(self.num_classes-1)] = 0

            # Resize original image & label mask to model input shape
            if self.resize_shape and not self.crop_shape:
                image = cv2.resize(image, self.resize_shape)
                label = cv2.resize(label, self.resize_shape, interpolation = cv2.INTER_NEAREST)

            # Do augmentation
            if self.augment:
                # random horizontal flip image
                image, label = random_horizontal_flip(image, label)

                # random vertical flip image
                image, label = random_vertical_flip(image, label)

                # random zoom & rotate image
                image, label = random_zoom_rotate(image, label)

                # random adjust brightness
                image = random_brightness(image)

                # random adjust color level
                image = random_chroma(image)

                # random adjust contrast
                image = random_contrast(image)

                # random adjust sharpness
                image = random_sharpness(image)

                # random convert image to grayscale
                image = random_grayscale(image)

                # random do gaussian blur to image
                image = random_blur(image)

                # random crop image & label if needed
                if self.crop_shape:
                    image, label = random_crop(image, label, self.crop_shape)

            # Histogram equalization using CLAHE
            if self.histeq:
                img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = self.clahe.apply(img_yuv[:,:,0])
                image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) # to BGR

            label = label.astype('int32')
            y = label.flatten()

            # we reset all the invalid label value as 0(background) in training,
            # but as 255(invalid) in eval
            if self.is_eval:
                y[y>(self.num_classes-1)] = 255
            else:
                y[y>(self.num_classes-1)] = 0

            # append input image and label array
            self.X[n] = image
            self.Y[n]  = np.expand_dims(y, -1)

            ###########################################################################
            #
            # generating adaptive pixels weights array, for unbalanced classes training
            #
            ###########################################################################

            # Create adaptive pixels weights for all classeson one image,
            # according to pixel number of classes
            class_list = np.unique(y)
            if len(class_list):
                class_weights = class_weight.compute_class_weight('balanced', class_list, y)
                class_weights = {class_id : weight for class_id , weight in zip(class_list, class_weights)}
            # class_weigts dict would be like:
            # {
            #    0: 0.5997304983036035,
            #   12: 2.842871240958237,
            #   15: 1.0195474451419193
            # }
            for class_id in class_list:
                np.putmask(self.PIXEL_WEIGHTS[n], y==class_id, class_weights[class_id])

        # A trick of keras data generator: the last item yield
        # from a generator could be a sample weights array
        sample_weight_dict = {'pred_mask' : self.PIXEL_WEIGHTS}

        if self.weighted_type == 'adaptive':
            return self.X, self.Y, sample_weight_dict
        else:
            return self.X, self.Y

    def on_epoch_end(self):
        # Shuffle dataset for next epoch
        c = list(zip(self.image_path_list, self.label_path_list))
        random.shuffle(c)
        self.image_path_list, self.label_path_list = zip(*c)

