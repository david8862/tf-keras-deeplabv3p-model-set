#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Miscellaneous utility functions."""

import os
import numpy as np
import copy
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec

from deeplabv3p.models.layers import normalize, img_resize
from deeplabv3p.models.deeplabv3p_mobilenetv3 import hard_sigmoid, hard_swish
import tensorflow as tf


def optimize_tf_gpu(tf, K):
    if tf.__version__.startswith('2'):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    #tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
        config.gpu_options.per_process_gpu_memory_fraction = 0.9  #GPU memory threshold 0.3
        session = tf.Session(config=config)

        # set session
        K.set_session(session)


def get_custom_objects():
    '''
    form up a custom_objects dict so that the customized
    layer/function call could be correctly parsed when keras
    .h5 model is loading or converting
    '''
    custom_objects_dict = {
        'tf': tf,
        'normalize': normalize,
        'img_resize': img_resize,
        'hard_sigmoid': hard_sigmoid,
        'hard_swish': hard_swish,
    }
    return custom_objects_dict

"""
def calculate_weigths_labels(dataset_generator, num_classes, save_path=None):
    '''
    calculate a static segment classes (including background) weights
    coefficient based on class pixel
    '''
    # Initialize class count list array
    class_counts = np.zeros((num_classes,))

    # collecting class pixel count
    pbar = tqdm(total=len(dataset_generator), desc='Calculating classes weights')
    for n, (_, y) in enumerate(dataset_generator):
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        class_counts += count_l
        pbar.update(1)
    pbar.close()
    # sum() to get total valid pixel count
    total_count = np.sum(class_counts)
    # get class weights with 1/(log(1.02+(class_count/total_count)))
    class_weights = []
    for class_count in class_counts:
        class_weight = 1 / (np.log(1.02 + (class_count / total_count)))
        class_weights.append(class_weight)

    class_weights = np.array(class_weights)
    # save class weights array to file for reloading next time
    if save_path:
        classes_weights_path = os.path.join(save_path, 'classes_weights.npy')
        np.save(classes_weights_path, class_weights)

    return class_weights
"""


def calculate_weigths_labels(dataset_generator, num_classes, save_path=None):
    '''
    calculate a static segment classes (including background) weights
    coefficient based on class pixel
    '''
    # Initialize class count list array
    class_counts = np.zeros((num_classes,))

    # collecting class pixel count
    pbar = tqdm(total=len(dataset_generator), desc='Calculating classes weights')
    for n, (_, y) in enumerate(dataset_generator):
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        class_counts += count_l
        pbar.update(1)
    pbar.close()
    # sum() to get total valid pixel count
    total_count = np.sum(class_counts)

    #
    # use following formula to calculate balanced class weights:
    # class_weights = sample_count / (num_classes * np.bincount(labels))
    #
    # which is same as
    # class_weight.compute_class_weight('balanced', class_list, y)
    #
    class_weights = total_count / (num_classes * class_counts)
    class_weights = np.array(class_weights)
    # save class weights array to file for reloading next time
    if save_path:
        classes_weights_path = os.path.join(save_path, 'classes_weights.txt')
        save_class_weights(classes_weights_path, class_weights)

    return class_weights


def save_class_weights(save_path, class_weights):
    '''
    save class weights array with shape (num_classes,)
    '''
    weights_file = open(save_path, 'w')
    for class_weight in list(class_weights):
            weights_file.write(str(class_weight))
            weights_file.write('\n')
    weights_file.close()


def load_class_weights(classes_weights_path):
    '''
    load saved class weights txt file and convert
    to numpy array with shape (num_classes,)
    '''
    with open(classes_weights_path) as f:
        classes_weights = f.readlines()
    classes_weights = [float(c.strip()) for c in classes_weights]

    return np.array(classes_weights)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_data_list(data_list_file, shuffle=True):
    with open(data_list_file) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    if shuffle:
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)

    return lines


def figure_to_image(figure):
    '''
    Convert a Matplotlib figure to a Pillow image with RGBA channels

    # Arguments
        figure: matplotlib figure
                usually create with plt.figure()

    # Returns
        image: numpy array image
    '''
    # draw the renderer
    figure.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = figure.canvas.get_width_height()
    buf = np.fromstring(figure.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    # Convert RGBA to RGB
    image = np.asarray(image)[..., :3]
    return image


def create_pascal_label_colormap():
    """
    create label colormap with PASCAL VOC segmentation dataset definition

    # Returns
        colormap: Colormap array for visualizing segmentation
    """
    colormap = np.zeros((256, 3), dtype=int)
    index = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((index >> channel) & 1) << shift
        index >>= 3

    return colormap


def label_to_color_image(label):
    """
    mapping the segmentation label to color indexing array

    # Arguments
        label: 2D uint8 numpy array, with segmentation label

    # Returns
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PascalVOC color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def visualize_segmentation(image, mask, gt_mask=None, class_names=None, overlay=0.7, ignore_count_threshold=100, title=None, gt_title=None):
    """
    Visualize segmentation mask on input image, using PascalVOC
    Segmentation color map

    # Arguments
        image: image array
            numpy array for input image
        mask: predict mask array
            2D numpy array for predict segmentation mask
        gt_mask: ground truth mask array
            2D numpy array for gt segmentation mask
        class_names: label class definition
            list of label class names
        ignore_count_threshold: threshold to filter label
            integer scalar to filter the label value with small count
        title: predict segmentation title
            title string for predict segmentation result plot
        gt_title: ground truth segmentation title
            title string for ground truth segmentation plot

    # Returns
        img: A numpy image with segmentation result
    """
    if (gt_mask is not None) and (class_names is not None):
        grid_spec = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 1])
        figsize = (15, 10)
    elif (gt_mask is not None) and (class_names is None):
        grid_spec = gridspec.GridSpec(1, 2, width_ratios=[6, 6])
        figsize = (15, 10)
    elif (gt_mask is None) and (class_names is not None):
        grid_spec = gridspec.GridSpec(1, 2, width_ratios=[6, 1])
        figsize = (10, 10)
    else:
        grid_spec = [111]
        figsize = (10, 10)

    figure = plt.figure(figsize=figsize)

    # convert mask array to color mapped image
    mask_image = label_to_color_image(mask).astype(np.uint8)
    # show segmentation result image
    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.imshow(mask_image, alpha=overlay)
    plt.axis('off')
    # add plt title, optional
    if title:
        plt.title(title)

    if gt_mask is not None:
        # reset invalid label value as 0(background)
        filtered_gt_mask = copy.deepcopy(gt_mask)
        filtered_gt_mask[filtered_gt_mask>len(class_names)-1] = 0
        # convert gt mask array to color mapped image
        gt_mask_image = label_to_color_image(filtered_gt_mask).astype(np.uint8)
        # show gt segmentation image
        plt.subplot(grid_spec[1])
        plt.imshow(image)
        plt.imshow(gt_mask_image, alpha=overlay)
        plt.axis('off')
        # add plt title, optional
        if gt_title:
            plt.title(gt_title)

    # if class name list is provided, plot a legend graph of
    # classes color map
    if class_names:
        classes_index = np.arange(len(class_names)).reshape(len(class_names), 1)
        classes_color_map = label_to_color_image(classes_index)

        labels, count= np.unique(mask, return_counts=True)
        # filter some corner pixel labels, may be caused by mask resize
        labels = np.array([labels[i] for i in range(len(labels)) if count[i] > ignore_count_threshold])

        if gt_mask is not None:
            gt_labels, gt_count= np.unique(filtered_gt_mask, return_counts=True)
            # filter some corner pixel labels, may be caused by mask resize
            gt_labels = np.array([gt_labels[i] for i in range(len(gt_labels)) if gt_count[i] > ignore_count_threshold])

            # merge labels & gt labels
            labels = list(set(list(labels)+list(gt_labels)))
            labels.sort()
            labels = np.array(labels)

        ax = plt.subplot(grid_spec[-1])
        plt.imshow(classes_color_map[labels].astype(np.uint8), interpolation='nearest')

        # adjust subplot display
        ax.yaxis.tick_right()
        plt.yticks(range(len(labels)), np.asarray(class_names)[labels])
        plt.xticks([], [])
        ax.tick_params(width=0.0)
        plt.grid('off')

    # convert plt to numpy image
    img = figure_to_image(figure)
    plt.close("all")
    return img

