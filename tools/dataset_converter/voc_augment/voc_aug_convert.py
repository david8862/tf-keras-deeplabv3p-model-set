#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import glob
import numpy as np
from PIL import Image
import scipy.io
from tqdm import tqdm
from collections import OrderedDict
from labelme.utils import lblsave as label_save

PASCAL_VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                      'bottle', 'bus', 'car', 'cat', 'chair',
                      'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                      'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


# extract numpy array from mat for http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html
# 'GTcls' key is for class segmentation
# 'GTinst' key is for instance segmentation
def get_array_from_mat(mat_file, label_type):
    if label_type == 'semantic':
        key = 'GTcls'
    elif label_type == 'instance':
        key = 'GTinst'
    else:
        raise ValueError('invalid label type')

    mat = scipy.io.loadmat(mat_file, mat_dtype=True, squeeze_me=True, struct_as_record=False)
    return mat[key].Segmentation


#def get_pascal_palette(png_file):
    #'''
    #extract palette info from a sample PascalVOC
    #segmentation label png file
    #'''
    #label_img = Image.open(png_file)
    #palette = label_img.getpalette()
    #return palette


#def convert_to_color_seg(label_array, palette):
    #'''
    #set palette to generate a colorful png label file
    #'''
    #label_img = Image.fromarray(label_array, mode='P')
    #label_img.putpalette(palette)
    #return label_img


def label_convert(mat_label_path, png_label_path, label_type):
    if not os.path.isdir(mat_label_path):
        raise ValueError('Input path does not exist!\n')
    os.makedirs(png_label_path, exist_ok=True)

    # count class item number
    class_count = OrderedDict([(item, 0) for item in PASCAL_VOC_CLASSES])

    mat_files = glob.glob(os.path.join(mat_label_path, '*.mat'))
    pbar = tqdm(total=len(mat_files), desc='Label converting')
    for mat_file in mat_files:
        label_array = get_array_from_mat(mat_file, label_type)

        # count object class for statistic
        label_list = list(np.unique(label_array))
        for label in label_list:
            class_name = PASCAL_VOC_CLASSES[label]
            class_count[class_name] = class_count[class_name] + 1

        # save numpy label array as png label image,
        # using labelme utils function
        png_file_name = os.path.basename(mat_file).split('.')[0]+'.png'
        label_save(os.path.join(png_label_path, png_file_name), label_array)
        pbar.update(1)

    pbar.close()
    # show item number statistic
    print('Image number for each class:')
    for (class_name, number) in class_count.items():
        if class_name == 'background':
            continue
        print('%s: %d' % (class_name, number))
    print('total number of converted images: ', len(mat_files))


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='convert Augmented PascalVOC .mat label to .png label')
    parser.add_argument('--mat_label_path', required=True, type=str, help='path of mat label files in Augmented PascalVOC dataset')
    parser.add_argument('--png_label_path', required=True, type=str, help='output path of converted png label files')
    parser.add_argument('--label_type', required=False, type=str, default='semantic', choices=['semantic', 'instance'], help='label type: semantic/instance, default=%(default)s')

    args = parser.parse_args()

    label_convert(args.mat_label_path, args.png_label_path, args.label_type)



if __name__ == '__main__':
    main()

