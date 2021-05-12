#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import glob
import numpy as np
from PIL import Image
#import scipy.io
from tqdm import tqdm
#from collections import OrderedDict
from labelme.utils import lblsave as label_save


def gray_label_convert(input_path, output_path):
    if not os.path.isdir(input_path):
        raise ValueError('Input path does not exist!\n')
    os.makedirs(output_path, exist_ok=True)

    # count class item number
    #class_count = OrderedDict([(item, 0) for item in PASCAL_VOC_CLASSES])

    label_files = glob.glob(os.path.join(input_path, '*.png'))
    pbar = tqdm(total=len(label_files), desc='Label image converting')
    for label_file in label_files:
        label_array = np.asarray(Image.open(label_file))

        # count object class for statistic
        #label_list = list(np.unique(label_array))
        #for label in label_list:
            #class_name = PASCAL_VOC_CLASSES[label]
            #class_count[class_name] = class_count[class_name] + 1

        # save numpy label array as png label image,
        # using labelme utils function
        label_file_name = os.path.basename(label_file)
        label_save(os.path.join(output_path, label_file_name), label_array)
        pbar.update(1)

    pbar.close()
    # show item number statistic
    #print('Image number for each class:')
    #for (class_name, number) in class_count.items():
        #if class_name == 'background':
            #continue
        #print('%s: %d' % (class_name, number))
    print('total number of converted images: ', len(label_files))


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='convert grayscale .png label image to PascalVOC style color .png label image')
    parser.add_argument('--input_path', required=True, type=str, help='path to grayscale png label images')
    parser.add_argument('--output_path', required=True, type=str, help='path to color png label images')

    args = parser.parse_args()
    gray_label_convert(args.input_path, args.output_path)



if __name__ == '__main__':
    main()

