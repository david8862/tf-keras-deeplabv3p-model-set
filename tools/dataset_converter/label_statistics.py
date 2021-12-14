#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_classes, get_data_list

def label_stat(label_path, dataset_file, class_names):
    if not os.path.isdir(label_path):
        raise ValueError('Input path does not exist!\n')

    if dataset_file:
        # get dataset sample list
        dataset = get_data_list(dataset_file)
        png_files = [os.path.join(label_path, image_id.strip()+'.png') for image_id in dataset]
    else:
        png_files = glob.glob(os.path.join(label_path, '*.png'))

    num_classes = len(class_names)

    # add "ignore" label for count
    class_names += ['ignore']

    # count class item number
    class_count = OrderedDict([(item, 0) for item in class_names])
    valid_number = 0

    pbar = tqdm(total=len(png_files), desc='Labels checking')
    for png_file in png_files:
        label_array = np.array(Image.open(png_file))
        # treat all the invalid label value as a new value (ignore)
        label_array[label_array>(num_classes-1)] = num_classes

        # count object class for statistic
        label_list = list(np.unique(label_array))
        if sum(label_list) > 0:
            valid_number += 1
        for label in label_list:
            class_name = class_names[label]
            class_count[class_name] = class_count[class_name] + 1
        pbar.update(1)

    pbar.close()
    # show item number statistic
    print('Image number for each class:')
    for (class_name, number) in class_count.items():
        if class_name == 'background':
            continue
        print('%s: %d' % (class_name, number))
    print('total number of valid label image: ', valid_number)


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='label statistic info of dataset')
    parser.add_argument('--label_path', required=True, type=str, help='path to png label images')
    parser.add_argument('--classes_path', required=True, type=str, help='path to class definitions')
    parser.add_argument('--dataset_file', required=False, type=str, default=None, help='dataset txt file')

    args = parser.parse_args()
    # prepare class name list
    class_names = get_classes(args.classes_path)
    assert len(class_names) < 254, 'PNG image label only support less than 254 classes.'

    label_stat(args.label_path, args.dataset_file, class_names)



if __name__ == '__main__':
    main()

