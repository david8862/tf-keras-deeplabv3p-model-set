#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from labelme.utils import lblsave as label_save

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..'))
from common.utils import get_classes

def label_convert(input_label_path, output_label_path, class_names=None):
    if not os.path.isdir(input_label_path):
        raise ValueError('Input path does not exist!\n')
    os.makedirs(output_label_path, exist_ok=True)

    if class_names:
        # count class item number
        class_count = OrderedDict([(item, 0) for item in class_names])

    label_files = glob.glob(os.path.join(input_label_path, '*.png'))
    pbar = tqdm(total=len(label_files), desc='Label converting')
    for label_file in label_files:
        label_array = np.array(Image.open(label_file))

        if class_names:
            # count object class for statistic
            label_list = list(np.unique(label_array))
            for label in label_list:
                if label >= len(class_names):
                    continue
                class_name = class_names[label]
                class_count[class_name] = class_count[class_name] + 1

        # save numpy label array as png label image,
        # using labelme utils function
        png_file_name = os.path.basename(label_file).split('.')[0]+'.png'
        label_save(os.path.join(output_label_path, png_file_name), label_array)
        pbar.update(1)
    pbar.close()
    print('total number of converted images: ', len(label_files))

    if class_names:
        # show item number statistic
        print('Image number for each class:')
        for (class_name, number) in class_count.items():
            if class_name == 'background':
                continue
            print('%s: %d' % (class_name, number))


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='convert onboard gray PNG label to PascalVOC PNG label')
    parser.add_argument('--input_label_path', type=str, required=True, help='input path of gray label files')
    parser.add_argument('--output_label_path', type=str, required=True, help='output path of converted png label files')
    parser.add_argument('--classes_path', type=str, required=False, help='path to class definitions, optional', default=None)

    args = parser.parse_args()

    if args.classes_path:
        # add background class to match model & GT
        class_names = get_classes(args.classes_path)
        assert len(class_names) < 254, 'PNG image label only support less than 254 classes.'
    else:
        class_names = None

    label_convert(args.input_label_path, args.output_label_path, class_names)


if __name__ == '__main__':
    main()
