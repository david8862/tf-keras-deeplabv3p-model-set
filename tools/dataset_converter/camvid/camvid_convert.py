#!/usr/bin/python3
# -*- coding=utf-8 -*-
import os, sys, argparse
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
#from collections import OrderedDict
from labelme.utils import lblsave as label_save

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..'))
from common.utils import get_classes


class LabelProcessor(object):
    """
    CamVid color label image processor class. it build a hash table
    for mapping color label image to numpy label array

    Reference: https://zhuanlan.zhihu.com/p/293112559
    """
    def __init__(self, color_map_file):
        # load class names and color map from color_map dict file
        self.class_names, self.colormap = self.read_color_map(color_map_file)
        # encoding color map to get a color-label hash table
        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def read_color_map(color_map_file):
        """
        load CamVid .csv color map dict file to get all class names and
        color maps. The .csv file is like following format:

        name,r,g,b
        Animal,64,128,64
        Archway,192,0,128
        Bicyclist,0,128, 192
        Bridge,0, 128, 64
        Building,128, 0, 0
        Car,64, 0, 128
        ...

        """
        pd_label_color = pd.read_csv(color_map_file, sep=',')
        class_names = []
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            # use lower char for class name string
            class_name = str(tmp['name']).lower()
            color = [tmp['r'], tmp['g'], tmp['b']]

            class_names.append(class_name)
            colormap.append(color)

        return class_names, colormap


    @staticmethod
    def encode_label_pix(colormap):
        """
        encoding label image pixel color,
        build color-to-label hash table with following hash mapping:

        Hash function : (cm[0]*256+cm[1])*256+cm[2]
        Hash mapping  : cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i
        Hash table    : cm2lbl
        """
        # (R,G,B): (0-255, 0-255, 0-255), so color space = 256*256*256
        cm2lbl = np.zeros(256**3)

        for i,cm in enumerate(colormap):
            # mapping (R,G,B) color map value to space index via hash function,
            # and assign cm2lbl[index] to class label i
            cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i

        return cm2lbl


    def encode_label_img(self, img):
        """
        encode color image array to label array with hash table:

        (R,G,B) -> hash table index -> class label
        """
        # convert to numpy image array, shape == (H,W,3)
        data = np.array(img, dtype='int32')
        # apply hash function to get hash table index,
        # idx.shape == (H,W)
        idx=(data[:,:,0]*256+data[:,:,1])*256+data[:,:,2]

        # get class label array with cm2lbl[idx], shape == (H,W)
        return np.array(self.cm2lbl[idx], dtype='int64')


def class_label_convert(label_array, full_class_names, class_names):
    label = label_array.copy()

    for i, full_class_name in enumerate(full_class_names):
        if full_class_name in class_names:
            # for selected class, get its new label
            class_label = class_names.index(full_class_name)
        else:
            # for non-selected class, use 'void' (0) label
            #class_label = 254
            class_label = class_names.index('void')

        label[label_array == i] = class_label

    # use 254 to mark any invalid label
    label[label>(len(class_names)-1)] = 254

    return label


def camvid_convert(label_path, class_dict_path, class_names, output_path):
    if not os.path.isdir(label_path):
        raise ValueError('Input path does not exist!\n')
    os.makedirs(output_path, exist_ok=True)

    # create label encoding object and get full class names
    label_processor = LabelProcessor(class_dict_path)
    full_class_names = label_processor.class_names

    color_label_files = glob.glob(os.path.join(label_path, '*.png'))
    pbar = tqdm(total=len(color_label_files), desc='Color label image converting')
    for color_label_file in color_label_files:
        # convert color image array (H,W,3) to label array (H,W)
        label = Image.open(color_label_file)
        label_array = label_processor.encode_label_img(label)

        # transfer full classes label to selected classes label, if needed
        if class_names:
            label_array = class_label_convert(label_array, full_class_names, class_names)

        # save numpy label array as png label image,
        # using labelme utils function
        label_file_name = os.path.basename(color_label_file)
        label_save(os.path.join(output_path, label_file_name), label_array)
        pbar.update(1)

    pbar.close()
    print('total number of converted images: ', len(color_label_files))



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='convert CamVid color label image to PascalVOC style .png label')
    parser.add_argument('--label_path', required=True, type=str, help='path of CamVid color image labels')
    parser.add_argument('--class_dict_path', required=True, type=str, help='path of .csv class color map dict file')
    parser.add_argument('--classes_path', required=False, type=str, default=None, help='path to selected class definitions, default=%(default)s')
    parser.add_argument('--output_path', required=True, type=str, help='output path of converted png label files')

    args = parser.parse_args()

    # get selected class info
    if args.classes_path:
        class_names = get_classes(args.classes_path)
        # convert to lower string
        class_names = [class_name.lower() for class_name in class_names]
        assert class_names[0] == 'void', 'class 0 should be void to cover unlabeled pixel.'
    else:
        class_names = None

    camvid_convert(args.label_path, args.class_dict_path, class_names, args.output_path)


if __name__ == '__main__':
    main()
