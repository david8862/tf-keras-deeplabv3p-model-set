#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
#from collections import OrderedDict
from labelme.utils import lblsave as label_save

# Cityscapes class label definition:
#
# https://blog.csdn.net/zz2230633069/article/details/84591532
# https://github.com/CoinCheung/BiSeNet/blob/master/lib/cityscapes_cv2.py
#
#
#                  name |  id | trainId |       category | categoryId | hasInstances | ignoreInEval |        color
# ---------------------------------------------------------------------------------------------------------------------
#             unlabeled |   0 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
#           ego vehicle |   1 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
#  rectification border |   2 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
#            out of roi |   3 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
#                static |   4 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
#               dynamic |   5 |     255 |           void |          0 |            0 |            1 |      (111, 74, 0)
#                ground |   6 |     255 |           void |          0 |            0 |            1 |       (81, 0, 81)
#                  road |   7 |       0 |           flat |          1 |            0 |            0 |    (128, 64, 128)
#              sidewalk |   8 |       1 |           flat |          1 |            0 |            0 |    (244, 35, 232)
#               parking |   9 |     255 |           flat |          1 |            0 |            1 |   (250, 170, 160)
#            rail track |  10 |     255 |           flat |          1 |            0 |            1 |   (230, 150, 140)
#              building |  11 |       2 |   construction |          2 |            0 |            0 |      (70, 70, 70)
#                  wall |  12 |       3 |   construction |          2 |            0 |            0 |   (102, 102, 156)
#                 fence |  13 |       4 |   construction |          2 |            0 |            0 |   (190, 153, 153)
#            guard rail |  14 |     255 |   construction |          2 |            0 |            1 |   (180, 165, 180)
#                bridge |  15 |     255 |   construction |          2 |            0 |            1 |   (150, 100, 100)
#                tunnel |  16 |     255 |   construction |          2 |            0 |            1 |    (150, 120, 90)
#                  pole |  17 |       5 |         object |          3 |            0 |            0 |   (153, 153, 153)
#             polegroup |  18 |     255 |         object |          3 |            0 |            1 |   (153, 153, 153)
#         traffic light |  19 |       6 |         object |          3 |            0 |            0 |    (250, 170, 30)
#          traffic sign |  20 |       7 |         object |          3 |            0 |            0 |     (220, 220, 0)
#            vegetation |  21 |       8 |         nature |          4 |            0 |            0 |    (107, 142, 35)
#               terrain |  22 |       9 |         nature |          4 |            0 |            0 |   (152, 251, 152)
#                   sky |  23 |      10 |            sky |          5 |            0 |            0 |    (70, 130, 180)
#                person |  24 |      11 |          human |          6 |            1 |            0 |     (220, 20, 60)
#                 rider |  25 |      12 |          human |          6 |            1 |            0 |       (255, 0, 0)
#                   car |  26 |      13 |        vehicle |          7 |            1 |            0 |       (0, 0, 142)
#                 truck |  27 |      14 |        vehicle |          7 |            1 |            0 |        (0, 0, 70)
#                   bus |  28 |      15 |        vehicle |          7 |            1 |            0 |      (0, 60, 100)
#               caravan |  29 |     255 |        vehicle |          7 |            1 |            1 |        (0, 0, 90)
#               trailer |  30 |     255 |        vehicle |          7 |            1 |            1 |       (0, 0, 110)
#                 train |  31 |      16 |        vehicle |          7 |            1 |            0 |      (0, 80, 100)
#            motorcycle |  32 |      17 |        vehicle |          7 |            1 |            0 |       (0, 0, 230)
#               bicycle |  33 |      18 |        vehicle |          7 |            1 |            0 |     (119, 11, 32)
#         license plate |  -1 |      -1 |        vehicle |          7 |            0 |            1 |       (0, 0, 142)
#
# Example usages:
# ID of label 'car': 26
# Category of label with ID '26': vehicle
# Name of label with trainID '0': road


def cityscapes_train_label(label_array):
    train_labels = [255,  # unlabeled
                    255,  # ego vehicle
                    255,  # rectification border
                    255,  # out of roi
                    255,  # static
                    255,  # dynamic
                    255,  # ground
                      0,  # road
                      1,  # sidewalk
                    255,  # parking
                    255,  # rail track
                      2,  # building
                      3,  # wall
                      4,  # fence
                    255,  # guard rail
                    255,  # bridge
                    255,  # tunnel
                      5,  # pole
                    255,  # polegroup
                      6,  # traffic light
                      7,  # traffic sign
                      8,  # vegetation
                      9,  # terrain
                     10,  # sky
                     11,  # person
                     12,  # rider
                     13,  # car
                     14,  # truck
                     15,  # bus
                    255,  # caravan
                    255,  # trailer
                     16,  # train
                     17,  # motorcycle
                     18,  # bicycle
                   ]

    label = label_array.copy()
    for i, train_label in enumerate(train_labels):
        label[label_array == i] = train_label

    # use 254 as invalid label to avoid labelme PNG
    # save error
    label[label == 255] = 254

    return label


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
        # convert Cityscapes annotation label to train label
        label_array = cityscapes_train_label(label_array)

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

