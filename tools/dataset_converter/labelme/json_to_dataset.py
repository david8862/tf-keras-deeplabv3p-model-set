#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert labelme json label file to voc png label image,
which inherited from demo script in labelme package
"""
import os, sys, argparse
import glob
import json
import base64
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from labelme import utils

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..'))
from common.utils import get_classes


def label_convert(json_file_path, png_label_path, class_names, polygon_only):
    if not os.path.isdir(json_file_path):
        raise ValueError('Input path does not exist!\n')
    os.makedirs(png_label_path, exist_ok=True)

    # all the json annotation file list
    json_files = glob.glob(os.path.join(json_file_path, '*.json'))

    # form a dict of class_name to label value
    label_name_to_value = {}
    for i, class_name in enumerate(class_names):
        label_name_to_value[class_name] = i

    # count class item number
    class_count = OrderedDict([(item, 0) for item in class_names])

    pbar = tqdm(total=len(json_files), desc='Label converting')
    for i, json_file in enumerate(json_files):
        data = json.load(open(json_file))

        # get image info
        #imageData = data.get("imageData")
        #if not imageData:
            #imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"].replace('\\', '/'))
            #with open(imagePath, "rb") as f:
                #imageData = f.read()
                #imageData = base64.b64encode(imageData).decode("utf-8")
        #img = utils.img_b64_to_arr(imageData)
        #img_shape = img.shape

        img_shape = (data["imageHeight"], data["imageWidth"], 3)

        if polygon_only:
            shapes = [shape for shape in data["shapes"] if shape["shape_type"] == "polygon"]
        else:
            shapes = data["shapes"]

        # warning if no valid shapes
        if len(shapes) == 0:
            print("Warning! No valid shapes for", json_file)

        # convert json labels to numpy label array
        # and save to png
        label_array, _ = utils.shapes_to_label(
            img_shape, shapes, label_name_to_value
        )

        # count object class for statistic
        label_list = list(np.unique(label_array))
        for label in label_list:
            class_name = class_names[label]
            class_count[class_name] = class_count[class_name] + 1

        utils.lblsave(os.path.join(png_label_path, os.path.splitext(os.path.basename(json_file))[0]+".png"), label_array)
        pbar.update(1)

    pbar.close()
    # show item number statistic
    print('Image number for each class:')
    for (class_name, number) in class_count.items():
        if class_name == 'background':
            continue
        print('%s: %d' % (class_name, number))
    print('total number of converted images: ', len(json_files))


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='convert labelme json label to voc png label')

    parser.add_argument('--json_file_path', required=True, type=str, help='path to labelme annotated json label files')
    parser.add_argument('--classes_path', type=str, required=False, default='../../../configs/voc_classes.txt', help='path to class definitions, default=%(default)s')
    parser.add_argument('--png_label_path', required=True, type=str, help='output path of converted png label images')
    parser.add_argument('--polygon_only', help="only convert polygon annotations", default=False, action="store_true")

    args = parser.parse_args()

    class_names = get_classes(args.classes_path)
    assert len(class_names) < 254, 'PNG image label only support less than 254 classes.'

    label_convert(args.json_file_path, args.png_label_path, class_names, args.polygon_only)


if __name__ == "__main__":
    main()
