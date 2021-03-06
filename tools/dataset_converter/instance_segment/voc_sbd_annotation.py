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

PASCAL_VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
                      'bottle', 'bus', 'car', 'cat', 'chair',
                      'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                      'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

sets=['train', 'val']


# extract numpy array from mat for http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html
# 'GTcls' key is for class segmentation
# 'GTinst' key is for instance segmentation
def get_label_info(mat_file):

    mat = scipy.io.loadmat(mat_file, mat_dtype=True, squeeze_me=True, struct_as_record=False)

    # attribute of mat['GTinst']: Segmentation/Categories/Boundaries
    label_array = mat['GTinst'].Segmentation.astype(np.uint8)

    classes = mat['GTinst'].Categories
    classes = [classes] if isinstance(classes, float) else classes
    classes = np.array(classes, dtype=np.uint8)

    return label_array, classes


def mask_to_bbox(mask):
    """
    Get bbox coordinate from instance segment mask

    # Arguments
        mask: binary mask array for one instance
              with shape (height, width)

    # Returns
        bbox coordinate (xmin, ymin, xmax, ymax)
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return int(xmin), int(ymin), int(xmax), int(ymax)


def voc_sbd_annotation(dataset_path, output_path):
    # get real path for dataset
    dataset_realpath = os.path.realpath(dataset_path)

    # create output path
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels'), exist_ok=True)

    for dataset in sets:
        image_ids = open(os.path.join(dataset_realpath, dataset+'.txt')).read().strip().split()
        list_file = open(os.path.join(output_path, dataset+'.txt'), 'w')

        # count class item number
        class_count = OrderedDict([(item, 0) for item in PASCAL_VOC_CLASSES])
        max_instance_number = 0

        pbar = tqdm(total=len(image_ids), desc='{} label converting'.format(dataset))
        for image_id in image_ids:
            pbar.update(1)
            list_file.write(image_id)

            # load instance segment mat file and convert to numpy label array
            mat_file = os.path.join(dataset_path, 'inst', image_id+'.mat')
            label_array, classes = get_label_info(mat_file)
            instance_number = len(classes)

            # collect max instance number per image
            if instance_number >= max_instance_number:
                max_instance_number = instance_number

            # save bbox & class info in txt
            for i in range(instance_number):
                # id in classes starts from 1 (0 for background)
                class_id = classes[i] - 1
                # pick instance binary mask from instance mask
                mask = (label_array == (i + 1)).astype(np.uint8)
                bbox = mask_to_bbox(mask)

                # count instance class for statistic
                class_name = PASCAL_VOC_CLASSES[class_id]
                class_count[class_name] = class_count[class_name] + 1

                list_file.write(" " + ",".join([str(item) for item in bbox]) + ',' + str(class_id))
            list_file.write('\n')

            # save numpy label array as png label image,
            # using labelme utils function
            png_file_name = image_id + '.png'
            label_save(os.path.join(output_path, 'labels', png_file_name), label_array)

        pbar.close()
        list_file.close()
        # show instance number statistic
        print('Total number of converted images: ', len(image_ids))
        print('Instance number for each class:')
        for (class_name, number) in class_count.items():
            print('%s: %d' % (class_name, number))
        print('Max instance number in one image: ', max_instance_number)


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='convert PascalVOC SBD instance segment annotation to png & txt annotation')
    parser.add_argument('--dataset_path', required=True, type=str, help='path of PascalVOC SBD dataset')
    parser.add_argument('--output_path', required=True, type=str, help='output path for converted png & txt annotation')

    args = parser.parse_args()

    voc_sbd_annotation(args.dataset_path, args.output_path)



if __name__ == '__main__':
    main()

