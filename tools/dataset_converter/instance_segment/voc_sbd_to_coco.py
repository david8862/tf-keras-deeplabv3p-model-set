#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import numpy as np
import scipy.io
import json
import cv2
from tqdm import tqdm
from collections import OrderedDict
import pycocotools.mask

PASCAL_VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
                      'bottle', 'bus', 'car', 'cat', 'chair',
                      'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                      'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

#sets=['train', 'val']
sets=['val']

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


def voc_sbd_to_coco(dataset_path, output_path):
    # get real path for dataset
    dataset_realpath = os.path.realpath(dataset_path)

    # create output path
    os.makedirs(output_path, exist_ok=True)

    # initial image id & annotation id in COCO json
    coco_image_id = 1
    coco_annotation_id = 1

    for dataset in sets:
        image_ids = open('%s/%s.txt'%(dataset_realpath, dataset)).read().strip().split()
        annotations = []
        images = []

        # count class item number
        class_count = OrderedDict([(item, 0) for item in PASCAL_VOC_CLASSES])
        max_instance_number = 0

        pbar = tqdm(total=len(image_ids), desc='{} annotation converting'.format(dataset))
        for image_id in image_ids:
            pbar.update(1)

            # load image file to get height and width
            image = cv2.imread(os.path.join(dataset_path, 'img', image_id+'.jpg'))

            # load instance segment mat file and convert to numpy label array
            mat_file = os.path.join(dataset_path, 'inst', image_id+'.mat')
            label_array, classes = get_label_info(mat_file)
            instance_number = len(classes)

            # collect max instance number per image
            if instance_number >= max_instance_number:
                max_instance_number = instance_number

            for i in range(instance_number):
                # pick instance binary mask from instance mask
                mask = (label_array == (i + 1)).astype(np.uint8)
                bbox = mask_to_bbox(mask)
                category_id = int(classes[i])

                # encode mask array to COCO style RLE
                rle = pycocotools.mask.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle['counts'] = rle['counts'].decode('ascii')

                # append instance annotation
                annotations.append({
                    'id': coco_annotation_id,
                    'image_id': coco_image_id,
                    'category_id': category_id,
                    'segmentation': rle,
                    'area': float(mask.sum()),
                    'bbox': [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
                    'iscrowd': 0
                })

                coco_annotation_id += 1

                # count instance class for statistic
                class_name = PASCAL_VOC_CLASSES[category_id-1]
                class_count[class_name] = class_count[class_name] + 1

            # append image info
            images.append({
                'license': -1,
                'url': 'none',
                'date_captured': '2012/01/01',
                'id': coco_image_id,
                'width': int(image.shape[1]),
                'height': int(image.shape[0]),
                'file_name': image_id+'.jpg'
            })
            coco_image_id += 1
        pbar.close()

        categories = [{'supercategory': 'none', 'id': i+1, 'name': PASCAL_VOC_CLASSES[i]} for i in range(len(PASCAL_VOC_CLASSES))]

        # fixed info
        info = {
            'description': 'Pascal SBD',
            'url': 'http://home.bharathh.info/pubs/codes/SBD/download.html',
            'version': '1.0',
            'year': 2012,
            'contributor': 'UC Berkeley',
            'date_created': '2012/01/01'
        }

        # save annotation to json file
        output_file = os.path.join(output_path, 'instances_pascal_sbd_{}2012.json'.format(dataset))
        with open(output_file, 'w') as f:
            json.dump({
                'info': info,
                'licenses': {},
                'images': images,
                'type': 'instances',
                'annotations': annotations,
                'categories': categories
            }, f)

        # show instance number statistic
        print('Total number of converted images: ', len(image_ids))
        print('Instance number for each class:')
        for (class_name, number) in class_count.items():
            print('%s: %d' % (class_name, number))
        print('Max instance number in one image: ', max_instance_number)


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='convert PascalVOC SBD instance segment annotation to MSCOCO json annotation')
    parser.add_argument('--dataset_path', required=True, type=str, help='path of PascalVOC SBD dataset')
    parser.add_argument('--output_path', required=True, type=str, help='output path for converted json annotation')

    args = parser.parse_args()

    voc_sbd_to_coco(args.dataset_path, args.output_path)


if __name__ == '__main__':
    main()

