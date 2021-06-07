#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import numpy as np
from tqdm import trange
from collections import OrderedDict
from labelme.utils import lblsave as label_save
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..'))
from common.utils import get_classes

#VOC_CATEGORY_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
    #1, 64, 20, 63, 7, 72]
coco_category_list = ['background',
                      'person',
                      'bicycle',
                      'car',
                      'motorbike',
                      'aeroplane',
                      'bus',
                      'train',
                      'truck',
                      'boat',
                      'traffic light',
                      'fire hydrant',
                      'street sign',
                      'stop sign',
                      'parking meter',
                      'bench',
                      'bird',
                      'cat',
                      'dog',
                      'horse',
                      'sheep',
                      'cow',
                      'elephant',
                      'bear',
                      'zebra',
                      'giraffe',
                      'hat',
                      'backpack',
                      'umbrella',
                      'shoe',
                      'eye glasses',
                      'handbag',
                      'tie',
                      'suitcase',
                      'frisbee',
                      'skis',
                      'snowboard',
                      'sports ball',
                      'kite',
                      'baseball bat',
                      'baseball glove',
                      'skateboard',
                      'surfboard',
                      'tennis racket',
                      'bottle',
                      'plate',
                      'wine glass',
                      'cup',
                      'fork',
                      'knife',
                      'spoon',
                      'bowl',
                      'banana',
                      'apple',
                      'sandwich',
                      'orange',
                      'broccoli',
                      'carrot',
                      'hot dog',
                      'pizza',
                      'donut',
                      'cake',
                      'chair',
                      'sofa',
                      'pottedplant',
                      'bed',
                      'mirror',
                      'diningtable',
                      'window',
                      'desk',
                      'toilet',
                      'door',
                      'tvmonitor',
                      'laptop',
                      'mouse',
                      'remote',
                      'keyboard',
                      'cell phone',
                      'microwave',
                      'oven',
                      'toaster',
                      'sink',
                      'refrigerator',
                      'blender',
                      'book',
                      'clock',
                      'vase',
                      'scissors',
                      'teddy bear',
                      'hair drier',
                      'toothbrush',
                      'hair brush',
                      'banner',
                      'blanket',
                      'branch',
                      'bridge',
                      'building-other',
                      'bush',
                      'cabinet',
                      'cage',
                      'cardboard',
                      'carpet',
                      'ceiling-other',
                      'ceiling-tile',
                      'cloth',
                      'clothes',
                      'clouds',
                      'counter',
                      'cupboard',
                      'curtain',
                      'desk-stuff',
                      'dirt',
                      'door-stuff',
                      'fence',
                      'floor-marble',
                      'floor-other',
                      'floor-stone',
                      'floor-tile',
                      'floor-wood',
                      'flower',
                      'fog',
                      'food-other',
                      'fruit',
                      'furniture-other',
                      'grass',
                      'gravel',
                      'ground-other',
                      'hill',
                      'house',
                      'leaves',
                      'light',
                      'mat',
                      'metal',
                      'mirror-stuff',
                      'moss',
                      'mountain',
                      'mud',
                      'napkin',
                      'net',
                      'paper',
                      'pavement',
                      'pillow',
                      'plant-other',
                      'plastic',
                      'platform',
                      'playingfield',
                      'railing',
                      'railroad',
                      'river',
                      'road',
                      'rock',
                      'roof',
                      'rug',
                      'salad',
                      'sand',
                      'sea',
                      'shelf',
                      'sky-other',
                      'skyscraper',
                      'snow',
                      'solid-other',
                      'stairs',
                      'stone',
                      'straw',
                      'structural-other',
                      'table',
                      'tent',
                      'textile-other',
                      'towel',
                      'tree',
                      'vegetable',
                      'wall-brick',
                      'wall-concrete',
                      'wall-other',
                      'wall-panel',
                      'wall-stone',
                      'wall-tile',
                      'wall-wood',
                      'water-other',
                      'waterdrops',
                      'window-blind',
                      'window-other',
                      'wood']

def get_label_array(annotations, coco, height, width, class_names):
    # alloc label array for whole image
    label_array = np.zeros((height, width), dtype=np.uint8)

    for annotation in annotations:
        # decode mask & category_id for each annotated instance
        #rle = coco_mask.frPyObjects(annotation['segmentation'], height, width)
        #mask = coco_mask.decode(rle)
        mask = coco.annToMask(annotation)
        category_id = annotation['category_id']

        # search category_id from COCO full list to get its name,
        # and match to target class list
        category_name = coco_category_list[category_id]
        if category_name in class_names:
            class_id = class_names.index(category_name)
        else:
            continue
        # paste instance mask to label array
        if len(mask.shape) < 3:
            label_array[:, :] += (label_array == 0) * (mask * class_id)
        else:
            label_array[:, :] += (label_array == 0) * (((np.sum(mask, axis=2)) > 0) * class_id).astype(np.uint8)
    return label_array


def coco_label_convert(annotation_path, datasets, class_names, output_path):
    if not os.path.isdir(annotation_path):
        raise ValueError('Input path does not exist!\n')

    png_label_path = os.path.join(output_path, 'labels')
    os.makedirs(png_label_path, exist_ok=True)

    for dataset in datasets:
        # load COCO annotation
        coco_annotation_file = os.path.join(annotation_path, 'instances_{}{}.json'.format(dataset, '2017'))
        coco = COCO(coco_annotation_file)

        # list of all annotated images
        image_ids = list(coco.imgs.keys())

        # count class item number
        class_count = OrderedDict([(item, 0) for item in class_names])
        # record picked image id
        qualify_image_ids = []

        tbar = trange(len(image_ids))
        for i in tbar:
            image_id = image_ids[i]
            # get all instances annotation & metadata of one image
            image_annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
            image_metadata = coco.loadImgs(image_id)[0]

            # form a segmentation label array with instances annotation info
            label_array = get_label_array(image_annotations, coco, image_metadata['height'],
                                      image_metadata['width'], class_names)

            # label array filter. only more than 1k valid label pixels
            # would be used
            if (label_array > 0).sum() > 1000:
                qualify_image_ids.append(image_id)

                # count object class for statistic
                label_list = list(np.unique(label_array))
                for label in label_list:
                    class_name = class_names[label]
                    class_count[class_name] = class_count[class_name] + 1

                # save numpy label array as png label image,
                # using labelme utils function
                png_file_name = '%012d.png'%(image_id)
                label_save(os.path.join(png_label_path, png_file_name), label_array)

            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(image_ids), len(qualify_image_ids)))

        qualify_image_ids.sort()
        # save image list to dataset txt file
        dataset_file = open(os.path.join(output_path,'%s.txt'%(dataset)), 'w')
        for image_id in qualify_image_ids:
                dataset_file.write('%012d'%(image_id))
                dataset_file.write('\n')
        dataset_file.close()

        # show item number statistic
        print('Image number for each class:')
        for (class_name, number) in class_count.items():
            if class_name == 'background':
                continue
            print('%s: %d' % (class_name, number))
        print('total number of qualified images: ', len(qualify_image_ids))



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='convert MSCOCO 2017 segment annotation to .png label images')
    parser.add_argument('--annotation_path', type=str, required=True, help='path to MSCOCO 2017 annotation file')
    parser.add_argument('--set', required=False, type=str, default='all', choices=['all', 'train', 'val'], help='convert dataset, default=%(default)s')
    parser.add_argument('--classes_path', type=str, required=False, default='../../../configs/coco_classes.txt', help='path to selected class definitions, default=%(default)s')
    parser.add_argument('--output_path', required=True, type=str, help='output path containing converted png label image and dataset txt')

    args = parser.parse_args()

    # prepare class name list, add background class
    class_names = get_classes(args.classes_path)
    assert len(class_names) < 254, 'PNG image label only support less than 254 classes.'
    class_names = ['background'] + class_names

    # dataset list
    if args.set == 'all':
        datasets = ['train', 'val']
    else:
        datasets = [args.set]

    coco_label_convert(args.annotation_path, datasets, class_names, args.output_path)



if __name__ == '__main__':
    main()

