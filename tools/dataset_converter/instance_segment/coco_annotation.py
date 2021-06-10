#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import numpy as np
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from collections import OrderedDict
from labelme.utils import lblsave as label_save


def get_classes(classes_path):
    '''load the classes'''
    with open(classes_path) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    return classes


def get_coco_classes(json_path):
    '''load coco classes'''
    with open(json_path) as f:
        data = json.load(f)

    classes = []
    categories = data['categories']
    current_id = -1
    for category in categories:
        # category format:
        # {
        #  "supercategory": string,
        #  "id": int,
        #  "name": string,
        # }
        if category['id'] <= current_id:
            # category list should be in ascending order
            raise ValueError('categories did not follow ascending order')
        current_id = category['id']
        classes.append(category['name'])

    return classes


def convert_coco_category(category_id):
    # since original 80 COCO category_ids is discontinuous,
    # we need to align them to continuous id (0~79) for further process
    if category_id >= 1 and category_id <= 11:
        category_id = category_id - 1
    elif category_id >= 13 and category_id <= 25:
        category_id = category_id - 2
    elif category_id >= 27 and category_id <= 28:
        category_id = category_id - 3
    elif category_id >= 31 and category_id <= 44:
        category_id = category_id - 5
    elif category_id >= 46 and category_id <= 65:
        category_id = category_id - 6
    elif category_id == 67:
        category_id = category_id - 7
    elif category_id == 70:
        category_id = category_id - 9
    elif category_id >= 72 and category_id <= 82:
        category_id = category_id - 10
    elif category_id >= 84 and category_id <= 90:
        category_id = category_id - 11

    return category_id


def coco_annotation(json_path, output_path, classes_path, customize_coco):
    # load json annotation file
    coco = COCO(json_path)

    # create output path
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels'), exist_ok=True)

    # txt use same basename as json
    dataset = os.path.basename(json_path).split('.')[0]
    list_file = open(os.path.join(output_path, dataset+'.txt'), 'w')

    # parse coco class names
    coco_class_names = get_coco_classes(json_path)

    if classes_path:
        # pick some classes to check
        class_names = get_classes(classes_path)
        category_ids = coco.getCatIds(catNms=class_names)

        # need to collect image_id for each class,
        # and then merge them
        image_ids = []
        for category_id in category_ids:
            image_ids += coco.getImgIds(catIds=[category_id])
        image_ids = sorted(list(set(image_ids)))
    else:
        class_names = coco_class_names
        category_ids = []
        image_ids = sorted(coco.getImgIds(catIds=category_ids))

    # count class item number
    class_count = OrderedDict([(item, 0) for item in class_names])
    max_instance_number = 0

    # all the image ids in COCO
    #category_ids = []
    #image_ids = sorted(coco.getImgIds(catIds=category_ids))

    pbar = tqdm(total=len(image_ids), desc='{} label converting'.format(dataset))
    for i in range(len(image_ids)):
        pbar.update(1)
        # image_info format:
        # {
        #  "license": int,
        #  "file_name": string,
        #  "coco_url": string,
        #  "height": int,
        #  "width": int,
        #  "date_captured": string,
        #  "flickr_url": string,
        #  "id": int,
        # }
        image_info = coco.loadImgs(image_ids[i])[0]
        height, width = image_info['height'], image_info['width']

        # alloc instance label array for whole image
        label_array = np.zeros((height, width), dtype=np.uint8)

        # get image basename from image_info, as id in txt file
        image_basename = image_info['file_name'].split('.')[0]
        list_file.write(image_basename)

        # get all annotations for the specified image,
        # annotations is a list of annotation dict with format:
        # {
        #  "id": int,
        #  "image_id": int,
        #  "category_id": int,
        #  "segmentation": RLE or [polygon],
        #  "area": float,
        #  "bbox": [x,y,width,height],
        #  "iscrowd": 0 or 1
        # }
        annotation_ids = coco.getAnnIds(imgIds=image_info['id'], catIds=category_ids, iscrowd=None)
        annotations = coco.loadAnns(annotation_ids)


        # collect max instance number per image
        instance_number = len(annotations)
        if instance_number >= max_instance_number:
            max_instance_number = instance_number

        for j, annotation in enumerate(annotations):
            # get instance class id from category_id
            category_id = annotation['category_id']
            category_id = category_id-1 if customize_coco else convert_coco_category(category_id)

            # convert category_id to selected class id
            coco_class_name = coco_class_names[category_id]
            class_id = class_names.index(coco_class_name)

            # count instance class for statistic
            class_name = class_names[class_id]
            class_count[class_name] = class_count[class_name] + 1

            # save bbox & class id in
            bbox = annotation['bbox']
            x, y, w, h = bbox
            # convert to (xmin, ymin, xmax, ymax)
            bbox = (int(x), int(y), int(x + w), int(y + h))
            list_file.write(" " + ",".join([str(item) for item in bbox]) + ',' + str(class_id))

            # get instance segment binary mask array
            #rle = coco_mask.frPyObjects(annotation['segmentation'], height, width)
            #mask = coco_mask.decode(rle).squeeze()
            mask = coco.annToMask(annotation)

            # paste instance mask to label array
            if len(mask.shape) < 3:
                label_array[:, :] += (label_array == 0) * (mask * (j + 1))
            else:
                label_array[:, :] += (label_array == 0) * (((np.sum(mask, axis=2)) > 0) * (j + 1)).astype(np.uint8)

        list_file.write('\n')
        # save label array as png label image,
        # using labelme utils function
        png_file_name = image_basename + '.png'
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
    parser.add_argument('--json_path', required=True, type=str, help='path of mscoco json annotation file')
    parser.add_argument('--output_path', required=True, type=str, help='output path for converted png & txt annotation')
    parser.add_argument('--classes_path', required=False, type=str, default=None, help='path to selected class definitions')
    parser.add_argument('--customize_coco', default=False, action="store_true", help='It is a user customize coco dataset. Will not follow standard coco class label')

    args = parser.parse_args()

    coco_annotation(args.json_path, args.output_path, args.classes_path, args.customize_coco)


if __name__ == '__main__':
    main()
