#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import numpy as np
import json
import cv2, colorsys
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from matplotlib import pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


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


def draw_label(image, text, color, coords):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    image = cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)
    image = cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                fontScale=font_scale,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA)
    return image


def get_colors(number, bright=True):
    """
    Get random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv_tuples = [(x / number, 1., brightness)
                  for x in range(number)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to RGB image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


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


def coco_visualize(image_path, json_path, output_path, classes_path, customize_coco=False, show=False):
    # load json annotation file
    coco = COCO(json_path)

    # create output path
    os.makedirs(output_path, exist_ok=True)

    # parse coco class names
    coco_classes = get_coco_classes(json_path)

    if classes_path:
        # pick some classes to check
        classes = get_classes(classes_path)
        category_ids = coco.getCatIds(catNms=classes)

        # need to collect image_id for each class,
        # and then merge them
        image_ids = []
        for category_id in category_ids:
            image_ids += coco.getImgIds(catIds=[category_id])
        image_ids = sorted(list(set(image_ids)))
    else:
        category_ids = []
        image_ids = sorted(coco.getImgIds(catIds=category_ids))

    pbar = tqdm(total=len(image_ids), desc='Visual image')
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

        # load image
        image = cv2.imread(image_path + image_info['file_name'], cv2.IMREAD_COLOR)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

        # Generate random colors
        colors = get_colors(len(annotations))

        for j, annotation in enumerate(annotations):
            # get object class name from category_id
            category_id = annotation['category_id']
            category_id = category_id-1 if customize_coco else convert_coco_category(category_id)
            class_name = coco_classes[category_id]

            # get object segment binary mask array
            #rle = coco_mask.frPyObjects(annotation['segmentation'], height, width)
            #mask = coco_mask.decode(rle).squeeze()
            mask = coco.annToMask(annotation)

            # draw segment mask on image
            image = apply_mask(image, mask, colors[j])

            # draw bbox and class label on image
            bbox = annotation['bbox']
            x, y, w, h = bbox
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), colors[j], 2)
            image = draw_label(image, class_name, colors[j], (int(x), int(y)))

        # save and show image
        cv2.imwrite(os.path.join(output_path, image_info['file_name']), image)
        if show:
            cv2.imshow('Image', image)
            cv2.waitKey(0)

    pbar.close()


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Visualize bbox and instance segment label for MSCOCO dataset')
    parser.add_argument('--image_path', required=True, type=str, help='path of mscoco images')
    parser.add_argument('--json_path', required=True, type=str, help='path of mscoco json annotation file')
    parser.add_argument('--output_path', required=True, type=str, help='path for saving visualize images')
    parser.add_argument('--classes_path', required=False, type=str, default=None, help='path to selected class definitions')
    parser.add_argument('--customize_coco', default=False, action="store_true", help='It is a user customize coco dataset. Will not follow standard coco class label')
    parser.add_argument('--show', default=False, action="store_true", help='Dump out training model to inference model')

    args = parser.parse_args()

    coco_visualize(args.image_path, args.json_path, args.output_path, args.classes_path, args.customize_coco, args.show)


if __name__ == '__main__':
    main()
