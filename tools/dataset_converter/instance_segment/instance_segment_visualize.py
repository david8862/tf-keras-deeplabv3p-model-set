#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm
import cv2, colorsys
import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


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


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_dataset(annotation_file, shuffle=True):
    with open(annotation_file) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    if shuffle:
        np.random.seed(int(time.time()))
        np.random.shuffle(lines)
        #np.random.seed(None)

    return lines


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to RGB image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def draw_label(image, text, color, coords):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)
    cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                fontScale=font_scale,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA)

    return image


def instance_segment_visualize(args):
    annotation_lines = get_dataset(args.dataset_file, shuffle=False)
    class_names = get_classes(args.classes_path)

    # create output path
    os.makedirs(args.output_path, exist_ok=True)

    pbar = tqdm(total=len(annotation_lines), desc='Visual image')
    for annotation_line in annotation_lines:
        pbar.update(1)

        # parse image_id and boxes from annotation_line
        line = annotation_line.split()
        image_id = line[0]
        boxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        # open data image & mask image from dataset
        image = cv2.imread(os.path.join(args.dataset_path, 'images', image_id + '.jpg'))
        mask = np.array(Image.open(os.path.join(args.dataset_path, 'labels', image_id + '.png')))

        # Generate random colors by object number
        colors = get_colors(len(boxes))

        for i, box in enumerate(boxes):
            class_name = class_names[box[-1]]

            box_xmin = int(box[0])
            box_ymin = int(box[1])
            box_xmax = int(box[2])
            box_ymax = int(box[3])

            cv2.rectangle(image, (box_xmin, box_ymin), (box_xmax, box_ymax), colors[i], 1, cv2.LINE_AA)
            image = draw_label(image, class_name, colors[i], (box_xmin, box_ymin))

            # pick instance binary segment mask from full mask,
            # here we assume the instance mask id should
            # matches the obj box id "i"
            instance_mask = (mask == (i + 1)).astype(np.uint8)

            height_index, width_index = np.where(mask == i+1)
            # bypass if no instance mask
            if len(height_index) == 0 or len(width_index) == 0:
                continue

            # check if the segment mask exceed bbox
            segment_xmin = np.min(width_index)
            segment_ymin = np.min(height_index)
            segment_xmax = np.max(width_index)
            segment_ymax = np.max(height_index)
            if (segment_xmin < box_xmin-1) or (segment_ymin < box_ymin-1) or (segment_xmax > box_xmax+1) or (segment_ymax > box_ymax+1):
                print('in id {}, instance segment outside the bbox'.format(image_id))

            # draw segment mask on image
            image = apply_mask(image, instance_mask, colors[i])

        # save and show image
        cv2.imwrite(os.path.join(args.output_path, image_id+'.jpg'), image)
        if args.show:
            cv2.imshow('Image', image)
            cv2.waitKey(0)

    pbar.close()


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Visualize bbox and instance segment mask for instance segment dataset')
    parser.add_argument('--dataset_path', required=True, type=str, help='path to instance segment dataset')
    parser.add_argument('--dataset_file', required=True, type=str, help='dataset txt file, including image id and bbox info')
    parser.add_argument('--classes_path', required=True, type=str, default=None, help='path to class definitions')
    parser.add_argument('--output_path', required=True, type=str, help='path for saving visualize images')
    parser.add_argument('--show', default=False, action="store_true", help='Dump out training model to inference model')

    args = parser.parse_args()

    instance_segment_visualize(args)


if __name__ == '__main__':
    main()
