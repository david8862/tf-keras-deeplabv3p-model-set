#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_classes, get_data_list, visualize_segmentation


def dataset_visualize(dataset_path, dataset_file, classes_path):
    dataset_list = get_data_list(dataset_file, shuffle=False)

    class_names = get_classes(classes_path)
    assert len(class_names) < 254, 'PNG image label only support less than 254 classes.'
    num_classes = len(class_names)

    # get image & label file list
    dataset_realpath = os.path.realpath(dataset_path)
    image_path_list = [os.path.join(dataset_realpath, 'images', image_id.strip()+'.jpg') for image_id in dataset_list]
    label_path_list = [os.path.join(dataset_realpath, 'labels', image_id.strip()+'.png') for image_id in dataset_list]

    pbar = tqdm(total=len(image_path_list), desc='Visualize dataset')
    for i, (image_path, label_path) in enumerate(zip(image_path_list, label_path_list)):
        pbar.update(1)

        # Load image and label array
        img = Image.open(image_path).convert('RGB')
        lbl = Image.open(label_path)
        image = np.array(img)
        label = np.array(lbl)
        img.close()
        lbl.close()

        # reset all the invalid label value as 0(background)
        label[label>(num_classes-1)] = 0

        # render segmentation mask on image
        image = visualize_segmentation(image, label, class_names=class_names, overlay=0.5, ignore_count_threshold=1)

        # show image file info
        image_file_name = os.path.basename(image_path)
        cv2.putText(image, image_file_name+'({}/{})'.format(i+1, len(image_path_list)),
                    (3, 15),
                    cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(255, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA)

        # convert to BGR for display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.namedWindow("Image", 0)
        cv2.imshow("Image", image)
        keycode = cv2.waitKey(0) & 0xFF
        if keycode == ord('q') or keycode == 27: # 27 is keycode for Esc
            break
    pbar.close()



def main():
    parser = argparse.ArgumentParser(description='visualize dataset')

    parser.add_argument('--dataset_path', type=str, required=True, help='dataset path containing images and label png file')
    parser.add_argument('--dataset_file', type=str, required=True, help='data samples txt file')
    parser.add_argument('--classes_path', type=str, required=True, help='path to class definitions')

    args = parser.parse_args()

    dataset_visualize(args.dataset_path, args.dataset_file, args.classes_path)


if __name__ == '__main__':
    main()
