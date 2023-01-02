#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import numpy as np
import cv2
from PIL import Image

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
    dataset_list = list(zip(image_path_list, label_path_list))
    print('number of samples:', len(dataset_list))

    i=0
    while i < len(dataset_list):
        image_path, label_path = dataset_list[i]

        # Load image and label array
        img = Image.open(image_path).convert('RGB')
        lbl = Image.open(label_path)
        image = np.array(img)
        label = np.array(lbl)
        img.close()
        lbl.close()

        # reset all the invalid label value as 255
        #label[label>(num_classes-1)] = 255

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


        try:
            cv2.namedWindow("Dataset visualize f: forward; b: back; q: quit", 0)
            cv2.imshow("Dataset visualize f: forward; b: back; q: quit", image)
        except Exception as e:
            #print(repr(e))
            print('invalid image', image_path)
            try:
                cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE)
            except Exception as e:
                print('No valid window yet, try next image')
                i = i + 1

        keycode = cv2.waitKey(0) & 0xFF
        if keycode == ord('f'):
            #print('forward to next image')
            if i < len(dataset_list) - 1:
                i = i + 1
        elif keycode == ord('b'):
            #print('back to previous image')
            if i > 0:
                i = i - 1
        elif keycode == ord('q') or keycode == 27: # 27 is keycode for Esc
            print('exit')
            exit()
        else:
            print('unsupport key')



def main():
    parser = argparse.ArgumentParser(description='visualize dataset')

    parser.add_argument('--dataset_path', type=str, required=True, help='dataset path containing images and label png file')
    parser.add_argument('--dataset_file', type=str, required=True, help='data samples txt file')
    parser.add_argument('--classes_path', type=str, required=True, help='path to class definitions')

    args = parser.parse_args()

    dataset_visualize(args.dataset_path, args.dataset_file, args.classes_path)


if __name__ == '__main__':
    main()
