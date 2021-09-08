#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test data argument process
"""
import os, sys, argparse
import cv2
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from deeplabv3p.data import SegmentationGenerator
from common.data_utils import denormalize_image
from common.utils import get_classes, get_data_list, visualize_segmentation


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Test tool for data augment process')
    parser.add_argument('--dataset_path', type=str, required=True, help='dataset path containing images and label png file')
    parser.add_argument('--dataset_file', type=str, required=True, help='data samples txt file')
    parser.add_argument('--classes_path', type=str, required=True, help='path to keypoint class definition file')
    parser.add_argument('--model_input_shape', type=str, required=False, help='model image input shapeas <height>x<width>, default=%(default)s', default='512x512')
    parser.add_argument('--batch_size', type=int, required=False, help = "batch size for test data, default=%(default)s", default=16)

    parser.add_argument('--output_path', type=str, required=False,  help='output path for augmented images, default=%(default)s', default='./test')
    parser.add_argument('--show_mask', default=False, action="store_true", help='show gt segment mask on augmented image')

    args = parser.parse_args()

    class_names = get_classes(args.classes_path)
    assert len(class_names) < 254, 'PNG image label only support less than 254 classes.'
    class_names = ['background'] + class_names
    num_classes = len(class_names)

    height, width = args.model_input_shape.split('x')
    model_input_shape = (int(height), int(width))

    os.makedirs(args.output_path, exist_ok=True)

    # get train&val dataset
    dataset = get_data_list(args.dataset_file)

    # prepare train dataset (having augmented data process)
    data_generator = SegmentationGenerator(args.dataset_path, dataset,
                                           batch_size=1,
                                           num_classes=num_classes,
                                           target_size=model_input_shape[::-1],
                                           weighted_type=None,
                                           is_eval=False,
                                           augment=True)

    pbar = tqdm(total=args.batch_size, desc='Generate augment image')
    for i, (image_data, gt_mask) in enumerate(data_generator):
        if i >= args.batch_size:
            break
        pbar.update(1)

        # get ground truth keypoints (transformed)
        image = image_data[0].astype('uint8')
        gt_mask = gt_mask[0, :, 0].reshape(model_input_shape).astype('uint8')

        # currently data generator didn't normalize input image,
        # so we bypass the denormalize step
        #image = denormalize_image(image)

        if args.show_mask:
            # render segmentation mask on image
            image = visualize_segmentation(image, gt_mask, class_names=class_names, overlay=0.5, ignore_count_threshold=1)

        # save rendered image
        image = Image.fromarray(image)
        # here we handle the RGBA image
        if(len(image.split()) == 4):
            r, g, b, a = image.split()
            image = Image.merge("RGB", (r, g, b))
        image.save(os.path.join(args.output_path, str(i)+".jpg"))
    pbar.close()
    print('Done. augment images have been saved in', args.output_path)


if __name__ == "__main__":
    main()

