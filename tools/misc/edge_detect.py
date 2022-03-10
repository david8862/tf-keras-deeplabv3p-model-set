#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, argparse
import cv2
import numpy as np


def canny_edge(image):
    # kernel size could be：1, 3, 5, 7, 9, but better not too large
    kernel_size = 3

    # edge value thresholds
    threshold1 = 10
    threshold2 = 50

    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    edge_image = cv2.Canny(blurred, threshold1, threshold2, apertureSize=kernel_size, L2gradient=False)

    return edge_image


def gaussian_edge(image):
    blurred = cv2.GaussianBlur(image, (11, 11), 0) # gaussian filter with h=w=11，std=0
    edge_image = image - blurred

    # binarization the filtered image
    ret, edge_image = cv2.threshold(edge_image, 127, 255, cv2.THRESH_BINARY)

    return edge_image


def sobel_edge(image):
    # kernel size could be：1, 3, 5, 7, 9, but better not too large
    kernel_size = 3

    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # sobel-x
    sobel_X = cv2.convertScaleAbs(sobelx)
    # sobel-y
    sobel_Y = cv2.convertScaleAbs(sobely)
    # sobel-xy
    edge_image = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)

    # binarization the filtered image
    ret, edge_image = cv2.threshold(edge_image, 100, 255, cv2.THRESH_BINARY)

    return edge_image


def laplacian_edge(image):
    # kernel size could be：1, 3, 5, 7, 9, but better not too large
    kernel_size = 3
    laplacian = cv2.Laplacian(image, cv2.CV_8U, ksize=kernel_size)
    edge_image = cv2.convertScaleAbs(laplacian)

    # binarization the filtered image
    ret, edge_image = cv2.threshold(edge_image, 80, 255, cv2.THRESH_BINARY)

    return edge_image


def scharr_edge(image):
    scharr_x = cv2.Scharr(image, cv2.CV_8U, 1, 0)
    scharr_y = cv2.Scharr(image, cv2.CV_8U, 0, 1)
    scharrX = cv2.convertScaleAbs(scharr_x)
    scharrY = cv2.convertScaleAbs(scharr_y)

    edge_image = cv2.addWeighted(scharrX, 0.5, scharrY, 0.5, 0)

    # binarization the filtered image
    ret, edge_image = cv2.threshold(edge_image, 127, 255, cv2.THRESH_BINARY)

    return edge_image


def roberts_edge(image):
    # Roberts kernel
    kernelx = np.array([[-1, 0],
                        [0, 1]], dtype=int)

    kernely = np.array([[0, -1],
                        [1, 0]], dtype=int)

    x = cv2.filter2D(image, cv2.CV_16S, kernelx)
    y = cv2.filter2D(image, cv2.CV_16S, kernely)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    edge_image = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # binarization the filtered image
    ret, edge_image = cv2.threshold(edge_image, 15, 255, cv2.THRESH_BINARY)

    return edge_image


def prewitt_edge(image):
    # Prewitt kernel
    kernelx = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]], dtype=int)

    kernely = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], dtype=int)

    x = cv2.filter2D(image, cv2.CV_16S, kernelx)
    y = cv2.filter2D(image, cv2.CV_16S, kernely)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    edge_image = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # binarization the filtered image
    ret, edge_image = cv2.threshold(edge_image, 50, 255, cv2.THRESH_BINARY)

    return edge_image


def main():
    parser = argparse.ArgumentParser(description='detect edge curve in images')
    parser.add_argument('--image_path', help='image file or directory to predict', type=str, required=True)
    parser.add_argument('--output_path', help='Output path for the converted image', type=str, required=False, default=None)
    parser.add_argument('--edge_type', type=str, required=False, default='canny', choices=['canny', 'gaussian', 'sobel', 'laplacian', 'scharr', 'roberts', 'prewitt'],
                        help = "edge filter type, default=%(default)s")

    args = parser.parse_args()

    # get image file list or single image
    if os.path.isdir(args.image_path):
        image_files = glob.glob(os.path.join(args.image_path, '*'))
        assert args.output_path, 'need to specify output path if you use image directory as input.'
    else:
        image_files = [args.image_path]

    for i, image_file in enumerate(image_files):
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

        if args.edge_type == 'canny':
            edge_image = canny_edge(img)
        elif args.edge_type == 'gaussian':
            edge_image = gaussian_edge(img)
        elif args.edge_type == 'sobel':
            edge_image = sobel_edge(img)
        elif args.edge_type == 'laplacian':
            edge_image = laplacian_edge(img)
        elif args.edge_type == 'scharr':
            edge_image = scharr_edge(img)
        elif args.edge_type == 'roberts':
            edge_image = roberts_edge(img)
        elif args.edge_type == 'prewitt':
            edge_image = prewitt_edge(img)
        else:
            raise ValueError('invalid edge type')

        # save or show result
        if args.output_path:
            os.makedirs(args.output_path, exist_ok=True)
            output_file = os.path.join(args.output_path, os.path.basename(image_file))
            print(output_file, '{}/{}'.format(i, len(image_files)))
            cv2.imwrite(output_file, edge_image)
        else:
           cv2.imshow("Edge image", edge_image)
           cv2.waitKey(0)


if __name__ == "__main__":
    main()

