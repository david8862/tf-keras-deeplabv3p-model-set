#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Data process utility functions."""
import numpy as np
import random
import math
import cv2
from PIL import Image, ImageEnhance

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def random_horizontal_flip(image, label, prob=.5):
    """
    Random horizontal flip for image & label

    # Arguments
        image: origin image for horizontal flip
            numpy array containing image data
        label: origin label for horizontal flip
            numpy array containing segment label mask
        prob: probability for random flip,
            scalar to control the flip probability.

    # Returns
        image: adjusted numpy array image.
        label: adjusted numpy array label mask
    """
    flip = rand() < prob
    if flip:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)

    return image, label


def random_vertical_flip(image, label, prob=.5):
    """
    Random vertical flip for image & label

    # Arguments
        image: origin image for vertical flip
            numpy array containing image data
        label: origin label for vertical flip
            numpy array containing segment label mask
        prob: probability for random flip,
            scalar to control the flip probability.

    # Returns
        image: adjusted numpy array image.
        label: adjusted numpy array label mask
    """
    flip = rand() < prob
    if flip:
        image = cv2.flip(image, 0)
        label = cv2.flip(label, 0)

    return image, label


#def random_brightness(image, jitter=.3):
    #"""
    #Random adjust brightness for image

    ## Arguments
        #image: origin image for brightness change
            #numpy array containing image data
        #jitter: jitter range for random brightness,
            #scalar to control the random brightness level.

    ## Returns
        #new_image: adjusted numpy array image.
    #"""
    #factor = 1.0 + random.gauss(mu=0.0, sigma=jitter)
    #if random.randint(0,1) and abs(factor) > 0.1:
        #factor = 1.0/factor
    #table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    #new_image = cv2.LUT(image, table)

    #return new_image

def random_brightness(image, jitter=.5):
    """
    Random adjust brightness for image

    # Arguments
        image: origin image for brightness change
            numpy array containing image data
        jitter: jitter range for random brightness,
            scalar to control the random brightness level.

    # Returns
        image: adjusted numpy array image.
    """
    img = Image.fromarray(image)
    enh_bri = ImageEnhance.Brightness(img)
    brightness = rand(jitter, 1/jitter)
    new_img = enh_bri.enhance(brightness)
    image = np.asarray(new_img)

    return image


def random_blur(image, prob=.5, size=5):
    """
    Random add gaussian blur to image

    # Arguments
        image: origin image for blur
            numpy array containing image data
        prob: probability for blur,
            scalar to control the blur probability.
        size: kernel size for gaussian blur,
            scalar to control the filter size.

    # Returns
        image: adjusted numpy array image.
    """
    blur = rand() < prob
    if blur:
        image = cv2.GaussianBlur(image, (size, size), 0)

    return image


def random_histeq(image, size=8, prob=.2):
    """
    Random apply "Contrast Limited Adaptive Histogram Equalization"
    to image

    # Arguments
        image: origin image for histeq
            numpy array containing image data
        size: grid size for CLAHE,
            scalar to control the grid size.
        prob: probability for histeq,
            scalar to control the histeq probability.

    # Returns
        image: adjusted numpy array image.
    """
    histeq = rand() < prob
    if histeq:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(size, size))
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) # to BGR
    return image


def random_grayscale(image, prob=.2):
    """
    Random convert image to grayscale

    # Arguments
        image: origin image for grayscale convert
            numpy array containing image data
        prob: probability for grayscale convert,
            scalar to control the convert probability.

    # Returns
        image: adjusted numpy array image.
    """
    convert = rand() < prob
    if convert:
        #convert to grayscale first, and then
        #back to 3 channels fake BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def random_chroma(image, jitter=.5):
    """
    Random adjust chroma (color level) for image

    # Arguments
        image: origin image for chroma change
            numpy array containing image data
        jitter: jitter range for random chroma,
            scalar to control the random color level.

    # Returns
        image: adjusted numpy array image.
    """
    img = Image.fromarray(image)
    enh_col = ImageEnhance.Color(img)
    color = rand(jitter, 1/jitter)
    new_img = enh_col.enhance(color)
    image = np.asarray(new_img)

    return image


def random_contrast(image, jitter=.5):
    """
    Random adjust contrast for image

    # Arguments
        image: origin image for contrast change
            numpy array containing image data
        jitter: jitter range for random contrast,
            scalar to control the random contrast level.

    # Returns
        image: adjusted numpy array image.
    """
    img = Image.fromarray(image)
    enh_con = ImageEnhance.Contrast(img)
    contrast = rand(jitter, 1/jitter)
    new_img = enh_con.enhance(contrast)
    image = np.asarray(new_img)

    return image


def random_sharpness(image, jitter=.5):
    """
    Random adjust sharpness for image

    # Arguments
        image: origin image for sharpness change
            numpy array containing image data
        jitter: jitter range for random sharpness,
            scalar to control the random sharpness level.

    # Returns
        image: adjusted numpy array image.
    """
    img = Image.fromarray(image)
    enh_sha = ImageEnhance.Sharpness(img)
    sharpness = rand(jitter, 1/jitter)
    new_img = enh_sha.enhance(sharpness)
    image = np.asarray(new_img)

    return image


def random_zoom_rotate(image, label, rotate_range=30, zoom_range=0.2, prob=0.3):
    """
    Random do zoom & rotate for image & label

    # Arguments
        image: origin image for zoom & rotate
            numpy array containing image data
        label: origin label for zoom & rotate
            numpy array containing segment label mask
        prob: probability for random flip,
            scalar to control the flip probability.

    # Returns
        image: adjusted numpy array image.
        label: adjusted numpy array label mask
    """
    if rotate_range:
        angle = random.gauss(mu=0.0, sigma=rotate_range)
    else:
        angle = 0.0

    if zoom_range:
        scale = random.gauss(mu=1.0, sigma=zoom_range)
    else:
        scale = 1.0

    warpAffine = rand() < prob
    if warpAffine and (rotate_range or zoom_range):
        M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, scale)
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        label = cv2.warpAffine(label, M, (label.shape[1], label.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image, label


class Grid(object):
    def __init__(self, d1, d2, rotate=360, ratio=0.5, mode=1, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode=mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def __call__(self, img, label):
        h = img.shape[0]
        w = img.shape[1]

        if np.random.rand() > self.prob:
            return img, label

        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h*h + w*w)))

        d = np.random.randint(self.d1, self.d2)
        #d = self.d

        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d*self.ratio)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh//d+1):
                s = d*i + st_h
                t = s+self.l
                s = max(min(s, hh), 0)
                t = max(min(t, hh), 0)
                mask[s:t,:] *= 0
        for i in range(-1, hh//d+1):
                s = d*i + st_w
                t = s+self.l
                s = max(min(s, hh), 0)
                t = max(min(t, hh), 0)
                mask[:,s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (hh-w)//2:(hh-w)//2+w]

        if self.mode == 1:
            mask = 1-mask

        #mask = mask.expand_as(img)
        img = img * np.expand_dims(mask, -1)
        label = label * mask

        return img, label


def random_gridmask(image, label, prob=0.2):
    """
    Random do GridMask augment for image & label

    reference:
        https://arxiv.org/abs/2001.04086
        https://github.com/Jia-Research-Lab/GridMask/blob/master/imagenet_grid/utils/grid.py

    # Arguments
        image: origin image for GridMask
            numpy array containing image data
        label: origin label for zoom & rotate
            numpy array containing segment label mask
        prob: probability for GridMask,
            scalar to control the GridMask probability.

    # Returns
        image: adjusted numpy array image.
        label: adjusted numpy array label mask
    """
    grid = Grid(d1=image.shape[1]//7, d2=image.shape[1]//3, rotate=360, ratio=0.5, prob=prob)
    image, label = grid(image, label)

    return image, label


def random_crop(image, label, crop_shape, prob=.1):
    """
    Random crop a specific size area from image
    and label

    # Arguments
        image: origin image for vertical flip
            numpy array containing image data
        label: origin label for vertical flip
            numpy array containing segment label mask
        crop_shape: target crop shape,
            list or tuple in (width, height).
        prob: probability for crop,
            scalar to control the crop probability.

    # Returns
        image: croped numpy array image.
        label: croped numpy array label mask
    """
    # check if the image and label are same shape
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')

    crop = rand() < prob
    if crop:
        if (crop_shape[0] < image.shape[1]) and (crop_shape[1] < image.shape[0]):
            x = random.randrange(image.shape[1]-crop_shape[0])
            y = random.randrange(image.shape[0]-crop_shape[1])

            image = image[y:y+crop_shape[1], x:x+crop_shape[0], :]
            label = label[y:y+crop_shape[1], x:x+crop_shape[0]]
        else:
            image = cv2.resize(image, crop_shape)
            label = cv2.resize(label, crop_shape, interpolation = cv2.INTER_NEAREST)

    return image, label



def normalize_image(image):
    """
    normalize image array from 0 ~ 255
    to -1.0 ~ 1.0

    # Arguments
        image: origin input image
            numpy image array with dtype=float, 0.0 ~ 255.0

    # Returns
        image: numpy image array with dtype=float, -1.0 ~ 1.0
    """
    image = image.astype(np.float32) / 127.5 - 1

    return image


def denormalize_image(image):
    """
    Denormalize image array from -1.0 ~ 1.0
    to 0 ~ 255

    # Arguments
        image: normalized image array with dtype=float, -1.0 ~ 1.0

    # Returns
        image: numpy image array with dtype=uint8, 0 ~ 255
    """
    image = (image * 127.5 + 127.5).astype(np.uint8)

    return image


def preprocess_image(image, model_image_size):
    """
    Prepare model input image data with
    resize, normalize and dim expansion

    # Arguments
        image: origin input image
            PIL Image object containing image data
        model_image_size: model input image size
            tuple of format (height, width).

    # Returns
        image_data: numpy array of image data for model input.
    """
    resized_image = image.resize(model_image_size, Image.BICUBIC)
    image_data = np.asarray(resized_image).astype('float32')
    #image_data = normalize_image(image_data)
    image_data = np.expand_dims(image_data, 0)
    return image_data


def mask_resize(mask, target_size):
    """
    Resize predict segmentation mask array to target size
    with bilinear interpolation

    # Arguments
        mask: predict mask array to be resize
            uint8 numpy array with shape (height, width, 1)
        target_size: target image size,
            tuple of format (width, height).

    # Returns
        resize_mask: resized mask array.

    """
    dst_w, dst_h = target_size # dest width & height
    src_h, src_w = mask.shape[:2] # src width & height

    if src_h == dst_h and src_w == dst_w:
        return mask.copy()

    scale_x = float(src_w) / dst_w # resize scale for width
    scale_y = float(src_h) / dst_h # resize scale for height

    # create & go through the target image array
    resize_mask = np.zeros((dst_h, dst_w), dtype=np.uint8)
    for dst_y in range(dst_h):
        for dst_x in range(dst_w):
            # mapping dest point back to src point
            src_x = (dst_x + 0.5) * scale_x - 0.5
            src_y = (dst_y + 0.5) * scale_y - 0.5
            # calculate round point in src image
            src_x_0 = int(np.floor(src_x))
            src_y_0 = int(np.floor(src_y))
            src_x_1 = min(src_x_0 + 1, src_w - 1)
            src_y_1 = min(src_y_0 + 1, src_h - 1)

            # Bilinear interpolation
            value0 = (src_x_1 - src_x) * mask[src_y_0, src_x_0] + (src_x - src_x_0) * mask[src_y_0, src_x_1]
            value1 = (src_x_1 - src_x) * mask[src_y_1, src_x_0] + (src_x - src_x_0) * mask[src_y_1, src_x_1]
            resize_mask[dst_y, dst_x] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)

    return resize_mask


def mask_resize_fast(mask, target_size):
    """
    Use cv2 to do a quick resize on predict
    segmentation mask array to target size

    # Arguments
        mask: predict mask array to be resize
            uint8 numpy array with shape (height, width, 1)
        target_size: target image size,
            tuple of format (width, height).

    # Returns
        resize_mask: resized mask array.

    """
    mask = cv2.merge([mask, mask, mask]).astype('uint8')
    #resize_mask = cv2.resize(mask, target_size, cv2.INTER_AREA)
    resize_mask = cv2.resize(mask, target_size, cv2.INTER_NEAREST)
    (resize_mask, _, _) = cv2.split(np.array(resize_mask))

    return resize_mask

