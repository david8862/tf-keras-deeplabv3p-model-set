#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels


# Fully connected CRF post processing function
def crf_postprocess(im, mask, zero_unsure=True):
    colors, labels = np.unique(mask, return_inverse=True)
    image_size = mask.shape[:2]
    n_labels = len(set(labels.flat))
    d = dcrf.DenseCRF2D(image_size[1], image_size[0], n_labels)  # width, height, nlabels
    U = unary_from_labels(labels, n_labels, gt_prob=.7, zero_unsure=zero_unsure)
    d.setUnaryEnergy(U)
    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3,3), compat=3)
    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im.astype('uint8'), compat=10)
    Q = d.inference(5) # 5 - num of iterations
    MAP = np.argmax(Q, axis=0).reshape(image_size)
    unique_map = np.unique(MAP)
    result = np.copy(MAP)
    for u in unique_map: # get original labels back
        np.putmask(result, MAP == u, colors[u])
    return result
    # MAP = crf_postprocess(frame, labels.astype('int32'), zero_unsure=False)

