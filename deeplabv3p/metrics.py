#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np

from tensorflow.keras import backend as K
import tensorflow as tf


def mIOU(gt, preds):
    ulabels = np.unique(gt)
    iou = np.zeros(len(ulabels))
    for k, u in enumerate(ulabels):
        inter = (gt == u) & (preds==u)
        union = (gt == u) | (preds==u)
        iou[k] = inter.sum()/union.sum()
    return np.round(iou.mean(), 2)


def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = y_pred.shape.as_list()[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = tf.cast(K.flatten(y_true), tf.int64)
    legal_labels = ~K.equal(y_true, nb_classes)
    return K.sum(tf.cast(legal_labels & K.equal(y_true,
                                                    K.argmax(y_pred, axis=-1)), tf.float32)) / K.sum(tf.cast(legal_labels, tf.float32))



def Jaccard(y_true, y_pred):
    nb_classes = y_pred.shape.as_list()[-1]
    iou = []
    pred_pixels = K.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes+1):
        true_labels = K.equal(y_true[:,:,0], i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.cast(true_labels & pred_labels, tf.int32)
        union = tf.cast(true_labels | pred_labels, tf.int32)
        legal_batches = K.sum(tf.cast(true_labels, tf.int32), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(ious[legal_batches]))
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = iou[legal_labels]
    return K.mean(iou)

