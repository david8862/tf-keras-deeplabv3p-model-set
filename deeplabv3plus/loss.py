#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    num_classes = K.shape(y_pred)[-1]
    y_true = K.one_hot(tf.cast(y_true[..., 0], tf.int32), num_classes+1)[..., :-1]
    return K.categorical_crossentropy(y_true, y_pred)

def sparse_crossentropy(y_true, y_pred):
    num_classes = K.shape(y_pred)[-1]
    y_true = K.one_hot(tf.cast(y_true[..., 0], tf.int32), num_classes)
    return K.categorical_crossentropy(y_true, y_pred)


def softmax_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, from_logits=False):
    """
    Compute softmax focal loss.
    Reference Paper:
        "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_pixel, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_pixel, num_classes).
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.

    # Returns
        softmax_focal_loss: Softmax focal loss, tensor of shape (?, num_pixel).
    """
    if from_logits:
        y_pred = K.softmax(y_pred)

    # Clip the prediction value to prevent NaN's and Inf's
    #epsilon = K.epsilon()
    #y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    y_pred = K.maximum(K.minimum(y_pred, 1 - 1e-15), 1e-15)

    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)

    # Calculate Focal Loss
    softmax_focal_loss = K.mean(alpha * K.pow(1 - y_pred, gamma) * cross_entropy, axis=-1)
    return softmax_focal_loss


class WeightedSparseCategoricalCrossEntropy(object):
  def __init__(self, weights, from_logits=False):
    self.weights = np.array(weights).astype('float32')
    self.from_logits = from_logits
    self.__name__ = 'weighted_sparse_categorical_crossentropy'

  def __call__(self, y_true, y_pred):
    return self.weighted_sparse_categorical_crossentropy(y_true, y_pred)

  def weighted_sparse_categorical_crossentropy(self, y_true, y_pred):
    num_classes = len(self.weights)
    y_true = K.one_hot(tf.cast(y_true[..., 0], tf.int32), num_classes)
    if self.from_logits:
        y_pred = K.softmax(y_pred)

    log_pred = K.log(y_pred)
    unweighted_losses = -K.sum(y_true*log_pred, axis=-1)

    weights = K.sum(K.constant(self.weights) * y_true, axis=-1)
    weighted_losses = unweighted_losses * weights
    return weighted_losses

