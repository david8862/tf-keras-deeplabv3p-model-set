#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""custom model callbacks."""
import os, sys, random, tempfile
import numpy as np
import glob
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.callbacks import Callback

from eval import eval_mIOU


class CheckpointCleanCallBack(Callback):
    def __init__(self, checkpoint_dir, max_val_keep=5, max_eval_keep=2):
        self.checkpoint_dir = checkpoint_dir
        self.max_val_keep = max_val_keep
        self.max_eval_keep = max_eval_keep

    def on_epoch_end(self, epoch, logs=None):

        # filter out eval checkpoints and val checkpoints
        all_checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'ep*.h5')))
        eval_checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'ep*-mIOU*.h5')))
        val_checkpoints = sorted(list(set(all_checkpoints) - set(eval_checkpoints)))

        # keep latest val checkpoints
        for val_checkpoint in val_checkpoints[:-(self.max_val_keep)]:
            os.remove(val_checkpoint)

        # keep latest eval checkpoints
        for eval_checkpoint in eval_checkpoints[:-(self.max_eval_keep)]:
            os.remove(eval_checkpoint)


class EvalCallBack(Callback):
    def __init__(self, dataset_path, dataset, class_names, model_input_shape, model_pruning, log_dir, eval_epoch_interval=10, save_eval_checkpoint=False):
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.class_names = class_names
        self.model_input_shape = model_input_shape
        self.model_pruning = model_pruning
        self.log_dir = log_dir
        self.eval_epoch_interval = eval_epoch_interval
        self.save_eval_checkpoint = save_eval_checkpoint
        self.best_mIOU = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.eval_epoch_interval == 0:
            # Do eval every eval_epoch_interval epochs
            mIOU = eval_mIOU(self.model, 'H5', self.dataset_path, self.dataset, self.class_names, self.model_input_shape, do_crf=False, save_result=False, show_background=True)

            if self.save_eval_checkpoint and mIOU > self.best_mIOU:
                # Save best mIOU value and model checkpoint
                self.best_mIOU = mIOU
                self.model.save(os.path.join(self.log_dir, 'ep{epoch:03d}-loss{loss:.3f}-Jaccard{Jaccard:.3f}-val_loss{val_loss:.3f}-val_Jaccard{val_Jaccard:.3f}-mIOU{mIOU:.3f}.h5'.format(epoch=(epoch+1), loss=logs.get('loss'), Jaccard=logs.get('Jaccard'), val_loss=logs.get('val_loss'), val_Jaccard=logs.get('val_Jaccard'), mIOU=mIOU)))
