#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the deeplabv3p model for your own dataset.
"""
import os, sys, argparse, time
import warnings
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN

from deeplabv3p.model import get_deeplabv3p_model
from unet.model import get_unet_model
from fast_scnn.model import get_fast_scnn_model
from deeplabv3p.data import SegmentationGenerator
from deeplabv3p.loss import sparse_crossentropy, softmax_focal_loss, WeightedSparseCategoricalCrossEntropy
from deeplabv3p.metrics import Jaccard#, sparse_accuracy_ignoring_last_label
from common.utils import get_classes, get_data_list, optimize_tf_gpu, calculate_weigths_labels, load_class_weights
from common.model_utils import get_optimizer
from common.callbacks import EvalCallBack

# Try to enable Auto Mixed Precision on TF 2.0
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
optimize_tf_gpu(tf, K)


def main(args):
    log_dir = 'logs/000/'
    # get class info, add background class to match model & GT
    class_names = get_classes(args.classes_path)
    assert len(class_names) < 254, 'PNG image label only support less than 254 classes.'
    class_names = ['background'] + class_names
    num_classes = len(class_names)

    # callbacks for training process
    monitor = 'Jaccard'

    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-Jaccard{Jaccard:.3f}-val_loss{val_loss:.3f}-val_Jaccard{val_Jaccard:.3f}.h5'),
        monitor='val_{}'.format(monitor),
        mode='max',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_{}'.format(monitor), factor=0.5, mode='max',
                patience=5, verbose=1, cooldown=0, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_{}'.format(monitor), min_delta=0, patience=100, verbose=1, mode='max')
    terminate_on_nan = TerminateOnNaN()

    callbacks=[tensorboard, checkpoint, reduce_lr, early_stopping, terminate_on_nan]


    # get train&val dataset
    dataset = get_data_list(args.dataset_file)
    if args.val_dataset_file:
        val_dataset = get_data_list(args.val_dataset_file)
        num_train = len(dataset)
        num_val = len(val_dataset)
        dataset.extend(val_dataset)
    else:
        val_split = args.val_split
        num_val = int(len(dataset)*val_split)
        num_train = len(dataset) - num_val

    # prepare train&val data generator
    train_generator = SegmentationGenerator(args.dataset_path, dataset[:num_train],
                                            args.batch_size,
                                            num_classes,
                                            target_size=args.model_input_shape[::-1],
                                            weighted_type=args.weighted_type,
                                            is_eval=False,
                                            augment=True)

    valid_generator = SegmentationGenerator(args.dataset_path, dataset[num_train:],
                                            args.batch_size,
                                            num_classes,
                                            target_size=args.model_input_shape[::-1],
                                            weighted_type=args.weighted_type,
                                            is_eval=False,
                                            augment=False)

    # prepare online evaluation callback
    if args.eval_online:
        eval_callback = EvalCallBack(args.dataset_path, dataset[num_train:], class_names, args.model_input_shape, args.model_pruning, log_dir, eval_epoch_interval=args.eval_epoch_interval, save_eval_checkpoint=args.save_eval_checkpoint)
        callbacks.append(eval_callback)

    # prepare optimizer
    optimizer = get_optimizer(args.optimizer, args.learning_rate, average_type=None, decay_type=None)

    # prepare loss according to loss type & weigted type
    if args.weighted_type == 'balanced':
        classes_weights_path = os.path.join(args.dataset_path, 'classes_weights.txt')
        if os.path.isfile(classes_weights_path):
            weights = load_class_weights(classes_weights_path)
        else:
            weights = calculate_weigths_labels(train_generator, num_classes, save_path=args.dataset_path)
        losses = WeightedSparseCategoricalCrossEntropy(weights)
        sample_weight_mode = None
    elif args.weighted_type == 'adaptive':
        losses = sparse_crossentropy
        sample_weight_mode = 'temporal'
    elif args.weighted_type == None:
        losses = sparse_crossentropy
        sample_weight_mode = None
    else:
        raise ValueError('invalid weighted_type {}'.format(args.weighted_type))

    if args.loss == 'focal':
        warnings.warn("Focal loss doesn't support weighted class balance, will ignore related config")
        losses = softmax_focal_loss
        sample_weight_mode = None
    elif args.loss == 'crossentropy':
        # using crossentropy will keep the weigted type setting
        pass
    else:
        raise ValueError('invalid loss type {}'.format(args.loss))

    # prepare metric
    metrics = {'pred_mask' : Jaccard}

    # support multi-gpu training
    if args.gpu_num >= 2:
        # devices_list=["/gpu:0", "/gpu:1"]
        devices_list=["/gpu:{}".format(n) for n in range(args.gpu_num)]
        strategy = tf.distribute.MirroredStrategy(devices=devices_list)
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            # get multi-gpu train model
            if args.model_type.startswith('unet_'):
                model = get_unet_model(args.model_type, num_classes, args.model_input_shape, args.freeze_level, weights_path=args.weights_path)
            elif args.model_type.startswith('fast_scnn'):
                model = get_fast_scnn_model(args.model_type, num_classes, args.model_input_shape, weights_path=args.weights_path)
            else:
                model = get_deeplabv3p_model(args.model_type, num_classes, args.model_input_shape, args.output_stride, args.freeze_level, weights_path=args.weights_path)
            # compile model
            model.compile(optimizer=optimizer, sample_weight_mode=sample_weight_mode,
                          loss = losses, metrics = metrics)
    else:
        # get normal train model
        if args.model_type.startswith('unet_'):
            model = get_unet_model(args.model_type, num_classes, args.model_input_shape, args.freeze_level, weights_path=args.weights_path)
        elif args.model_type.startswith('fast_scnn'):
            model = get_fast_scnn_model(args.model_type, num_classes, args.model_input_shape, weights_path=args.weights_path)
        else:
            model = get_deeplabv3p_model(args.model_type, num_classes, args.model_input_shape, args.output_stride, args.freeze_level, weights_path=args.weights_path)
        # compile model
        model.compile(optimizer=optimizer, sample_weight_mode=sample_weight_mode,
                      loss = losses, metrics = metrics)
    model.summary()

    # Transfer training some epochs with frozen layers first if needed, to get a stable loss.
    initial_epoch = args.init_epoch
    epochs = initial_epoch + args.transfer_epoch
    print("Transfer training stage")
    print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(num_train, num_val, args.batch_size, args.model_input_shape))
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_generator),
                        validation_data=valid_generator,
                        validation_steps=len(valid_generator),
                        epochs=epochs,
                        initial_epoch=initial_epoch,
                        verbose=1,
                        workers=1,
                        use_multiprocessing=False,
                        max_queue_size=10,
                        callbacks = callbacks)

    # Wait 2 seconds for next stage
    time.sleep(2)

    if args.decay_type or args.average_type:
        # rebuild optimizer to apply learning rate decay or weights averager,
        # only after unfreeze all layers
        if args.decay_type:
            callbacks.remove(reduce_lr)
        steps_per_epoch = max(1, len(train_generator))
        decay_steps = steps_per_epoch * (args.total_epoch - args.init_epoch - args.transfer_epoch)
        optimizer = get_optimizer(args.optimizer, args.learning_rate, average_type=args.average_type, decay_type=args.decay_type, decay_steps=decay_steps)

    # Unfreeze the whole network for further tuning
    # NOTE: more GPU memory is required after unfreezing the body
    print("Unfreeze and continue training, to fine-tune.")
    if args.gpu_num >= 2:
        with strategy.scope():
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=optimizer, sample_weight_mode=sample_weight_mode,
                          loss = losses, metrics = metrics) # recompile to apply the change

    else:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=optimizer, sample_weight_mode=sample_weight_mode,
                      loss = losses, metrics = metrics) # recompile to apply the change

    print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(num_train, num_val, args.batch_size, args.model_input_shape))
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_generator),
                        validation_data=valid_generator,
                        validation_steps=len(valid_generator),
                        epochs=args.total_epoch,
                        initial_epoch=epochs,
                        verbose=1,
                        workers=1,
                        use_multiprocessing=False,
                        max_queue_size=10,
                        callbacks = callbacks)

    # Finally store model
    model.save(os.path.join(log_dir, 'trained_final.h5'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model definition options
    parser.add_argument('--model_type', type=str, required=False, default='mobilenetv2_lite',
        help='DeepLabv3+ model type: mobilenetv2/mobilenetv2_lite/resnet50, default=%(default)s')
    parser.add_argument('--weights_path', type=str, required=False, default=None,
        help = "Pretrained model/weights file for fine tune")
    parser.add_argument('--model_input_shape', type=str, required=False, default='512x512',
        help = "model image input shape as <height>x<width>, default=%(default)s")
    parser.add_argument('--output_stride', type=int, required=False, default=16, choices=[8, 16, 32],
        help = "model output stride, default=%(default)s")

    # Data options
    parser.add_argument('--dataset_path', type=str, required=False, default='VOC2012/',
        help='dataset path containing images and label png file, default=%(default)s')
    parser.add_argument('--dataset_file', type=str, required=False, default='VOC2012/ImageSets/Segmentation/trainval.txt',
        help='train samples txt file, default=%(default)s')
    parser.add_argument('--val_dataset_file', type=str, required=False, default=None,
        help='val samples txt file, default=%(default)s')
    parser.add_argument('--val_split', type=float, required=False, default=0.1,
        help = "validation data persentage in dataset if no val dataset provide, default=%(default)s")
    parser.add_argument('--classes_path', type=str, required=False, default='configs/voc_classes.txt',
        help='path to class definitions, default=%(default)s')

    # Training options
    parser.add_argument("--batch_size", type=int, required=False, default=16,
        help='batch size for training, default=%(default)s')
    parser.add_argument('--optimizer', type=str, required=False, default='sgd', choices=['adam', 'rmsprop', 'sgd'],
        help = "optimizer for training (adam/rmsprop/sgd), default=%(default)s")
    parser.add_argument('--loss', type=str, required=False, default='crossentropy', choices=['crossentropy', 'focal'],
        help = "loss type for training (crossentropy/focal), default=%(default)s")
    parser.add_argument('--weighted_type', type=str, required=False, default=None, choices=[None, 'adaptive', 'balanced'],
        help = "class balance weighted type, default=%(default)s")
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-2,
        help = "Initial learning rate, default=%(default)s")
    parser.add_argument('--average_type', type=str, required=False, default=None, choices=[None, 'ema', 'swa', 'lookahead'],
        help = "weights average type, default=%(default)s")
    parser.add_argument('--decay_type', type=str, required=False, default=None, choices=[None, 'cosine', 'exponential', 'polynomial', 'piecewise_constant'],
        help = "Learning rate decay type, default=%(default)s")
    parser.add_argument('--transfer_epoch', type=int, required=False, default=5,
        help = "Transfer training stage epochs, default=%(default)s")
    parser.add_argument('--freeze_level', type=int, required=False, default=1, choices=[0, 1, 2],
        help = "Freeze level of the model in transfer training stage. 0:NA/1:backbone/2:only open prediction layer")

    parser.add_argument("--init_epoch", type=int, required=False, default=0,
        help="initial training epochs for fine tune training, default=%(default)s")
    parser.add_argument("--total_epoch", type=int, required=False, default=150,
        help="total training epochs, default=%(default)s")
    parser.add_argument('--gpu_num', type=int, required=False, default=1,
        help='Number of GPU to use, default=%(default)s')
    parser.add_argument('--model_pruning', default=False, action="store_true",
        help='Use model pruning for optimization, only for TF 1.x')

    # Evaluation options
    parser.add_argument('--eval_online', default=False, action="store_true",
        help='Whether to do evaluation on validation dataset during training')
    parser.add_argument('--eval_epoch_interval', type=int, required=False, default=10,
        help = "Number of iteration(epochs) interval to do evaluation, default=%(default)s")
    parser.add_argument('--save_eval_checkpoint', default=False, action="store_true",
        help='Whether to save checkpoint with best evaluation result')

    args = parser.parse_args()
    height, width = args.model_input_shape.split('x')
    args.model_input_shape = (int(height), int(width))

    main(args)
