#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate mIOU and related metrics for semantic segmentation model on validation dataset
"""
import os, sys, argparse, time
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import copy
import itertools
from tqdm import tqdm
from collections import OrderedDict
import operator

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..'))
from common.utils import get_data_list, get_classes


def plot_confusion_matrix(cm, classes, mIOU, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm[np.isnan(cm)] = 0
    trained_classes = classes
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=11)
    tick_marks = np.arange(len(classes))
    plt.xticks(np.arange(len(trained_classes)), classes, rotation=90,fontsize=9)
    plt.yticks(tick_marks, classes,fontsize=9)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j],2), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=7)
    plt.tight_layout()
    plt.ylabel('True label',fontsize=9)
    plt.xlabel('Predicted label',fontsize=9)

    plt.title('Mean IOU: '+ str(np.round(mIOU*100, 2)))
    output_path = os.path.join('result','confusion_matrix.png')
    os.makedirs('result', exist_ok=True)
    plt.savefig(output_path)
    #plt.show()

    # close the plot
    plt.close()
    return


def adjust_axes(r, t, fig, axes):
    """
     Plot - adjust axes
    """
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    """
     Draw plot using Matplotlib
    """
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in (green=true predictions) & (red=false predictions)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            #   first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    else:
      plt.barh(range(n_classes), sorted_values, color=plot_color)
      """
       Write number on side of bar
      """
      fig = plt.gcf() # gcf - get current figure
      axes = plt.gca()
      r = fig.canvas.get_renderer()
      for i, val in enumerate(sorted_values):
          str_val = " " + str(val) # add a space before
          if val < 1.0:
              str_val = " {0:.2f}".format(val)
          t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
          # re-set axes to show number inside the figure
          if i == (len(sorted_values)-1): # largest bar
              adjust_axes(r, t, fig, axes)
    # set window title
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15    # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


def plot_mIOU_result(IOUs, mIOU, num_classes):
    '''
     Draw mIOU plot (Show IOU's of all classes in decreasing order)
    '''
    window_title = "mIOU"
    plot_title = "mIOU: {0:.3f}%".format(mIOU*100)
    x_label = "Intersection Over Union"
    output_path = os.path.join('result','mIOU.png')
    os.makedirs('result', exist_ok=True)
    draw_plot_func(IOUs, num_classes, window_title, plot_title, x_label, output_path, to_show=False, plot_color='royalblue', true_p_bar='')


def generate_matrix(gt_mask, pre_mask, num_classes):
    valid = (gt_mask >= 0) & (gt_mask < num_classes)
    label = num_classes * gt_mask[valid].astype('int') + pre_mask[valid]
    count = np.bincount(label, minlength=num_classes**2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    return confusion_matrix


def eval_mIOU(dataset, gt_label_path, pred_label_path, class_names, model_output_shape):
    num_classes = len(class_names)

    # confusion matrix for all classes
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=float)

    # get model prediction result
    pbar = tqdm(total=len(dataset), desc='Eval model')
    for n, image_id in enumerate(dataset):
        gt_label_filename = os.path.join(gt_label_path, image_id.strip()+'.png')
        pred_label_filename = os.path.join(pred_label_path, image_id.strip()+'.png')

        # load groundtruth label mask
        gt_mask = np.array(Image.open(gt_label_filename))
        # reset all the invalid label value as 255
        gt_mask[gt_mask>(num_classes-1)] = 255
        gt_mask = cv2.resize(gt_mask, model_output_shape[::-1], interpolation = cv2.INTER_NEAREST)

        # load model predict label mask
        pred_mask = np.array(Image.open(pred_label_filename))

        # check if mask shape matches
        assert gt_mask.shape == pred_mask.shape, 'prediction mask shape mismatch with GT.'

        # update confusion matrix
        pred_mask = pred_mask.astype('int')
        gt_mask = gt_mask.astype('int')
        confusion_matrix += generate_matrix(gt_mask, pred_mask, num_classes)

        # compare prediction result with label
        # to update confusion matrix
        #flat_pred = np.ravel(pred_mask).astype('int')
        #flat_label = np.ravel(gt_mask).astype('int')
        #for p, l in zip(flat_pred, flat_label):
            #if l == num_classes or l == 255:
                #continue
            #if l < num_classes and p < num_classes:
                #confusion_matrix[l, p] += 1
            #else:
                #print('Invalid entry encountered, skipping! Label: ', l,
                        #' Prediction: ', p)
        pbar.update(1)
    pbar.close()

    # calculate Pixel accuracy
    PixelAcc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()

    # calculate Class accuracy
    ClassAcc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    ClassAcc[np.isnan(ClassAcc)] = 0
    mClassAcc = np.nanmean(ClassAcc)

    # calculate mIoU
    I = np.diag(confusion_matrix)
    U = np.sum(confusion_matrix, axis=0) + np.sum(confusion_matrix, axis=1) - I
    IoU = I/U
    IoU[np.isnan(IoU)] = 0
    mIoU = np.nanmean(IoU)

    # calculate FW (Frequency Weighted) IoU
    Freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    Freq[np.isnan(Freq)] = 0
    FWIoU = (Freq[Freq > 0] * IoU[Freq > 0]).sum()

    # calculate Dice Coefficient
    DiceCoef = 2*I / (U+I)
    DiceCoef[np.isnan(DiceCoef)] = 0

    # collect IOU/ClassAcc/Dice/Freq for every class
    IOUs, CLASS_ACCs, DICEs, FREQs = {}, {}, {}, {}
    for i,(class_name, iou, class_acc, dice, freq) in enumerate(zip(class_names, IoU, ClassAcc, DiceCoef, Freq)):
        IOUs[class_name] = iou
        CLASS_ACCs[class_name] = class_acc
        DICEs[class_name] = dice
        FREQs[class_name] = freq

    #sort IoU result by value, in descending order
    IOUs = OrderedDict(sorted(IOUs.items(), key=operator.itemgetter(1), reverse=True))

    #calculate mIOU from final IOU dict
    mIoU = np.nanmean(list(IOUs.values()))

    #show result
    print('\nevaluation summary')
    for class_name, iou in IOUs.items():
        print('%s: IoU %.4f, Freq %.4f, ClassAcc %.4f, Dice %.4f' % (class_name, iou, FREQs[class_name], CLASS_ACCs[class_name], DICEs[class_name]))
    print('mIoU=%.3f' % (mIoU*100))
    print('FWIoU=%.3f' % (FWIoU*100))
    print('PixelAcc=%.3f' % (PixelAcc*100))
    print('mClassAcc=%.3f' % (mClassAcc*100))

    # Plot mIOU & confusion matrix
    plot_mIOU_result(IOUs, mIoU, num_classes)
    plot_confusion_matrix(confusion_matrix, class_names, mIoU, normalize=True)

    return mIoU



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='evaluate Semantic Segmentation model with test dataset')
    '''
    Command line options
    '''
    parser.add_argument(
        '--dataset_file', type=str, required=True,
        help='eval samples txt file')

    parser.add_argument(
        '--gt_label_path', type=str, required=True,
        help='path containing groundtruth label png file')

    parser.add_argument(
        '--pred_label_path', type=str, required=True,
        help='path containing model predict label png file')

    parser.add_argument(
        '--classes_path', type=str, required=False, default='configs/voc_classes.txt',
        help='path to class definitions, default=%(default)s')

    parser.add_argument(
        '--model_output_shape', type=str,
        help='model mask output size as <height>x<width>, default=%(default)s', default='512x512')

    args = parser.parse_args()

    # param parse
    height, width = args.model_output_shape.split('x')
    model_output_shape = (int(height), int(width))

    # get class names
    class_names = get_classes(args.classes_path)
    assert len(class_names) < 254, 'PNG image label only support less than 254 classes.'

    # get dataset list
    dataset = get_data_list(args.dataset_file)

    start = time.time()
    eval_mIOU(dataset, args.gt_label_path, args.pred_label_path, class_names, model_output_shape)
    end = time.time()
    print("Evaluation time cost: {:.6f}s".format(end - start))


if __name__ == '__main__':
    main()
