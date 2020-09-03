#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..'))
from common.utils import get_data_list


def merge_set(voc_set_file, sbd_set_file, output_file):
    voc_set_list = get_data_list(voc_set_file, shuffle=False)
    sbd_set_list = get_data_list(sbd_set_file, shuffle=False)

    # use set() struct to clean duplicate items
    output_list = list(set(voc_set_list + sbd_set_list))
    output_list.sort()

    # save merged list
    output_file = open(output_file, 'w')
    for image_id in output_list:
            output_file.write(image_id)
            output_file.write('\n')
    output_file.close()



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Merge PascalVOC & SBD (VOC Augmented) ImageSets')

    parser.add_argument('--voc_set_file', required=True, type=str, help='path to PascalVOC imageset txt file')
    parser.add_argument('--sbd_set_file', required=True, type=str, help='path to SBD imageset txt file')
    parser.add_argument('--output_file', required=True, type=str, help='output imageset txt file')

    args = parser.parse_args()

    merge_set(args.voc_set_file, args.sbd_set_file, args.output_file)


if __name__ == '__main__':
    main()
