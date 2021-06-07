#!/bin/bash

# original PASCAL VOC 2012, 2 GB
echo "Downloading PascalVOC 2012 dataset..."
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
mv VOCdevkit/VOC2012 VOC2012_orig && rm -r VOCdevkit


# augmented PASCAL VOC, 1.3 GB
echo "Downloading SBD dataset..."
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
tar xvf benchmark.tgz
mv benchmark_RELEASE VOC_aug


# convert SBD semantic & instance mat label file to PascalVOC png label file
echo "SBD label convert..."
python voc_aug_convert.py --mat_label_path=./VOC_aug/dataset/cls --png_label_path=./VOC_aug/dataset/cls_png --label_type=semantic
python voc_aug_convert.py --mat_label_path=./VOC_aug/dataset/inst --png_label_path=./VOC_aug/dataset/inst_png --label_type=instance

# merge dataset txt files
echo "merge dataset..."
python imageset_merge.py --voc_set_file=./VOC2012_orig/ImageSets/Segmentation/train.txt --sbd_set_file=./VOC_aug/dataset/train.txt --output_file=./train.txt
python imageset_merge.py --voc_set_file=./VOC2012_orig/ImageSets/Segmentation/val.txt --sbd_set_file=./VOC_aug/dataset/val.txt --output_file=./val.txt
python imageset_merge.py --voc_set_file=./train.txt --sbd_set_file=./val.txt --output_file=./trainval.txt

# merge semantic segment label files
cp -rf ./VOC2012_orig/SegmentationClass/* ./VOC_aug/dataset/cls_png/
cp -rf ./VOC_aug/dataset/cls_png/*  ./VOC2012_orig/SegmentationClass/

# merge instance segment label files
cp -rf ./VOC2012_orig/SegmentationObject/* ./VOC_aug/dataset/inst_png/
cp -rf ./VOC_aug/dataset/inst_png/*  ./VOC2012_orig/SegmentationObject/

# merge dataset files
cp -rf ./train.txt ./VOC2012_orig/ImageSets/Segmentation/train.txt
cp -rf ./val.txt ./VOC2012_orig/ImageSets/Segmentation/val.txt
cp -rf ./trainval.txt ./VOC2012_orig/ImageSets/Segmentation/trainval.txt

# create soft-link for train process access
cd VOC2012_orig && ln -s JPEGImages images && ln -s SegmentationClass labels && cd ..

rm -rf train.txt val.txt trainval.txt VOC_aug
mv VOC2012_orig ../../../VOC2012

echo "Done"
