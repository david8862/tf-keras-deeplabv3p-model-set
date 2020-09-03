#!/bin/bash

# MSCOCO 2017 train set, 19GB
echo "Downloading MSCOCO 2017 train set..."
wget http://images.cocodataset.org/zips/train2017.zip

# MSCOCO 2017 val set, 778MB
echo "Downloading MSCOCO 2017 val set..."
wget http://images.cocodataset.org/zips/val2017.zip

# MSCOCO 2017 train&val annotation, 242MB
echo "Downloading MSCOCO 2017 annotation..."
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# extract data
echo "Extracting dataset & annotation..."
unzip -e train2017.zip
unzip -e val2017.zip
unzip -e annotations_trainval2017.zip

# convert COCO annotation
echo "Converting MSCOCO 2017 segmant annotation..."
python coco_convert.py --dataset_path=./annotations/ --set=all --classes_path=../../../configs/coco_classes.txt --output_path=mscoco2017/

# move image and convert COCO annotation
echo "Final step..."
mv train2017/ mscoco2017/images/
mv val2017/* mscoco2017/images/
# merge a trainval dataset
cp mscoco2017/train.txt mscoco2017/trainval.txt && cat mscoco2017/val.txt >> mscoco2017/trainval.txt

# clean up
rm -rf train2017 val2017 annotations train2017.zip val2017.zip annotations_trainval2017.zip
mv mscoco2017 ../../../
echo "Done"

