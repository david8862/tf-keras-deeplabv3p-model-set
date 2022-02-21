#!/bin/bash
# CIHP & ATR dataset package for human parsing, 2.91GB, from
# https://keras.io/examples/vision/deeplabv3_plus/
echo "Downloading CIHP dataset..."
pip install gdown
gdown https://drive.google.com/uc?id=1B9A9UCJYMwTL4oBEo4RZfbMZMaZhKJaz

# extract data
echo "Extracting dataset..."
mkdir instance-level-human-parsing
unzip -eq instance-level-human-parsing.zip -d instance-level-human-parsing

# convert CIHP dataset
echo "Convert CIHP dataset..."
cd instance-level-human-parsing/instance-level_human_parsing/instance-level_human_parsing/
mkdir images labels

mv Training/Images/* images/
mv Validation/Images/* images/

# convert grayscale png annotation image to PascalVOC style color png label image
python ../../../../ade20k/gray_label_convert.py --input_path=Training/Category_ids/ --output_path=labels/
python ../../../../ade20k/gray_label_convert.py --input_path=Validation/Category_ids/ --output_path=labels/

mv Training/train_id.txt ./
mv Validation/val_id.txt ./
cd -

# convert ATR dataset
echo "Convert ATR dataset..."
python ../ade20k/gray_label_convert.py --input_path=instance-level-human-parsing/ICCV15_fashion_dataset\(ATR\)/humanparsing/SegmentationClassAug --output_path=instance-level-human-parsing/ICCV15_fashion_dataset\(ATR\)/humanparsing/labels/

cd instance-level-human-parsing/ICCV15_fashion_dataset\(ATR\)/humanparsing/
ln -s JPEGImages images
cd -

cd instance-level-human-parsing/ICCV15_fashion_dataset\(ATR\)/humanparsing/JPEGImages/
ls | cut -d . -f1 > ../data.txt
cd -

# clean up
rm -rf instance-level-human-parsing.zip
mv instance-level-human-parsing ../../../
echo "Done"

