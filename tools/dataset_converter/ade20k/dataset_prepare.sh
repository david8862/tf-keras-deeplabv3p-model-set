#!/bin/bash

# ADE20K dataset for semantic segmentation, 923MB (train: 20210, val: 2000)
echo "Downloading ADE20K dataset..."
wget -O ADEChallengeData2016.zip http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

# extract data
echo "Extracting dataset..."
unzip -e ADEChallengeData2016.zip


# convert ADE20K annotation
echo "Merge images & converting ADE20K segmant annotation..."

# merge jpg input images
cd ADEChallengeData2016/images/training
ls | cut -d . -f1 > ../../train.txt
mv *.jpg ../
cd -

cd ADEChallengeData2016/images/validation
ls | cut -d . -f1 > ../../val.txt
mv *.jpg ../
cd -
rm -rf ADEChallengeData2016/images/training ADEChallengeData2016/images/validation

# merge png annotation images
cd ADEChallengeData2016/annotations/training
mv *.png ../
cd -

cd ADEChallengeData2016/annotations/validation
mv *.png ../
cd -
rm -rf ADEChallengeData2016/annotations/training ADEChallengeData2016/annotations/validation

# convert grayscale png annotation image to PascalVOC style color png label image
python gray_label_convert.py --input_path=ADEChallengeData2016/annotations/ --output_path=ADEChallengeData2016/labels/


# clean up
rm -rf ADEChallengeData2016/annotations ADEChallengeData2016.zip
mv ADEChallengeData2016 ../../../
echo "Done"

