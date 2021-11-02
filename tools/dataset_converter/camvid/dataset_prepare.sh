#!/bin/bash

# CamVid dataset for semantic segmentation, (data image: 557MB, color label image: 16MB, sample number: 701)
echo "Downloading CamVid dataset..."
mkdir CamVid
wget -O CamVid/701_StillsRaw_full.zip http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip
wget -O CamVid/LabeledApproved_full.zip http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/data/LabeledApproved_full.zip

# extract data
echo "Extracting dataset..."
pushd CamVid/
unzip -eq 701_StillsRaw_full.zip
unzip -eq LabeledApproved_full.zip -d origin_labels
mv 701_StillsRaw_full images
popd

# convert png input images to jpg
echo "convert png input images to jpg..."
pushd CamVid/images
ls -1 *.png | xargs -n 1 bash -c 'convert "$0" "${0%.png}.jpg"'
rm -rf *.png
popd

# create data list file
# CamVid doesn't have default train/val/test separation, so we just have single list
echo "Create data list file..."
pushd CamVid/images/
ls | cut -d . -f1 > ../data.txt
popd

# rename label images to get ride of "_L" suffix
echo "Rename labels..."
pushd CamVid/origin_labels/
ls *.png |awk -F "_L" '{print "mv "$0" "$1$2""}' | bash
popd

# convert origin color label images to PascalVOC style .png label, and filter with
# selected class label
python camvid_convert.py --label_path=CamVid/origin_labels/ --class_dict_path=./class_dict.csv --output_path=CamVid/labels/ --classes_path=../../../configs/camvid_classes.txt

# clean up
rm -rf CamVid/origin_labels
mv CamVid ../../../
echo "Done"

