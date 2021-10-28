#!/bin/bash
#

# Cityscapes dataset, 12GB
#echo "Downloading ADE20K dataset..."
#wget -O Cityscapes.zip http://xxx.com/Cityscapes.zip

# extract data
echo "Extracting dataset..."
unzip -eq Cityscapes.zip

# convert Cityscapes annotation
echo "Merge images & converting Cityscapes segmant annotation..."

# merge png input images
mkdir -p Cityscapes/images/train/
pushd Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/
find -name *.png | xargs -n1 -i mv {} ../../../images/train/
popd
pushd Cityscapes/images/train/
# rename images to get ride of "_leftImg8bit" suffix
ls *.png |awk -F "_leftImg8bit" '{print "mv "$0" "$1$2""}' | bash
ls | cut -d . -f1 > ../../train.txt
popd

mkdir -p Cityscapes/images/val/
pushd Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/
find -name *.png | xargs -n1 -i mv {} ../../../images/val/
popd
pushd Cityscapes/images/val/
# rename images to get ride of "_leftImg8bit" suffix
ls *.png |awk -F "_leftImg8bit" '{print "mv "$0" "$1$2""}' | bash
ls | cut -d . -f1 > ../../val.txt
popd

#mkdir -p Cityscapes/images/test/
#pushd Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test/
#find -name *.png | xargs -n1 -i mv {} ../../../images/test/
#popd
#pushd Cityscapes/images/test/
# rename images to get ride of "_leftImg8bit" suffix
#ls *.png |awk -F "_leftImg8bit" '{print "mv "$0" "$1$2""}' | bash
#ls | cut -d . -f1 > ../../test.txt
#popd

# convert png input images to jpg, then merge together
pushd Cityscapes/images/train/
ls -1 *.png | xargs -n 1 bash -c 'convert "$0" "${0%.png}.jpg"'
popd
mv Cityscapes/images/train/*.jpg Cityscapes/images/

pushd Cityscapes/images/val/
ls -1 *.png | xargs -n 1 bash -c 'convert "$0" "${0%.png}.jpg"'
popd
mv Cityscapes/images/val/*.jpg Cityscapes/images/

#pushd Cityscapes/images/test/
#ls -1 *.png | xargs -n 1 bash -c 'convert "$0" "${0%.png}.jpg"'
#popd
#mv Cityscapes/images/test/*.jpg Cityscapes/images/

rm -rf Cityscapes/images/train Cityscapes/images/val Cityscapes/images/test


# merge train & val png annotation images, test annotations are INVALID
# xxx_xxx_xxx_gtFine_labelIds.png is for semantic segmentation label
mkdir -p Cityscapes/gray_labels/
pushd Cityscapes/gtFine_trainvaltest/gtFine/train/
find -name *_gtFine_labelIds.png | xargs -n1 -i mv {} ../../../gray_labels/
popd
pushd Cityscapes/gtFine_trainvaltest/gtFine/val/
find -name *_gtFine_labelIds.png | xargs -n1 -i mv {} ../../../gray_labels/
popd

# rename annotation images to get ride of "_gtFine_labelIds" suffix
pushd Cityscapes/gray_labels/
ls *.png |awk -F "_gtFine_labelIds" '{print "mv "$0" "$1$2""}' | bash
popd

# convert grayscale png annotation image to PascalVOC style color png label image
#python ../ade20k/gray_label_convert.py --input_path=Cityscapes/gray_labels/ --output_path=Cityscapes/labels/
python gray_label_convert.py --input_path=Cityscapes/gray_labels/ --output_path=Cityscapes/labels/


# clean up
rm -rf Cityscapes/gray_labels Cityscapes/leftImg8bit_trainvaltest Cityscapes/gtFine_trainvaltest
mv Cityscapes ../../../
echo "Done"

