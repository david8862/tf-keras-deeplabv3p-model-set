#!/bin/bash
#
# Cityscapes class label definition:
#
# https://blog.csdn.net/zz2230633069/article/details/84591532
# https://github.com/CoinCheung/BiSeNet/blob/master/lib/cityscapes_cv2.py
#
#
#                  name |  id | trainId |       category | categoryId | hasInstances | ignoreInEval |        color
# ---------------------------------------------------------------------------------------------------------------------
#             unlabeled |   0 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
#           ego vehicle |   1 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
#  rectification border |   2 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
#            out of roi |   3 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
#                static |   4 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
#               dynamic |   5 |     255 |           void |          0 |            0 |            1 |      (111, 74, 0)
#                ground |   6 |     255 |           void |          0 |            0 |            1 |       (81, 0, 81)
#                  road |   7 |       0 |           flat |          1 |            0 |            0 |    (128, 64, 128)
#              sidewalk |   8 |       1 |           flat |          1 |            0 |            0 |    (244, 35, 232)
#               parking |   9 |     255 |           flat |          1 |            0 |            1 |   (250, 170, 160)
#            rail track |  10 |     255 |           flat |          1 |            0 |            1 |   (230, 150, 140)
#              building |  11 |       2 |   construction |          2 |            0 |            0 |      (70, 70, 70)
#                  wall |  12 |       3 |   construction |          2 |            0 |            0 |   (102, 102, 156)
#                 fence |  13 |       4 |   construction |          2 |            0 |            0 |   (190, 153, 153)
#            guard rail |  14 |     255 |   construction |          2 |            0 |            1 |   (180, 165, 180)
#                bridge |  15 |     255 |   construction |          2 |            0 |            1 |   (150, 100, 100)
#                tunnel |  16 |     255 |   construction |          2 |            0 |            1 |    (150, 120, 90)
#                  pole |  17 |       5 |         object |          3 |            0 |            0 |   (153, 153, 153)
#             polegroup |  18 |     255 |         object |          3 |            0 |            1 |   (153, 153, 153)
#         traffic light |  19 |       6 |         object |          3 |            0 |            0 |    (250, 170, 30)
#          traffic sign |  20 |       7 |         object |          3 |            0 |            0 |     (220, 220, 0)
#            vegetation |  21 |       8 |         nature |          4 |            0 |            0 |    (107, 142, 35)
#               terrain |  22 |       9 |         nature |          4 |            0 |            0 |   (152, 251, 152)
#                   sky |  23 |      10 |            sky |          5 |            0 |            0 |    (70, 130, 180)
#                person |  24 |      11 |          human |          6 |            1 |            0 |     (220, 20, 60)
#                 rider |  25 |      12 |          human |          6 |            1 |            0 |       (255, 0, 0)
#                   car |  26 |      13 |        vehicle |          7 |            1 |            0 |       (0, 0, 142)
#                 truck |  27 |      14 |        vehicle |          7 |            1 |            0 |        (0, 0, 70)
#                   bus |  28 |      15 |        vehicle |          7 |            1 |            0 |      (0, 60, 100)
#               caravan |  29 |     255 |        vehicle |          7 |            1 |            1 |        (0, 0, 90)
#               trailer |  30 |     255 |        vehicle |          7 |            1 |            1 |       (0, 0, 110)
#                 train |  31 |      16 |        vehicle |          7 |            1 |            0 |      (0, 80, 100)
#            motorcycle |  32 |      17 |        vehicle |          7 |            1 |            0 |       (0, 0, 230)
#               bicycle |  33 |      18 |        vehicle |          7 |            1 |            0 |     (119, 11, 32)
#         license plate |  -1 |      -1 |        vehicle |          7 |            0 |            1 |       (0, 0, 142)
#
# Example usages:
# ID of label 'car': 26
# Category of label with ID '26': vehicle
# Name of label with trainID '0': road


# Cityscapes dataset, 12GB
#echo "Downloading ADE20K dataset..."
#wget -O Cityscapes.zip http://xxx.com/Cityscapes.zip

# extract data
echo "Extracting dataset..."
unzip -e Cityscapes.zip

# convert Cityscapes annotation
echo "Merge images & converting Cityscapes segmant annotation..."

# merge png input images
mkdir -p Cityscapes/images/train/
pushd Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/
find -name *.png | xargs -n1 -i mv {} ../../../images/train/
popd
pushd Cityscapes/images/train/
ls | cut -d . -f1 > ../../train.txt
popd

mkdir -p Cityscapes/images/val/
pushd Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/
find -name *.png | xargs -n1 -i mv {} ../../../images/val/
popd
pushd Cityscapes/images/val/
ls | cut -d . -f1 > ../../val.txt
popd

mkdir -p Cityscapes/images/test/
pushd Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test/
find -name *.png | xargs -n1 -i mv {} ../../../images/test/
popd
pushd Cityscapes/images/test/
ls | cut -d . -f1 > ../../test.txt
popd

mv Cityscapes/images/train/*.png Cityscapes/images/
mv Cityscapes/images/val/*.png Cityscapes/images/
mv Cityscapes/images/test/*.png Cityscapes/images/
rm -rf Cityscapes/images/train Cityscapes/images/val Cityscapes/images/test


# merge train & val png annotation images, test annotations are INVALID
# xxx_xxx_xxx_xxx_labelIds.png is for semantic segmentation label
mkdir -p Cityscapes/gray_labels/
pushd Cityscapes/gtFine_trainvaltest/gtFine/train/
find -name *_labelIds.png | xargs -n1 -i mv {} ../../../gray_labels/
popd
pushd Cityscapes/gtFine_trainvaltest/gtFine/val/
find -name *_labelIds.png | xargs -n1 -i mv {} ../../../gray_labels/
popd

# convert grayscale png annotation image to PascalVOC style color png label image
python ../ade20k/gray_label_convert.py --input_path=Cityscapes/gray_labels/ --output_path=Cityscapes/labels/


# clean up
rm -rf Cityscapes/gray_labels Cityscapes/leftImg8bit_trainvaltest Cityscapes/gtFine_trainvaltest
mv Cityscapes ../../../
echo "Done"

