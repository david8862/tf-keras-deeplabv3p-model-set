#!/bin/bash

if [[ "$#" -ne 5 ]]; then
    echo "Usage: $0 <model_file> <image_path> <dataset_file> <class_file> <output_path>"
    exit 1
fi

MODEL_FILE=$1
IMAGE_PATH=$2
DATASET_FILE=$3
CLASS_FILE=$4
OUTPUT_PATH=$5

IMAGE_LIST=$(cat $DATASET_FILE)
IMAGE_NUM=$(cat $DATASET_FILE | wc -l)

# prepare process bar
i=0
ICON_ARRAY=("\\" "|" "/" "-")

# create output path first
mkdir -p $OUTPUT_PATH

for IMAGE_ID in $IMAGE_LIST
do
    ./deeplabSegment -m $MODEL_FILE -i $IMAGE_PATH"/"$IMAGE_ID".jpg" -l $CLASS_FILE -k $OUTPUT_PATH"/"$IMAGE_ID".png" -t 4 -c 1 -w 1 -p 0 2>&1 >> /dev/null
    # update process bar
    let index=i%4
    let percent=i*100/IMAGE_NUM
    let num=percent/2
    bar=$(seq -s "#" $num | tr -d "[:digit:]")
+    #printf "inference process: %d/%d [%c]\r" "$i" "$IMAGE_NUM" "${ICON_ARRAY[$index]}"
+    printf "inference process: %d/%d [%-50s] %d%% \r" "$i" "$IMAGE_NUM" "$bar" "$percent"
    let i=i+1
done
printf "\nDone\n"
