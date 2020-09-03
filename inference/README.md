## C++ on-device (X86/ARM) inference app for DeepLab v3+ semantic segmentation modelset

Here are some C++ implementation of the on-device inference for trained DeepLab v3+ inference model, including forward propagation of the model, postprocess and predict segment mask output. Generally it should support all DeepLab v3+ archs and all kinds of backbones. Now we have 2 approaches with different inference engine for that:

* Tensorflow-Lite (verified on commit id: 1b8f5bc8011a1e85d7a110125c852a4f431d0f59)
* [MNN](https://github.com/alibaba/MNN) from Alibaba (verified on release: [1.0.0](https://github.com/alibaba/MNN/releases/tag/1.0.0))


### MNN

1. Install Python runtime and Build libMNN

Refer to [MNN build guide](https://www.yuque.com/mnn/cn/build_linux), we need to prepare cmake & protobuf first for MNN build. And since MNN support both X86 & ARM platform, we can do either native compile or ARM cross-compile
```
# apt install cmake autoconf automake libtool ocl-icd-opencl-dev
# wget https://github.com/google/protobuf/releases/download/v3.4.1/protobuf-cpp-3.4.1.tar.gz
# tar xzvf protobuf-cpp-3.4.1.tar.gz
# cd protobuf-3.4.1
# ./autogen.sh
# ./configure && make && make check && make install && ldconfig
# pip install --upgrade pip && pip install --upgrade mnn

# git clone https://github.com/alibaba/MNN.git <Path_to_MNN>
# cd <Path_to_MNN>
# ./schema/generate.sh
# ./tools/script/get_model.sh  # optional
# mkdir build && cd build
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>]
        [-DMNN_BUILD_QUANTOOLS=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TRAIN=ON -MNN_BUILD_TRAIN_MINI=ON -MNN_USE_OPENCV=OFF] ..
        && make -j4

### MNN OpenCL backend build
# apt install ocl-icd-opencl-dev
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>]
        [-DMNN_OPENCL=ON -DMNN_SEP_BUILD=OFF -DMNN_USE_SYSTEM_LIB=ON] ..
        && make -j4
```
If you want to do cross compile for ARM platform, "CMAKE_TOOLCHAIN_FILE" should be specified

"MNN_BUILD_QUANTOOLS" is for enabling MNN Quantization tool

"MNN_BUILD_CONVERTER" is for enabling MNN model converter

"MNN_BUILD_BENCHMARK" is for enabling on-device inference benchmark tool

"MNN_BUILD_TRAIN" related are for enabling MNN training tools


2. Build demo inference application
```
# cd tf-keras-deeplabv3p-model-set/inference/MNN
# mkdir build && cd build
# cmake -DMNN_ROOT_PATH=<Path_to_MNN> [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
# make
```

3. Convert trained DeepLab v3+ model to MNN model

Refer to [Model dump](https://github.com/david8862/tf-keras-deeplabv3p-model-set#model-dump), [Tensorflow model convert](https://github.com/david8862/tf-keras-deeplabv3p-model-set#tensorflow-model-convert) and [MNN model convert](https://www.yuque.com/mnn/cn/model_convert), we need to:

* dump out inference model from training checkpoint:

    ```
    # python deeplab.py --model_type=mobilenetv2_lite --weights_path=logs/000/<checkpoint>.h5 --classes_path=configs/voc_classes.txt --model_input_shape=512x512 --output_stride=16 --dump_model --output_model_file=model.h5
    ```

* convert keras .h5 model to tensorflow frozen pb model:

    ```
    # python keras_to_tensorflow.py
        --input_model="path/to/keras/model.h5"
        --output_model="path/to/save/model.pb"
    ```

* convert TF pb model to MNN model:

    ```
    # cd <Path_to_MNN>/build/
    # ./MNNConvert -f TF --modelFile model.pb --MNNModel model.pb.mnn --bizCode MNN
    ```
    or

    ```
    # mnnconvert -f TF --modelFile model.pb --MNNModel model.pb.mnn
    ```

MNN support Post Training Integer quantization, so we can use its python CLI interface to do quantization on the generated .mnn model to get quantized .mnn model for ARM acceleration . A json config file [quantizeConfig.json](https://github.com/david8862/tf-keras-deeplabv3p-model-set/blob/master/inference/MNN/configs/quantizeConfig.json) is needed to describe the feeding data:

* Quantized MNN model:

    ```
    # cd <Path_to_MNN>/build/
    # ./quantized.out model.pb.mnn model_quant.pb.mnn quantizeConfig.json
    ```
    or

    ```
    # mnnquant model.pb.mnn model_quant.pb.mnn quantizeConfig.json
    ```

4. Run validate script to check MNN model
```
# cd tf-keras-deeplabv3p-model-set/tools/evaluation/
# python validate_deeplab.py --model_path=model_quant.pb.mnn --classes_path=../../configs/voc_classes.txt --image_file=../../examples/dog.jpg --loop_count=5
```

Visualized segmentation result:

<p align="center">
  <img src="../assets/dog_inference.png">
</p>

#### You can also use [eval.py](https://github.com/david8862/tf-keras-deeplabv3p-model-set#evaluation) to do evaluation on the MNN model


5. Run application to do inference with model, or put all the assets to your ARM board and run if you use cross-compile
```
# cd tf-keras-deeplabv3p-model-set/inference/MNN/build
# ./deeplabSegment -h
Usage: deeplabSegment
--mnn_model, -m: model_name.mnn
--image, -i: image_name.jpg
--classes, -l: classes labels for the model
--input_mean, -b: input mean
--input_std, -s: input standard deviation
--threads, -t: number of threads
--count, -c: loop model run for certain times
--warmup_runs, -w: number of warmup runs
--mask, -k: mask png file to save segment output
--keep_shape, -p: [0|1] keep predict mask as the same shape of input image

# ./deeplabSegment -m model.pb.mnn -i ../../../examples/dog.jpg -t 4 -c 10 -w 2 -l ../../../configs/voc_classes.txt -p 0
Can't Find type=4 backend, use 0 instead
num_classes: 21
image_input: width:512 , height:512, channel: 3
origin image size: width:768, height:576, channel:3
model invoke average time: 111.680200 ms
output tensor: name:pred_mask/Softmax, width:512, height:512, channel: 21
Tensorflow format: NHWC
batch 0:
deeplab_postprocess time: 8.132000 ms
Segment class:
car
bicycle
dog
Segmentation result has been saved to: ./mask.png
```
Here the [classes](https://github.com/david8862/tf-keras-deeplabv3p-model-set/blob/master/configs/voc_classes.txt) file format are the same as used in training part




### Tensorflow-Lite

1. Build TF-Lite lib

We can do either native compile for X86 or cross-compile for ARM

```
# git clone https://github.com/tensorflow/tensorflow <Path_to_TF>
# cd <Path_to_TF>
# ./tensorflow/lite/tools/make/download_dependencies.sh
# make -f tensorflow/lite/tools/make/Makefile   #for X86 native compile
# ./tensorflow/lite/tools/make/build_rpi_lib.sh #for ARM cross compile, e.g Rasperberry Pi
```

2. Build demo inference application
```
# cd tf-keras-deeplabv3p-model-set/inference/tflite
# mkdir build && cd build
# cmake -DTF_ROOT_PATH=<Path_to_TF> [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] [-DTARGET_PLAT=<target>] ..
# make
```
If you want to do cross compile for ARM platform, "CMAKE_TOOLCHAIN_FILE" and "TARGET_PLAT" should be specified. Refer [CMakeLists.txt](https://github.com/david8862/tf-keras-deeplabv3p-model-set/blob/master/inference/tflite/CMakeLists.txt) for details.

3. Convert trained DeepLab v3+ model to tflite model

Tensorflow-lite support both Float32 and UInt8 type model. We can dump out the keras .h5 model to Float32 .tflite model or use [post_train_quant_convert.py](https://github.com/david8862/tf-keras-deeplabv3p-model-set/blob/master/tools/model_converter/post_train_quant_convert.py) script to convert to UInt8 model with TF 2.0 Post-training integer quantization tech, which could be smaller and faster on ARM:

* dump out inference model from training checkpoint:

    ```
    # python deeplab.py --model_type=mobilenetv2_lite --weights_path=logs/000/<checkpoint>.h5 --classes_path=configs/voc_classes.txt --model_input_shape=512x512 --output_stride=16 --dump_model --output_model_file=model.h5
    ```

* convert keras .h5 model to Float32 tflite model:

    ```
    # tflite_convert --keras_model_file=model.h5 --output_file=model.tflite
    ```

* convert keras .h5 model to UInt8 tflite model with TF 2.0 Post-training integer quantization:

    ```
    # cd tf-keras-deeplabv3p-model-set/tools/model_converter/
    # python post_train_quant_convert.py --keras_model_file=model.h5 --dataset_path=../../VOC2012/ --dataset_file=../../VOC2012/ImageSets/Segmentation/val.txt --model_input_shape=512x512 --sample_num=30 --output_file=model_quant.tflite
    ```


4. Run validate script to check TFLite model
```
# cd tf-keras-deeplabv3p-model-set/tools/evaluation/
# python validate_deeplab.py --model_path=model.tflite --classes_path=../../configs/voc_classes.txt --image_file=../../examples/dog.jpg --loop_count=5
```
#### You can also use [eval.py](https://github.com/david8862/tf-keras-deeplabv3p-model-set#evaluation) to do evaluation on the TFLite model



5. Run application to do inference with model, or put assets to ARM board and run if cross-compile
```
# cd tf-keras-deeplabv3p-model-set/inference/tflite/build
# ./deeplabSegment -h
Usage: deeplabSegment
--tflite_model, -m: model_name.tflite
--image, -i: image_name.jpg
--classes, -l: classes labels for the model
--input_mean, -b: input mean
--input_std, -s: input standard deviation
--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not
--threads, -t: number of threads
--count, -c: loop interpreter->Invoke() for certain times
--warmup_runs, -w: number of warmup runs
--mask, -k: mask png file to save segment output
--keep_shape, -p: [0|1] keep predict mask as the same shape of input image
--verbose, -v: [0|1] print more information


# ./deeplabSegment -m model.tflite -i ../../../examples/dog.jpg -t 4 -c 10 -w 2 -l ../../../configs/voc_classes.txt -p 0 -v 1
Loaded model model.tflite
resolved reporter
num_classes: 21
input tensor info: type 1, batch 1, height 512, width 512, channels 3
origin image size: width:768, height:576, channel:3
invoked average time:428.07 ms
output tensor info: name pred_mask/Softmax, type 1, batch 1, height 512, width 512, channels 21
batch 0
deeplab_postprocess time: 7.399 ms
Segment class:
car
bicycle
dog
Segmentation result has been saved to: ./mask.png
```

### On-device evaluation

1. Build your MNN/TFLite version "deeplabSegment" application and put it on device together with [eval_inference.sh](https://github.com/david8862/tf-keras-deeplabv3p-model-set/blob/master/inference/eval_inference.sh). Then run the script to generate on-device inference result txt file for test images:

```
# ./eval_inference.sh
Usage: ./eval_inference.sh <model_file> <image_path> <dataset_file> <class_file> <output_path>
```

The output label mask PNG image will be saved at `<output_path>/<image_id>.png`

2. Use independent evaluation tool [semantic_segment_eval.py](https://github.com/david8862/tf-keras-deeplabv3p-model-set/blob/master/tools/evaluation/onboard/semantic_segment_eval.py) to calculate mIOU and other metrics with result PNG images.

