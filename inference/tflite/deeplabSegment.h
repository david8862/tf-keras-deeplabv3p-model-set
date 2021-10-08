//
//  deeplabSegment.h
//  Tensorflow-lite
//
//  Created by david8862 on 2020/08/26.
//
//

#ifndef DEEPLAB_SEGMENT_H_
#define DEEPLAB_SEGMENT_H_

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

namespace deeplabSegment {

struct Settings {
  bool verbose = false;
  bool accel = false;
  bool input_floating = false;
  bool allow_fp16 = false;
  bool keep_shape = false;
  int loop_count = 1;
  float input_mean = 127.5f;
  float input_std = 127.5f;
  std::string model_name = "./model.tflite";
  //tflite::FlatBufferModel* model;
  std::string input_img_name = "./dog.jpg";
  std::string classes_file_name = "./classes.txt";
  std::string mask_img_name = "./mask.png";
  //std::string input_layer_type = "uint8_t";
  int number_of_threads = 4;
  int number_of_warmup_runs = 2;
};

}  // namespace deeplabSegment

#endif  // DEEPLAB_SEGMENT_H_
