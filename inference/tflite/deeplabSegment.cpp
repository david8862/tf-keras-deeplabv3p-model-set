//
//  deeplabSegment.cpp
//  Tensorflow-lite
//
//  Created by david8862 on 2020/08/26.
//
#include <fcntl.h>
#include <math.h>
#include <getopt.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <assert.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <numeric>
#include <algorithm>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"

#include "deeplabSegment.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define LOG(x) std::cout

namespace deeplabSegment {

double get_us(struct timeval t)
{
    return (t.tv_sec * 1000000 + t.tv_usec);
}


// DeepLab postprocess for prediction mask tensor
void deeplab_postprocess(const TfLiteTensor* mask_tensor, uint8_t* mask_array, std::vector<uint8_t> &class_indexes)
{
    // 1. do following transform to get the output segmentation
    //    mask array:
    //
    //    mask = np.argmax(prediction, -1)
    //
    const float* data = reinterpret_cast<float*>(mask_tensor->data.raw);

    TfLiteIntArray* output_dims = mask_tensor->dims;
    int batch = output_dims->data[0];
    int height = output_dims->data[1];
    int width = output_dims->data[2];
    int channel = output_dims->data[3];
    auto unit = sizeof(float);

    // TF/TFLite tensor format: NHWC
    auto bytesPerRow   = channel * unit;
    auto bytesPerImage = width * bytesPerRow;
    auto bytesPerBatch = height * bytesPerImage;

    // Check and clear output mask array
    assert(mask_array != nullptr);
    bzero((void*)mask_array, height * width * 1 * sizeof(uint8_t));

    for (int b = 0; b < batch; b++) {
        auto bytes = data + b * bytesPerBatch / unit;
        LOG(INFO) << "batch " << b << "\n";

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                //get bbox prediction data offset for each anchor, each feature point
                int class_scores_offset, class_scores_step;
                // Tensorflow format tensor, NHWC
                class_scores_offset = h * width * channel + w * channel;
                class_scores_step = 1;

                // Get class index with max score (index 0 should be background),
                // just as Python postprocess:
                //
                // mask = np.argmax(prediction, -1)
                //
                uint8_t class_index = 0;
                float max_score = 0.0;
                for (int i = 0; i < channel; i++) {
                    if (bytes[class_scores_offset + i * class_scores_step] > max_score) {
                        class_index = i;
                        max_score = bytes[class_scores_offset + i * class_scores_step];
                    }
                }
                int mask_offset = h * width + w;
                mask_array[mask_offset] = class_index;

                if(class_index != 0 && std::count(class_indexes.begin(), class_indexes.end(), class_index) == 0) {
                    class_indexes.emplace_back(class_index);
                }
            }
        }
    }
    return;
}


// Resize image to model input shape
uint8_t* image_resize(uint8_t* inputImage, int image_width, int image_height, int image_channel, int input_width, int input_height, int input_channel)
{
    // assume the data channel match
    assert(image_channel == input_channel);

    uint8_t* input_image = (uint8_t*)malloc(input_height * input_width * input_channel * sizeof(uint8_t));
    if (input_image == nullptr) {
        LOG(ERROR) << "Can't alloc memory\n";
        exit(-1);
    }
    stbir_resize_uint8(inputImage, image_width, image_height, 0,
                     input_image, input_width, input_height, 0, image_channel);

    return input_image;
}


// Nearest resize predict mask
uint8_t* mask_resize(uint8_t* input_mask, int mask_width, int mask_height, int target_width, int target_height)
{
    float scale_w = float(mask_width) / float(target_width);
    float scale_h = float(mask_height) / float(target_height);

    uint8_t* resized_mask = (uint8_t*)malloc(target_height * target_width * sizeof(uint8_t));
    if (resized_mask == nullptr) {
        LOG(ERROR) << "Can't alloc memory\n";
        exit(-1);
    }

    // go through resized mask to get nearest value
    for (int h = 0; h < target_height; h++) {
        for (int w = 0; w < target_width; w++) {
            int mask_x = int(w * scale_w);
            int mask_y = int(h * scale_h);
            resized_mask[h*target_width + w] = input_mask[mask_y*mask_width + mask_x];
        }
    }

    return resized_mask;
}


template <class T>
void fill_data(T* out, uint8_t* in, int input_width, int input_height,
            int input_channels, Settings* s) {
  auto output_number_of_pixels = input_height * input_width * input_channels;

  for (int i = 0; i < output_number_of_pixels; i++) {
    if (s->input_floating)
      out[i] = (in[i] - s->input_mean) / s->input_std;
    else
      out[i] = (uint8_t)in[i];
  }

  return;
}


void RunInference(Settings* s) {
  if (!s->model_name.c_str()) {
    LOG(ERROR) << "no model file name\n";
    exit(-1);
  }

  // load model
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
  if (!model) {
    LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
    exit(-1);
  }
  //s->model = model.get();
  LOG(INFO) << "Loaded model " << s->model_name << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";

  // prepare model interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }

  interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);
  if (s->number_of_threads != -1) {
    interpreter->SetNumThreads(s->number_of_threads);
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

  // get classes labels and add background label
  std::vector<std::string> classes;
  classes.emplace_back("background");
  std::ifstream classesOs(s->classes_file_name.c_str());
  std::string line;
  while (std::getline(classesOs, line)) {
      classes.emplace_back(line);
  }
  int num_classes = classes.size();
  LOG(INFO) << "num_classes: " << num_classes << "\n";


  // assuming one input only
  const std::vector<int> inputs = interpreter->inputs();
  assert(inputs.size() == 1);

  // get input dimension from the input tensor metadata
  int input = interpreter->inputs()[0];
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int input_batch = dims->data[0];
  int input_height = dims->data[1];
  int input_width = dims->data[2];
  int input_channels = dims->data[3];

  if (s->verbose) LOG(INFO) << "input tensor info: "
                            << "type " << interpreter->tensor(input)->type << ", "
                            << "batch " << input_batch << ", "
                            << "height " << input_height << ", "
                            << "width " << input_width << ", "
                            << "channels " << input_channels << "\n";

  // read input image
  int image_width, image_height, image_channel;

  auto input_image = (uint8_t*)stbi_load(s->input_img_name.c_str(), &image_width, &image_height, &image_channel, 3);
  if (input_image == nullptr) {
      LOG(FATAL) << "Can't open" << s->input_img_name << "\n";
      exit(-1);
  }

  LOG(INFO) << "origin image size: width:" << image_width
            << ", height:" << image_height
            << ", channel:" << image_channel
            << "\n";

  // resize input image
  uint8_t* resizeImage = image_resize(input_image, image_width, image_height, image_channel, input_width, input_height, input_channels);

  // free input image
  stbi_image_free(input_image);
  input_image = nullptr;

  // fulfill image data to model input tensor
  switch (interpreter->tensor(input)->type) {
    case kTfLiteFloat32:
      s->input_floating = true;
      fill_data<float>(interpreter->typed_tensor<float>(input), resizeImage,
                    input_width, input_height, input_channels, s);
      break;
    case kTfLiteUInt8:
      fill_data<uint8_t>(interpreter->typed_tensor<uint8_t>(input), resizeImage,
                    input_width, input_height, input_channels, s);
      break;
    default:
      LOG(FATAL) << "cannot handle input type "
                 << interpreter->tensor(input)->type << " yet";
      exit(-1);
  }

  // run warm up session
  if (s->loop_count > 1)
    for (int i = 0; i < s->number_of_warmup_runs; i++) {
      if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke tflite!\n";
      }
    }

  // run model sessions to get output
  struct timeval start_time, stop_time;
  gettimeofday(&start_time, nullptr);
  for (int i = 0; i < s->loop_count; i++) {
    if (interpreter->Invoke() != kTfLiteOk) {
      LOG(FATAL) << "Failed to invoke tflite!\n";
    }
  }
  gettimeofday(&stop_time, nullptr);
  LOG(INFO) << "invoked average time:" << (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000) << " ms \n";

  // get output tensor info, assume only 1 output tensor (pred_mask/Softmax)
  // image_input: 1 x 512 x 512 x 3
  // "pred_mask/Softmax": 1 x 512 x 512 x num_classes
  const std::vector<int> outputs = interpreter->outputs();
  assert(outputs.size() == 1);

  // Now we only support float32 type output tensor
  assert(mask_output->type == kTfLiteFloat32);

  int output = interpreter->outputs()[0];
  TfLiteTensor* mask_output = interpreter->tensor(output);

  TfLiteIntArray* output_dims = mask_output->dims;
  int mask_batch = output_dims->data[0];
  int mask_height = output_dims->data[1];
  int mask_width = output_dims->data[2];
  int mask_channels = output_dims->data[3];

  if (s->verbose) LOG(INFO) << "output tensor info: "
      << "name " << mask_output->name << ", "
          << "type " << mask_output->type << ", "
          << "batch " << mask_batch << ", "
          << "height " << mask_height << ", "
          << "width " << mask_width << ", "
          << "channels " << mask_channels << "\n";

  // check if predict mask channel number
  // matches classes definition
  assert(num_classes == mask_channels);

  // Alloc mask array for post process
  uint8_t* mask_array = (uint8_t*)malloc(mask_height * mask_width * 1 * sizeof(uint8_t));
  if (mask_array == nullptr) {
      LOG(ERROR) << "Can't alloc memory\n";
      exit(-1);
  }
  std::vector<uint8_t> class_indexes;

  // Do deeplab_postprocess to generate mask array
  gettimeofday(&start_time, nullptr);
  deeplab_postprocess(mask_output, mask_array, class_indexes);
  gettimeofday(&stop_time, nullptr);
  LOG(INFO) << "deeplab_postprocess time: " << (get_us(stop_time) - get_us(start_time)) / 1000 << " ms\n";

  int save_width, save_height;
  if (s->keep_shape) {
      // Resize the prediction mask back to original image shape
      uint8_t* origin_mask_array = mask_resize(mask_array, mask_width, mask_height, image_width, image_height);
      // free prediction mask
      free(mask_array);
      mask_array = origin_mask_array;
      save_width = image_width;
      save_height = image_height;
  } else {
      save_width = mask_width;
      save_height = mask_height;
  }

  // Show segment class result
  LOG(INFO) << "Segment class:\n";
  for(auto class_index : class_indexes) {
      LOG(INFO) << classes[class_index] << "\n";
  }

  // Save mask array to png image file
  stbi_write_png(s->mask_img_name.c_str(), save_width, save_height, 1, mask_array, 0);
  LOG(INFO) << "Segmentation result has been saved to: " << s->mask_img_name << "\n";

  return;
}

void display_usage() {
  LOG(INFO)
      << "Usage: deeplabSegment\n"
      << "--tflite_model, -m: model_name.tflite\n"
      << "--image, -i: image_name.jpg\n"
      << "--classes, -l: classes labels for the model\n"
      << "--input_mean, -b: input mean\n"
      << "--input_std, -s: input standard deviation\n"
      << "--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not\n"
      << "--threads, -t: number of threads\n"
      << "--count, -c: loop interpreter->Invoke() for certain times\n"
      << "--warmup_runs, -w: number of warmup runs\n"
      << "--mask, -k: mask png file to save segment output\n"
      << "--keep_shape, -p: [0|1] keep predict mask as the same shape of input image\n"
      << "--verbose, -v: [0|1] print more information\n"
      << "\n";
}


int Main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"tflite_model", required_argument, nullptr, 'm'},
        {"image", required_argument, nullptr, 'i'},
        {"classes", required_argument, nullptr, 'l'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"threads", required_argument, nullptr, 't'},
        {"allow_fp16", required_argument, nullptr, 'f'},
        {"count", required_argument, nullptr, 'c'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        {"mask", required_argument, nullptr, 'k'},
        {"keep_shape", required_argument, nullptr, 'p'},
        {"verbose", required_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "b:c:f:i:hk:l:m:p:s:t:v:w:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'f':
        s.allow_fp16 =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'i':
        s.input_img_name = optarg;
        break;
      case 'l':
        s.classes_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      case 'v':
        s.verbose =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'w':
        s.number_of_warmup_runs =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'p':
        s.keep_shape =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'k':
        s.mask_img_name = optarg;
        break;
      case 'h':
      case '?':
      default:
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
        exit(-1);
    }
  }
  RunInference(&s);
  return 0;
}

}  // namespace deeplabSegment

int main(int argc, char** argv) {
  return deeplabSegment::Main(argc, argv);
}
