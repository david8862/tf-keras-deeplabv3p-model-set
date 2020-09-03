//
//  deeplabSegment.cpp
//  MNN
//
//  Created by david8862 on 2020/08/25.
//

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <sys/time.h>
#include "MNN/AutoTime.hpp"
#include "MNN/ErrorCode.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace MNN;
using namespace MNN::CV;


// model inference settings
struct Settings {
    int loop_count = 1;
    int number_of_threads = 4;
    int number_of_warmup_runs = 2;
    float input_mean = 0.0f;
    float input_std = 1.0f;
    std::string model_name = "./model.mnn";
    std::string input_img_name = "./dog.jpg";
    std::string classes_file_name = "./classes.txt";
    std::string mask_img_name = "./mask.png";
    bool keep_shape = false;
    bool input_floating = false;
    //bool verbose = false;
    //string input_layer_type = "uint8_t";
};


double get_us(struct timeval t)
{
    return (t.tv_sec * 1000000 + t.tv_usec);
}


void display_usage() {
    std::cout
        << "Usage: deeplabSegment\n"
        << "--mnn_model, -m: model_name.mnn\n"
        << "--image, -i: image_name.jpg\n"
        << "--classes, -l: classes labels for the model\n"
        << "--input_mean, -b: input mean\n"
        << "--input_std, -s: input standard deviation\n"
        << "--threads, -t: number of threads\n"
        << "--count, -c: loop model run for certain times\n"
        << "--warmup_runs, -w: number of warmup runs\n"
        << "--mask, -k: mask png file to save segment output\n"
        << "--keep_shape, -p: [0|1] keep predict mask as the same shape of input image\n"
        //<< "--verbose, -v: [0|1] print more information\n"
        << "\n";
    return;
}


// DeepLab postprocess for prediction mask tensor
void deeplab_postprocess(const Tensor* mask_tensor, uint8_t* mask_array, std::vector<uint8_t> &class_indexes)
{
    // 1. do following transform to get the output segmentation
    //    mask array:
    //
    //    mask = np.argmax(prediction, -1)
    //
    const float* data = mask_tensor->host<float>();
    auto unit = sizeof(float);
    auto dimType = mask_tensor->getDimensionType();

    auto batch   = mask_tensor->batch();
    auto channel = mask_tensor->channel();
    auto height  = mask_tensor->height();
    auto width   = mask_tensor->width();

    int bytesPerRow, bytesPerImage, bytesPerBatch;
    if (dimType == Tensor::TENSORFLOW) {
        // Tensorflow format tensor, NHWC
        MNN_PRINT("Tensorflow format: NHWC\n");

        bytesPerRow   = channel * unit;
        bytesPerImage = width * bytesPerRow;
        bytesPerBatch = height * bytesPerImage;

    } else if (dimType == Tensor::CAFFE) {
        // Caffe format tensor, NCHW
        MNN_PRINT("Caffe format: NCHW\n");

        bytesPerRow   = width * unit;
        bytesPerImage = height * bytesPerRow;
        bytesPerBatch = channel * bytesPerImage;

    } else if (dimType == Tensor::CAFFE_C4) {
        MNN_PRINT("Caffe format: NC4HW4, not supported\n");
        exit(-1);
    } else {
        MNN_PRINT("Invalid tensor dim type: %d\n", dimType);
        exit(-1);
    }

    // Check and clear output mask array
    MNN_ASSERT(mask_array != nullptr);
    bzero((void*)mask_array, height * width * 1 * sizeof(uint8_t));

    for (int b = 0; b < batch; b++) {
        auto bytes = data + b * bytesPerBatch / unit;
        MNN_PRINT("batch %d:\n", b);

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                //get bbox prediction data offset for each anchor, each feature point
                int class_scores_offset, class_scores_step;
                if (dimType == Tensor::TENSORFLOW) {
                    // Tensorflow format tensor, NHWC
                    class_scores_offset = h * width * channel + w * channel;
                    class_scores_step = 1;
                } else if (dimType == Tensor::CAFFE) {
                    // Caffe format tensor, NCHW
                    class_scores_offset = h * width + w;
                    class_scores_step = width * height;

                } else if (dimType == Tensor::CAFFE_C4) {
                    MNN_PRINT("Caffe format: NC4HW4, not supported\n");
                    exit(-1);
                } else {
                    MNN_PRINT("Invalid tensor dim type: %d\n", dimType);
                    exit(-1);
                }

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


//Resize image to model input shape
uint8_t* image_resize(uint8_t* inputImage, int image_width, int image_height, int image_channel, int input_width, int input_height, int input_channel)
{
    // assume the data channel match
    MNN_ASSERT(image_channel == input_channel);

    uint8_t* input_image = (uint8_t*)malloc(input_height * input_width * input_channel * sizeof(uint8_t));
    if (input_image == nullptr) {
        MNN_PRINT("Can't alloc memory\n");
        exit(-1);
    }
    stbir_resize_uint8(inputImage, image_width, image_height, 0,
                     input_image, input_width, input_height, 0, image_channel);

    return input_image;
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
    // record run time for every stage
    struct timeval start_time, stop_time;

    // create model & session
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(s->model_name.c_str()));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_AUTO; //MNN_FORWARD_CPU, MNN_FORWARD_OPENCL
    config.backupType = MNN_FORWARD_CPU;
    config.numThread = s->number_of_threads;

    BackendConfig bnconfig;
    bnconfig.memory = BackendConfig::Memory_Normal; //Memory_High, Memory_Low
    bnconfig.power = BackendConfig::Power_Normal; //Power_High, Power_Low
    bnconfig.precision = BackendConfig::Precision_Normal; //Precision_High, Precision_Low
    config.backendConfig = &bnconfig;

    auto session = net->createSession(config);
    // since we don't need to create other sessions any more,
    // just release model data to save memory
    net->releaseModel();

    // get classes labels and add background label
    std::vector<std::string> classes;
    classes.emplace_back("background");
    std::ifstream classesOs(s->classes_file_name.c_str());
    std::string line;
    while (std::getline(classesOs, line)) {
        classes.emplace_back(line);
    }
    int num_classes = classes.size();
    MNN_PRINT("num_classes: %d\n", num_classes);

    // get input tensor info, assume only 1 input tensor (image_input)
    auto inputs = net->getSessionInputAll(session);
    MNN_ASSERT(inputs.size() == 1);
    auto image_input = inputs.begin()->second;
    int input_width = image_input->width();
    int input_height = image_input->height();
    int input_channel = image_input->channel();
    MNN_PRINT("image_input: width:%d , height:%d, channel: %d\n", input_width, input_height, input_channel);

    //auto shape = image_input->shape();
    //shape[0] = 1;
    //net->resizeTensor(image_input, shape);
    //net->resizeSession(session);

    // load input image
    auto inputPath = s->input_img_name.c_str();
    int image_width, image_height, image_channel;
    uint8_t* inputImage = (uint8_t*)stbi_load(inputPath, &image_width, &image_height, &image_channel, input_channel);
    if (nullptr == inputImage) {
        MNN_ERROR("Can't open %s\n", inputPath);
        return;
    }
    MNN_PRINT("origin image size: width:%d, height:%d, channel:%d\n", image_width, image_height, image_channel);

    // resize input image
    uint8_t* resizeImage = image_resize(inputImage, image_width, image_height, image_channel, input_width, input_height, input_channel);

    // free input image
    stbi_image_free(inputImage);
    inputImage = nullptr;

    // assume input tensor type is float
    MNN_ASSERT(image_input->getType().code == halide_type_float);
    s->input_floating = true;

    // run warm up session
    if (s->loop_count > 1)
        for (int i = 0; i < s->number_of_warmup_runs; i++) {
            fill_data<float>(image_input->host<float>(), resizeImage,
                input_width, input_height, input_channel, s);
            if (net->runSession(session) != NO_ERROR) {
                MNN_PRINT("Failed to invoke MNN!\n");
            }
        }

    // run model sessions to get output
    gettimeofday(&start_time, nullptr);
    for (int i = 0; i < s->loop_count; i++) {
        fill_data<float>(image_input->host<float>(), resizeImage,
            input_width, input_height, input_channel, s);
        if (net->runSession(session) != NO_ERROR) {
            MNN_PRINT("Failed to invoke MNN!\n");
        }
    }
    gettimeofday(&stop_time, nullptr);
    MNN_PRINT("model invoke average time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / (1000 * s->loop_count));

    // get output tensor info, assume only 1 output tensor (pred_mask/Softmax)
    // image_input: 1 x 512 x 512 x 3
    // "pred_mask/Softmax": 1 x 512 x 512 x num_classes
    auto outputs = net->getSessionOutputAll(session);
    MNN_ASSERT(outputs.size() == 1);

    auto mask_output = outputs.begin()->second;
    int mask_width = mask_output->width();
    int mask_height = mask_output->height();
    int mask_channel = mask_output->channel();
    MNN_PRINT("output tensor: name:%s, width:%d, height:%d, channel: %d\n", outputs.begin()->first.c_str(), mask_width, mask_height, mask_channel);

    // check if predict mask channel number
    // matches classes definition
    MNN_ASSERT(num_classes == mask_channel);

    // Copy output tensors to host, for further postprocess
    auto dim_type = mask_output->getDimensionType();
    if (mask_output->getType().code != halide_type_float) {
        dim_type = Tensor::TENSORFLOW;
    }
    std::shared_ptr<Tensor> output_tensor(new Tensor(mask_output, dim_type));
    mask_output->copyToHostTensor(output_tensor.get());

    // Now we only support float32 type output tensor
    MNN_ASSERT(output_tensor->getType().code == halide_type_float);
    MNN_ASSERT(output_tensor->getType().bits == 32);

    // Alloc mask array for post process
    uint8_t* mask_array = (uint8_t*)malloc(mask_height * mask_width * 1 * sizeof(uint8_t));
    if (mask_array == nullptr) {
        MNN_PRINT("Can't alloc memory\n");
        exit(-1);
    }
    std::vector<uint8_t> class_indexes;

    // Do deeplab_postprocess to generate mask array
    gettimeofday(&start_time, nullptr);
    deeplab_postprocess(output_tensor.get(), mask_array, class_indexes);
    gettimeofday(&stop_time, nullptr);
    MNN_PRINT("deeplab_postprocess time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / 1000);

    int save_width, save_height;
    if (s->keep_shape) {
        // Resize the prediction mask back to original image shape
        uint8_t* origin_mask_array = image_resize(mask_array, mask_width, mask_height, 1, image_width, image_height, 1);
        // free prediction mask
        free(mask_array);
        mask_array = origin_mask_array;
        save_width = image_width;
        save_height = image_height;
    } else {
        save_width = mask_width;
        save_height = mask_height;
    }

    // Show detection result
    MNN_PRINT("Segment class:\n");
    for(auto class_index : class_indexes) {
        MNN_PRINT("%s\n", classes[class_index].c_str());
    }

    // Save mask array to png image file
    stbi_write_png(s->mask_img_name.c_str(), save_width, save_height, 1, mask_array, 0);
    MNN_PRINT("Segmentation result has been saved to: %s\n", s->mask_img_name.c_str());

    // Release session and model
    net->releaseSession(session);
    //net->releaseModel();
    return;
}


int main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"mnn_model", required_argument, nullptr, 'm'},
        {"image", required_argument, nullptr, 'i'},
        {"classes", required_argument, nullptr, 'l'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"threads", required_argument, nullptr, 't'},
        {"count", required_argument, nullptr, 'c'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        {"mask", required_argument, nullptr, 'k'},
        {"keep_shape", required_argument, nullptr, 'p'},
        //{"verbose", required_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "b:c:hi:l:k:m:p:s:t:w:", long_options,
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
      case 'p':
        s.keep_shape =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      //case 'v':
        //s.verbose =
            //strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        //break;
      case 'w':
        s.number_of_warmup_runs =
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
    }
  }
  RunInference(&s);
  return 0;
}

