#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <experimental/filesystem>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "tensorflow/lite/delegates/external/external_delegate.h"
#include "opencv2/opencv.hpp"
#include "efficientdet_utils.hpp"
#include "cxxopts.hpp"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    fprintf(stderr, "Invalid number of arguments ... \n");
    fprintf(stderr, "Usage: ./efficientdet_demo <tflite model> <path to input file> [TF_DELEGATE_PATH]\n");
    return 1;
  }

  cxxopts::Options appOptions("EfficientDet detection example", "Example object detection using EfficientDet on an input video file.");

  appOptions.add_options()
  ("m,model", "Path to EfficientDet model", cxxopts::value<std::string>()->default_value(""))
  ("i,input", "Path to input video file", cxxopts::value<std::string>()->default_value(""))
  ("b,backend", "Backend to use for inference (CPU, NNAPI, ...)", cxxopts::value<std::string>()->default_value("CPU"))
  ("d,delegate", "Path to external delegate (ie. VX)", cxxopts::value<std::string>()->default_value(""));

  std::cout << "EfficientDet detection example" << std::endl;
  std::cout << "==============================" << std::endl;

  auto parsedOptions = appOptions.parse(argc, argv);

  std::string modelFile    = parsedOptions["model"].as<std::string>();
  std::string videoFile    = parsedOptions["input"].as<std::string>();
  std::string backend      = parsedOptions["backend"].as<std::string>();
  std::string delegatePath = parsedOptions["delegate"].as<std::string>();
  
  if(modelFile.empty() || videoFile.empty()){
    std::cout << "Please provide path to model (-m) and input file (-i) as command line arguments" << std::endl;
    return 1;
  }

  if(toUpperCase(backend) == std::string("VX") && delegatePath.empty()){
    std::cout << "No VX_DELEGATE supplied ..." << std::endl;
    return 1;
  }

  int  CHANNELS    = 3;
  int  MODEL_RES   = parseModelRes(modelFile);
  bool KERAS_MODEL = parseKerasModel(modelFile);

  // Prepare string streams for FPS display
  std::stringstream fpsString;
  fpsString.precision(4);

  cv::Mat img;
  cv::Mat RGBImg;
  cv::Mat outMat;

  std::vector<std::vector<float>> outputs;

  int imgCnt = 0;

  // Open video file
  cv::VideoCapture cap(videoFile);

  if(!cap.isOpened()){
    std::cout << "Failed to open input file ..." << std::endl;
    return -1;
  }

  double fps = cap.get(cv::CAP_PROP_FPS);
  std::cout << "Input File FPS: " << fps << std::endl;

  double framecount = cap.get(cv::CAP_PROP_FRAME_COUNT);
  std::cout << "Input File Frame Count: " << framecount << std::endl;

  double framewidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  std::cout << "Input file Frame width: " << framewidth << std::endl;

  double frameheight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  std::cout << "Input file Frame height: " << frameheight << std::endl;

  // Parse the fourcc from input video file
  // fourcc is returned as double, need to parse
  int fourcc = cap.get(cv::CAP_PROP_FOURCC);
  char first = fourcc & 255;
  char second = (fourcc >> 8) & 255;
  char third = (fourcc >> 16) & 255;
  char fourth = (fourcc >> 24) & 255;

  std::cout << "Input file fourcc: " << first << second << third << fourth << std::endl;
  std::cout << "==============================" << std::endl << std::endl;
  
  // Prepare output file
  // Output file will have the same resolution as input file
  // Output format is avi because mp4 is not supported
  cv::VideoWriter  out("out.avi",
                  cv::CAP_GSTREAMER,
                  cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                  cap.get(cv::CAP_PROP_FPS),
                  cv::Size(framewidth, frameheight),
                  true);

  if(!out.isOpened()){
    std::cout << "Failed to open output file ..." << std::endl;
    return -1;
  }

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(modelFile.c_str());
  TFLITE_MINIMAL_CHECK(model != nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  interpreter->SetNumThreads(4);

  interpreter->SetAllowFp16PrecisionForFp32(true);

  if (toUpperCase(backend) == std::string("NNAPI")){
    tflite::StatefulNnApiDelegate::Options options;
    auto delegate = tflite::evaluation::CreateNNAPIDelegate(options);
    if (!delegate) {
      std::cout << "NNAPI acceleration is unsupported on this platform." << std::endl;
    } else {
      std::cout << "Use NNAPI acceleration." << std::endl;
    }

    if (interpreter->ModifyGraphWithDelegate(std::move(delegate)) !=
        kTfLiteOk) {
      std::cout << "Failed to apply NNAPI delegate." << std::endl;
      return 1;
    }
  }

  else if(toUpperCase(backend) == std::string("VX")){
    auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault(delegatePath.c_str());
    auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
    if(!ext_delegate_ptr){
      std::cout << "VX acceleration failed to initialize." << std::endl;
    }
    else{
      std::cout << "VX acceleration enabled." << std::endl;
    }

    if(interpreter->ModifyGraphWithDelegate(ext_delegate_ptr) != kTfLiteOk){
      std::cout << "Failed to apply VX delegate." << std::endl;
    }
  }

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  TfLiteTensor* inTensor  = interpreter->input_tensor(0);
  TfLiteTensor* outTensor = interpreter->output_tensor(0);

  int8_t* input = reinterpret_cast<int8_t*>(inTensor->data.raw);

  // Evaluate on provided video file
  while(true){

    // Capture a frame
    cap >> img;

    if(img.empty()){
      std::cout << "End of file, exitting ..." << std::endl;
      break;
    }

    // OpenCV loads images in BGR format. image has to be converted to RGB.
    cv::cvtColor(img, RGBImg, cv::COLOR_BGR2RGB);

    // Resize input image to fit the model
    cv::resize(RGBImg, img, cv::Size(MODEL_RES, MODEL_RES), 0, 0, cv::INTER_CUBIC);

    memcpy((void*)input, (void*) img.data, MODEL_RES * MODEL_RES * CHANNELS * sizeof(int8_t));

    auto inferenceTimeDuration = timedInference(interpreter.get());

    double fps = 1 / (static_cast<int>(inferenceTimeDuration.count()) / 1000.0);

    fpsString << fps;

    // Keras-converted models have different output tensors
    if(KERAS_MODEL){
      outputs = getOutputVectors(outTensor, 100, 4);
      drawBoundingBoxesScaled(outputs, img, MODEL_RES);
    }

    else{
      outputs = getOutputVectors(outTensor, 100, 7);
      drawBoundingBoxes(outputs, img);
    }

    // Convert back to BGR since OpenCV works with BGR
    cv::cvtColor(img, outMat, cv::COLOR_RGB2BGR);

    cv::resize(outMat, outMat, cv::Size(framewidth, frameheight), 0, 0, cv::INTER_CUBIC);

    cv::putText(outMat, "FPS: " + fpsString.str(),
                 cv::Point(15, 45), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(255, 0, 0), 2);

    // Clear the content of sstream
    fpsString.str(std::string());

    out << outMat;

    std::cout << "Frames processed: " << imgCnt++ << " / " << framecount << std::endl;
  }

  // Finalize the output video
  out.release();

  std::cout << "Done" << std::endl;

  return 0;
}