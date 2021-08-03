#include <cstdio>
#include <cstdint> // uint8_t
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <experimental/filesystem> // inference all images in a file.
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "efficientdet_utils.hpp"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "Invalid number of arguments ... \n");
    fprintf(stderr, "Usage: ./efficientdet_demo <tflite model> <path to input file> <input size>\n");
    return 1;
  }

  std::cout << "EfficientDet detection example" << std::endl;
  std::cout << "==============================" << std::endl;

  const char* modelFile = argv[1];
  const char* videoFile = argv[2];
  const char* modelRes  = argv[3];

  int CHANNELS  = 3; // default for images
  int MODEL_RES = std::stoi(std::string(modelRes));

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
    printf("Failed to open input file ...\n");
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
      tflite::FlatBufferModel::BuildFromFile(modelFile);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  interpreter->SetNumThreads(4);

  interpreter->SetAllowFp16PrecisionForFp32(true);

   tflite::StatefulNnApiDelegate::Options options;
    auto delegate = tflite::evaluation::CreateNNAPIDelegate(options);
    if (!delegate) {
      std::cout << "NNAPI acceleration is unsupported on this platform.\n";
    } else {
      std::cout << "Use NNAPI acceleration.\n";
    }

    if (interpreter->ModifyGraphWithDelegate(std::move(delegate)) !=
        kTfLiteOk) {
      std::cout << "Failed to apply NNAPI delegate.";
      exit(-1);
    }

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  TfLiteTensor* inTensor   = interpreter->input_tensor(0);
  TfLiteTensor* outTensor  = interpreter->output_tensor(0);

  uint8_t* input = reinterpret_cast<uint8_t*>(inTensor->data.raw);

  // Evaluate on provided video file
  while(true)
  {
    // Capture a frame
    cap >> img;

    if(img.empty()){
      printf("End of file, exitting ...\n");
      break;
    }

    // OpenCV loads images in BGR format. image has to be converted to RGB.
    cv::cvtColor(img, RGBImg, cv::COLOR_BGR2RGB);

    // Resize input image to fit the model
    cv::resize(RGBImg, img, cv::Size(MODEL_RES, MODEL_RES), 0, 0, cv::INTER_CUBIC);

    memcpy((void*)input, (void*) img.data, MODEL_RES * MODEL_RES * CHANNELS * sizeof(uint8_t));

    auto inferenceTimeDuration = timedInference(interpreter.get());

    double fps = 1 / (static_cast<int>(inferenceTimeDuration.count()) / 1000.0);

    fpsString << fps;

    outputs = getOutputVectors(outTensor, 100, 7);

    drawBoundingBoxes(outputs, img);

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