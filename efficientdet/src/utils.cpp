#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>
#include <regex>
#include <fstream>
#include <unistd.h>
#include "utils.hpp"
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

cv::Mat readImage(const std::string& imgPath, const int width, const int height)
{
  cv::Mat img;
  cv::Mat resizedImg;
  cv::Mat RGBImg;

  // Open image
  // Opening using imread will have continuous memory
  img = cv::imread(imgPath, cv::IMREAD_COLOR);
  if (img.empty()){
      fprintf(stderr, "Failed to read image ...\n");
      return img;
  }

  // OpenCV loads images in BGR format. image has to be converted to RGB.
  cv::cvtColor(img, RGBImg, cv::COLOR_BGR2RGB);

  // Resize input image to fit the model
  cv::resize(RGBImg, resizedImg, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);

  return resizedImg;
}

// Read memory statistics from /proc/self/smaps_rollup
void logMemoryUsage(std::ofstream& stream)
{
  std::ifstream memInfo("/proc/self/smaps_rollup");
  std::string line;

  while(std::getline(memInfo, line)){
    log<std::string>(stream, line);
  }

  stream << std::endl;

  memInfo.close();
}

// Build the output log filename for further manipulation
std::string createLogFileName(const std::string& model,const std::string& time)
{
  std::string modelCopy(model);
  std::string timeCopy(time);

  std::string prep        = modelCopy.append(timeCopy);
  std::string prep2       = std::regex_replace(prep,  std::regex("\\.tflite"), "-");
  std::string prep3       = std::regex_replace(prep2, std::regex("--"), "-");
  std::string prep4       = prep3.substr(0, prep3.size() - 1);
  std::string logFileName = prep4.append(".txt");

  return logFileName; 
}

// Return current time in a string format
std::string getTime()
{
  std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::string res(std::ctime(&time));

  // Replace all spaces by '-'
  std::replace(res.begin(), res.end(), ' ', '-');

  return res;
}

// Performs a single inference with time measurement. returns the duration
std::chrono::milliseconds timedInference(tflite::Interpreter* interpreter)
{
    auto inferenceTimeStart = std::chrono::high_resolution_clock::now();

    if(interpreter->Invoke() != kTfLiteOk){
      printf("Error happened in Invoke()! Logs will be invalid!\n");
      return std::chrono::duration_cast<std::chrono::milliseconds>(inferenceTimeStart - inferenceTimeStart);
    }

    auto inferenceTimeEnd = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(inferenceTimeEnd - inferenceTimeStart);
}

void printVector(const std::vector<float>& v)
{
  for (size_t i = 0; i < v.size(); ++i)
  {
    std::cout << v[i] << ", ";
  }

  std::cout << std::endl;
}

std::vector<std::vector<float>> getOutputVectors(const TfLiteTensor* tensor_ptr,const int num_outputs,const int output_size)
{
  std::vector<std::vector<float>> outputs;

  float* output = reinterpret_cast<float*>(tensor_ptr->data.raw);

  for (int i = 0; i < num_outputs; ++i)
  {
    std::vector<float> outvec;

    for (int j = 0; j < output_size; ++j)
    {
      outvec.push_back(output[(i * output_size) + j]);
    }

    outputs.push_back(outvec);
  }

  return outputs;
}

// Debugging function
void drawBoundingBoxes(const std::vector<std::vector<float>>& outputs, cv::Mat& image)
{
  std::vector<std::vector<float>> boxesToDraw;

  // First output will always be drawn
  boxesToDraw.push_back(outputs[0]);

  for(size_t i = 1; i < outputs.size() - 1; i++){
    if(outputs[i] != outputs[i-1]){
      boxesToDraw.push_back(outputs[i]);
    }
  }

  for(std::vector<float>& vec : boxesToDraw){
    int imgNum = vec[0];
    int ymin   = vec[1];
    int xmin   = vec[2];
    int ymax   = vec[3];
    int xmax   = vec[4];
    int score  = vec[5];
    int label  = vec[6];

    cv::Point topRight(xmin, ymin);
    cv::Point botLeft(xmax, ymax);

    cv::rectangle(image, topRight, botLeft, cv::Scalar(0, 255, 0));
  }
}