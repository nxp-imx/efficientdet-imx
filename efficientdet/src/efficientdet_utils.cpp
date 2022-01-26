/*
* Copyright 2022 NXP
* SPDX-License-Identifier: Apache-2.0
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>
#include <regex>
#include <fstream>
#include <unistd.h>
#include "efficientdet_utils.hpp"
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

std::string toUpperCase(const std::string& str){
  std::string res;

  for(auto c : str){
    res.push_back(std::toupper(c));
  }

  return res;
}

int parseModelRes(const std::string& modelName)
{
  std::map<std::string, int> efficientDetResMap{
    {"efficientdet-d0",     512},
    {"efficientdet-d1",     640},
    {"efficientdet-d2",     768},
    {"efficientdet-d3",     896},
    {"efficientdet-d4",     1024},
    {"efficientdet-d5",     1280},
    {"efficientdet-d6",     1280},
    {"efficientdet-d7",     1536},
    {"efficientdet-d7x",    1536},
    {"efficientdet-lite0",  320},
    {"efficientdet-lite1",  384},
    {"efficientdet-lite2",  448},
    {"efficientdet-lite3",  512},
    {"efficientdet-lite3x", 512},
    {"efficientdet-lite4",  512}};

    for (const auto &mapping : efficientDetResMap)
    {
      if(modelName.find(mapping.first) != std::string::npos)
      {
        return efficientDetResMap[mapping.first];
      }
    }

    std::cout << "No suitable resolution detected!\n Make sure your model is named 'efficientdet-<version>.tflite'" << std::endl;
    return -1;
}

bool parseKerasModel(const std::string& modelName)
{
  return (modelName.find("int8") != std::string::npos) || 
         (modelName.find("fp32") != std::string::npos) ||
         (modelName.find("fp16") != std::string::npos);
}

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
      outvec.push_back(static_cast<float>(output[(i * output_size) + j]));
    }

    outputs.push_back(outvec);
  }

  return outputs;
}

// Function for drawing bounding boxes into the input image
// In this method, coordinates aren't normalized to 0-1 range
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

void drawBoundingBoxesScaled(const std::vector<std::vector<float>>& outputs, cv::Mat& image, const int scale)
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
    float ymin   = vec[0] * scale;
    float xmin   = vec[1] * scale;
    float ymax   = vec[2] * scale;
    float xmax   = vec[3] * scale;

    cv::Point topRight(xmin, ymin);
    cv::Point botLeft(xmax, ymax);

    cv::rectangle(image, topRight, botLeft, cv::Scalar(0, 255, 0));
  }
}