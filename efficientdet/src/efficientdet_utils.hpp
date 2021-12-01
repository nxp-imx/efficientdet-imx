/*
* Copyright 2021 NXP
*/

#ifndef EFFICIENTDET_UTILS
#define EFFICIENTDET_UTILS

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

/*
	Converts a given string to uppercase format
*/
std::string toUpperCase(const std::string& str);

/*
	Performs a single inference with time measurement. Time measurement is then
	used to estimate FPS.

	interpreter: pointer to TfLite::Interpreter instance
*/
std::chrono::milliseconds timedInference(tflite::Interpreter* interpreter);


/*
	Determine the correct input size from the model name

	modelName: String name of the executed model
*/
int parseModelRes(const std::string& modelName);


/*
	Find if the supplied model was converted via Keras interface.
	Keras-converted models have different output tensor dimensions, format and
	different bbox coordinates scaling.

	modelName: String name of the executed model
*/
bool parseKerasModel(const std::string& modelName);


/*
	Read an image from imgPath and resize it to (WIDTH x HEIGHT)

	imgPath: Path to image
	width:   Width  to which image should be resized
	height:  Height to which image should be resized
*/
cv::Mat readImage(const std::string& imgPath, const int width, const int height);


/*
	Prints vector content to standard output

	v: Vector of float values
*/
void printVector(const std::vector<float>& v);


/*
  Load output vectors into a vector of outputs

	tensor_ptr:  Pointer to output tensor TfLiteTensor structure
	num_outputs: How many outputs does the model produce (EfficientDet fixed 100)
	output_size: How many elements does each output contain (EfficientDet 7)
				 			 [batch, ymin, xmin, ymax, xmax, score, label]
*/
std::vector<std::vector<float>> getOutputVectors(const TfLiteTensor* tensor_ptr,
	const int num_outputs,const int output_size);


/*
  Draw bounding boxes from outputs to image. This function expects NON-NORMALIZED 
  bounding box coordinates. 

	outputs: Vector of outputs from getOutputVectors()
	image  : cv::Mat structure to draw the boxes into. Image is expected to be resized to 
					 model's needs.
*/
void drawBoundingBoxes(const std::vector<std::vector<float>>& outputs, cv::Mat& image);


/*
  Draw bounding boxes from outputs to image. This function expects NORMALIZED
  bounding box coordinates. Keras-converted models output normalized coordinates.

	outputs: Vector of outputs from getOutputVectors()
	image  : cv::Mat structure to draw the boxes into. Image is expected to be resized to 
					 model's needs.
*/
void drawBoundingBoxesScaled(const std::vector<std::vector<float>>& outputs, cv::Mat& image, const int scale);


// Tensorflow Lite
#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

#endif
