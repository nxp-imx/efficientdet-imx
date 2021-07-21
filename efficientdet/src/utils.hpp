#ifndef THESIS_UTILS
#define THESIS_UTILS

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
// to be cleaned, prolly dont need all
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"


std::chrono::milliseconds timedInference(tflite::Interpreter* interpreter);

/*Read an image from imgPath and resize it to WIDTH x HEIGHT */
cv::Mat readImage(const std::string& imgPath, const int width, const int height);


/*	Log a metric into logging file. 
	stream: output file stream to write into. Assumed to be open.
	metric: string representing the measured feature, ie. memory usage
	value : generic value of the metric

	Implementation needs to be in header file because of visibility
*/
template <typename T>
void log(std::ofstream& stream, const char* metric, T value)
{
  stream << metric << ": " << value << std::endl;
}

// Similar function to log just value
template <typename T>
void log(std::ofstream& stream, T value)
{
  stream << value << std::endl;
}

template <typename T>
void log(std::ofstream& stream, std::vector<T> values)
{
	for (T& val : values)
	{
		stream << val << std::endl;
	}
}

template <typename T>
void logOutputs(std::ofstream& stream, std::vector<std::vector<T>> values)
{
	for(std::vector<T>& vec : values)
	{
		std::string out("[ ");
		for(size_t i = 0; i < vec.size(); i++){
			out.append(std::to_string(vec[i]));
			if(i != vec.size() - 1){
				out.append(", ");	
			}
		}
		out.append(" ]");

		stream << out << std::endl;
	}
}

/*  Create Logging file name from model name and current time
	model: string path from current directory, ie. "efficientdet-d0.tflite"
	time : current time in string format. Expected result from getTime()
*/
std::string createLogFileName(const std::string& model,const std::string& time);


// Get time in form of a string for logging purposes
std::string getTime();


// Get memory usage from /proc tree
void logMemoryUsage(std::ofstream& stream);

// Prints vector content to standard output
void printVector(const std::vector<float>& v);


/*  Load output vectors into a vector of outputs
	tensor_ptr:  Pointer to output tensor TfLiteTensor structure
	num_outputs: How many outputs does the model produce (EfficientDet fixed 100)
	output_size: How many elements does one output contain (EfficientDet 7)
				 [batch, ymin, xmin, ymax, xmax, score, label]
*/
std::vector<std::vector<float>> getOutputVectors(const TfLiteTensor* tensor_ptr,
	const int num_outputs,const int output_size);


/*  Draw bounding boxes from outputs to image
	outputs: Vector of outputs from getOutputVectors()
	image  : cv::Mat structure to draw the boxes into. Image is expected to be resized to 
			 model's needs.  
*/
void drawBoundingBoxes(const std::vector<std::vector<float>>& outputs, cv::Mat& image);


// Tensorflow Lite
#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

#endif
