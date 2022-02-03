# EfficientDet detection example

This demo shall demonstrate the usage of EfficientDet models on a sample video input. All necessary files for a basic example are provided. This is not a real-time detection and no camera hardware is needed.

## Files in the example directory
There are several files necessary to run the demo.

	1) `efficientdet_demo` - the binary application
	2) `efficientdet-lite0.tflite` - Efficientdet-lite0 tflite model file.
	3) `efficientdet-lite0-int8.tflite` - Efficientdet-lite0, INT8 quantized tflite model file.
	4) `cars_short.mp4` - an example video.

## Running the example
The `efficientdet_demo` binary expects a few arguments
	1) -m : Required. Path to tflite model file. Model name needs to be in format `efficientdet-<version>[-<quantization>].tflite`, in order to correctly deduce the input buffers resolution and postprocess scaling.
	2) -i : Required. Path to the input file. MP4 video formats are supported. Using other formats may cause issues with Gstreamer backend.
	3) -b : Back-end to use. ["CPU", "NNAPI", "VX"], default is "CPU". Case-insensitive.
	4) -d : When using "VX" as a backend, -d argument expects a path to the `.so` delegate file.

Basic execution therefore may look similar to this:
`./efficientdet_demo -m efficientdet-lite0.tflite -i cars_short.mp4`

execution with VX delegate may look similar to this:
`./efficientdet_demo -m efficientdet-lite0-int8.tflite -i cars_short.mp4 -b VX -d /usr/lib/libvx_delegate.so`

After the execution, you should find an `out.avi` file in your directory.

### Modifying the example
If you would like to tweak the parameters of the EfficientDet models, adjust the number of bounding boxes or adjust thresholds for detections, please refer to the README file in [NXP EfficientDet repo](https://bitbucket.sw.nxp.com/projects/AITEC/repos/efficientdet-imx)