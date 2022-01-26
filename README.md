# EfficientDet

EfficientDet is a family of convolutional neural networks used for detecting objects in an image or video input. EfficientDet puts emphasis on efficiency and scalability, while achieving state-of-the-art performance results. EfficientDets come in several versions. Models range from versions D0 - D7, with D0 being the smallest and D7 being the biggest. Higher versions perform better in terms of accuracy, but worse in terms of inference latency.

| EfficientDet version | Input size |  COCO 2017 mAP  |
| ---------------------|------------|------------------
| Efficientdet-D0      | 512        | 34.6            |
| Efficientdet-D1      | 640        | 40.5            |
| Efficientdet-D2      | 768        | 43.9            |
| Efficientdet-D3      | 896        | 47.2            |
| Efficientdet-D4      | 1024       | 49.7            |
| Efficientdet-D5      | 1280       | 51.5            |
| Efficientdet-D6      | 1280       | 52.6            |
| Efficientdet-D7      | 1536       | 53.7            |
| Efficientdet-D7x     | 1536       | 55.1            |

Along with aforementioned Efficientdet versions, even more efficient versions of EfficientDet exist. They are called Efficientdet-lite and again come in several versions. Compared to their standard counterparts, they use faster, though less accurate operations and operate on lower sized inputs.

| EfficientDet-lite version | Input size |  COCO 2017 mAP  |
| --------------------------|------------|------------------
| Efficientdet-lite0        | 320        | 26.1            |
| Efficientdet-lite1        | 384        | 31.5            |
| Efficientdet-lite2        | 448        | 35.1            |
| Efficientdet-lite3        | 512        | 38.8            |
| Efficientdet-lite3x       | 512        | 42.6            |
| Efficientdet-lite4        | 512        | 43.2            |

For more detailed information, refer to the following resources:
* [arXiv paper](https://arxiv.org/abs/1911.09070) 
* [Official GitHub repository](https://github.com/google/automl/tree/master/efficientdet)

# EfficientDet example on i.MX8

We provide an example demo, showcasing the usage of EfficientDets on i.MX8 boards.

Example video available for download from [Pexels](https://www.pexels.com/video/different-kinds-of-vehicles-on-the-freeway-2053100/)

## Preparing the model
* **Clone this repository**
    * Note: This step is mandatory if you wish to work with EfficientDet versions different than D0.
    	Open `automl/efficientdet/inference.py` in a text editor (notepad, nano, ...)
    	* Locate function `export`
        * Locate a line containing `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]` in the function
        * Change the line to `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]`

* **Download and convert models to tflite.**
   	* **Windows**: Use setup_tflite.py, for example `python3 setup_tflite.py d0`
        * You can also specifiy quantization options, such as `python3 setup_tflite.py d0 INT8`
        * Please note that the application is designed to work with models with no optimizations or INT8 quantization. Using FP32 or FP16 quantized models requires different input data type.
    * **Linux**  : Use setup_tflite.sh, for example `./setup_tflite.sh d0`
        * You can specify quantization options, such as `./setup_tflite.sh d0 INT8`
        * Please note that the application is designed to work with models with no optimizations or INT8 quantization. Using FP32 or FP16 quantized models requires different input data type.

* A `.tflite` file will be created in `models/<model>` folder. Copy the `.tflite` file to i.MX8 board
    
## Preparing the environment

### Create Virtual Environment

1. Create virtual environment: `python -m virtualenv pythonenv`
2. Activate on Windows: `.\pythonenv\Scripts\activate` or on Linux: `source pythonenv/bin/activate`
3. Install required modules: `pip install -r requirements.txt`

*Note: To create the tflite FP32 model on Windows, TF 2.4.0 must be used. For the quantized model TF 2.5.0 works.*

### Tensorflow  
  
Additional libraries need to be built. More specifically, you need to compile `libtensorflow-lite.so` shared library and install OpenCV.
For Tensorflow shared library, you will need to install the following:
* **Tensorflow github repository** : https://github.com/tensorflow/tensorflow.git
* **Bazel build tool**
    * Windows : https://docs.bazel.build/versions/main/install-windows.html
    * Linux   : https://docs.bazel.build/versions/main/install-ubuntu.html
* **Flatbuffers**                  : Unofficial guide can be found for example here https://stackoverflow.com/questions/55394537/how-to-install-flatc-and-flatbuffers-on-linux-ubuntu

After you set up all necessities, go to tensorflow git directory, where you can run `bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so` to build C++ shared library.
For more information, refer to https://www.tensorflow.org/lite/guide/build_arm#c_library. 

You can also choose to build Tensorflow using CMake. Building with CMake does not require bazel build tool, and will produce static archive `libtensorflow-lite.a`.
In the case you decide to build Tensorflow with CMake, please follow these steps:

* From the root directory of Tensorflow github, go to `tensorflow/lite/tools/make` and run `download_dependencies.sh`.
* Go back to Tensorflow root directory, and run `./tensorflow/lite/tools/make/build_aarch64_lib.sh`
* If everything runs successfully, you should find `libtensorflow-lite.a` file in `<tensorflow_root>/tensorflow/lite/tools/make/gen/<ARCH>/lib`

### OpenCV
You need to install OpenCV to successfully crosscompile EfficientDet demo.
 * For Linux users, refer to https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html
 * You do not need to build with opencv-contrib
 * After installation, run `export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib`, to ensure linker can find newly built files

## Running the application
* Proceed to `efficientdet/src` directory. Edit the Makefile's `INC` variable, so that it points to your `tensorflow` and `flatbuffers/include` directories. Edit also `EXT` variable accordingly.
* run `make efficientdet` in the `src` directory. This should produce `efficientdet_demo` ELF binary file. Copy this binary to i.MX8 board.
* Access the board and execute the binary as `./efficientdet_demo -m <efficientdet_model_file> -i <input_video_file>`
	* For example `./efficientdet_demo -m efficientdet-lite0.tflite -i myvideo.mp4`
    * If you wish to select a different backend than CPU, provide also an optional `-b` argument from `["CPU", "NNAPI", "VX"]`
    * If you want to use VX delegate, you also need to provide the path to the delegate via `-d` argument.
    * Running the application with VX delegate is therefore done by executing the following: 
    * `./efficientdet_demo -m <efficientdet_model_file> -i <input_video_file> -b VX -d <path_to_vx_delegate>`
* After the application is done, you should find `out.avi` file in the current directory

## Licenses

Repository contains a sample video to make running the sample application easier.

| File       | Source     | License     |
| ----------- | ---------- | ----------- |
| cars_short.mp4 | https://www.pexels.com/video/different-kinds-of-vehicles-on-the-freeway-2053100/ | Pexels License (Free to use)

Additionally there are efficientdet-d0.tflite and efficientdet-d0-int8.tflite pretrained models available which were created using setup_tflite.py/sh scripts available for the user's convenience. They are trained on the COCO dataset (https://cocodataset.org/) which is licensed under Creative Commons Attribution 4.0 License (https://creativecommons.org/licenses/by/4.0/).
