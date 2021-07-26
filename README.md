# EfficientDet

## EfficientDet example on i.MX8

### Preparing the model
	1) Clone this repository
    2) Note: This step is mandatory if you wish to work with EfficientDet versions different than d0.
    	Open "automl/efficientdet/inference.py" in a text editor (notepad, nano, ...)
    	- Locate function "export"
        - Locate a line containing "converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]" in the function
        - Change the line to "converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]"
    2) Download and convert models to tflite.
    	- Windows: Use setup_tflite.py, for example "python3 setup_tflite.py d0"
        - Linux  : Use setup_tflite.sh, for example "./setup_tflite.sh d0"
    3) A `.tflite` file will be created in `models/<model>` folder. Copy the `.tflite` file to i.MX8 board
    
### Preparing the environment
#### Tensorflow    
    Additional libraries need to be built. More specifically, you need to compile "libtensorflow.so" shared library and install OpenCV.
    For Tensorflow shared library, you will need to install the following:
    	- Tensorflow github repository : https://github.com/tensorflow/tensorflow.git
        - Bazel build tool             : Windows : https://docs.bazel.build/versions/main/install-windows.html
        							   : Linux   : https://docs.bazel.build/versions/main/install-ubuntu.html
        - Flatbuffers                  : Unofficial guide can be found for example here https://stackoverflow.com/questions/55394537/how-to-install-flatc-and-flatbuffers-on-linux-ubuntu
    
    After you set up all necessities, you can run "bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so" to build C++ shared library.
    For more information, refer to https://www.tensorflow.org/lite/guide/build_arm#c_library. 
    
    You can also choose to build Tensorflow using CMake. Building with CMake does not require bazel build tool, and will produce static archive "libtensorflow-lite.a".
    In the case you decide to build Tensorflow with CMake, please follow these steps:
    
    1) From the root directory of Tensorflow github, go to "tensorflow/lite/tools/make" and run "download_dependencies.sh".
    2) Go back to Tensorflow root directory, and run "./tensorflow/lite/tools/make/build_aarch64_lib.sh"
    3) If everything runs successfully, you should find "libtensorflow-lite.a" file in "<tensorflow_root>/tensorflow/lite/tools/make/gen/<ARCH>/lib"
    
#### OpenCV
	You need to install OpenCV to successfully crosscompile EfficientDet demo.
     - For Linux users, refer to https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html
	 - You do not need to build with opencv-contrib
     - After installation, run "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib", to ensure linker can find newly built files

### Running the application
	1) Proceed to `efficientdet/src` directory. Edit the Makefile's `INC` variable, so that it points to your `tensorflow` and `flatbuffers/include` directories. Do the same also for EXT variable.
    2) run `make efficientdet` in the `src` directory. This should produce `measure_efficientdet` ELF binary file. Copy this binary to i.MX8 board.
	3) Access the board and execute the binary as `./measure_efficientdet <efficientdet_model_file> <input_video_file> <model_input_size>`
    	For example `./measure_efficientdet efficientdet-lite0.tflite myvideo.mp4 320`
    4) After the application is done, you should find `out.avi` file in the current directory
    
