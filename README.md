# EfficientDet

## EfficientDet example on i.MX8
	1) Clone this repository
    2) run `setup_tflite.py <efficientdet_version>`, for example `python3 setup_tflite.py d0`
    3) A `.tflite` file will be created in corresponding directory in `models` folder. Copy the `.tflite` file to i.MX8 board
    4) Proceed to `efficientdet/src` directory. Edit the Makefile's `INC` variable, so that it points to your `tensorflow` and `flatbuffers/include` directories.
    5) run `make efficientdet` in the `src` directory. This should produce `measure_efficientdet` ELF binary file. Copy this binary to i.MX8 board.
    6) Access the board and execute the binary as `./measure_efficientdet <efficientdet_model_file> <input_video_file> <model_input_size>`
    	For example `./measure_efficientdet efficientdet-lite0.tflite myvideo.mp4 320`
    7) After the application is done, you should find `out.avi` file in the current directory
