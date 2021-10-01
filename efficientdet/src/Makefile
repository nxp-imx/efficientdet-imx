INC=-I/home/fapannen/Desktop/git/tensorflow \
	-I/home/fapannen/Desktop/git/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include

LIBS=-ltensorflow-lite \
	-lstdc++ \
	-lpthread \
	-lrt \
	-ldl \
	-lm \
	-lz \
	-lopencv_stitching \
	-lopencv_calib3d \
	-lopencv_features2d \
	-lopencv_highgui \
	-lopencv_photo \
	-lopencv_imgcodecs \
	-lopencv_imgproc \
	-lopencv_flann \
	-lopencv_core \
	-lopencv_videostab \
	-lopencv_videoio \
	-lopencv_video

BIN=efficientdet_demo

GPU_BIN=efficientdet_demo_gpu

EXT=../../../tensorflow/tensorflow/lite/nnapi/nnapi_implementation.cc

LDOPTS:=-L./libs

UTILS=efficientdet_utils

all: efficientdet

efficientdet: $(BIN).cpp $(UTILS).cpp $(UTILS).hpp
	$(CXX) -std=c++17 -O2 $(INC) $(UTILS).cpp $(BIN).cpp $(LDOPTS) $(LIBS) -o $(BIN)

efficientdet-gpu: $(BIN).cpp $(UTILS).cpp $(UTILS).hpp
	$(CXX) -DEFFICIENTDET_NNAPI=1 -std=c++17 -O2 $(INC) $(UTILS).cpp $(BIN).cpp $(LDOPTS) $(LIBS) -o  $(GPU_BIN)
clean:
	rm efficientdet_demo
	rm efficientdet_demo_gpu
