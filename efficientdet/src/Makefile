INC=-I../third-party/cxxopts

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

UTILS=efficientdet_utils

all: efficientdet

efficientdet: $(BIN).cpp $(UTILS).cpp $(UTILS).hpp
	$(CXX) -std=c++17 -O2 $(INC) $(UTILS).cpp $(BIN).cpp $(LDOPTS) $(LIBS) -o $(BIN)

clean:
	rm efficientdet_demo
