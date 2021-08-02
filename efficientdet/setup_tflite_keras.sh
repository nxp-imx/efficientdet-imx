# Download model as described in 3)
# https://github.com/google/automl/tree/master/efficientdet

# Script expects argument which version of efficientDet to download (ie "d0") and an optimization string, ie "INT8", "FP16", etc...


cd checkpoints

# Download models
# Standard and lite models have slightly different download paths

if [[ ${1} == *"lite"* ]]; then
	wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-${1}.tgz
	tar xf efficientdet-${1}.tgz
	rm efficientdet-${1}.tgz
else
	wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-${1}.tar.gz
	tar xf efficientdet-${1}.tar.gz
	rm efficientdet-${1}.tar.gz
fi

cd ..

# Script requires clean checkpoint, otherwise will fail
if [[ -d checkpoints/efficientdet-${1}/saved_model ]]; then
	rm -r checkpoints/efficientdet-${1}/saved_model
fi

if [[ ! -d models/efficientdet-${1} ]]; then
	mkdir models/efficientdet-${1}
fi

if [[ ${2} == "INT8" ]]; then
	SUFFIX=int8
fi

if [[ ${2} == "FP16" ]]; then
	SUFFIX=fp16
fi

if [[ ${2} == "FP32" ]]; then
	SUFFIX=fp32
fi

cd ../automl/efficientdet

python3 -m keras.inspector --mode=export --model_name=efficientdet-${1} \
  	  --saved_model_dir=../../efficientdet/checkpoints/efficientdet-${1}/saved_model \
  	  --tflite=${2}

cp ../../efficientdet/checkpoints/efficientdet-${1}/saved_model/${SUFFIX}.tflite ../../efficientdet/models/efficientdet-${1}/efficientdet-${1}-${SUFFIX}.tflite
