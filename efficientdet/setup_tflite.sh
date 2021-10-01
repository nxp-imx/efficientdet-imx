#!/bin/bash
# Script expects argument which version of efficientDet to download (ie "d0") and an optimization string, ie "INT8", "FP16", etc...

# Update submodule automl containing EfficientDet
cd ..
git submodule update --init --recursive

cd efficientdet/checkpoints

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

cd ../automl/efficientdet

TFLITE=${2}

if [[ ${2} == "INT8" ]]; then
	SUFFIX=int8
fi

if [[ ${2} == "FP16" ]]; then
	SUFFIX=fp16
fi

if [[ ${2} == "FP32" ]]; then
	SUFFIX=fp32
fi

if [[ -z ${2} ]]; then
	# No optimization selected, use default conversion
	python3 model_inspect.py --runmode=saved_model --model_name=efficientdet-${1} \
	  --ckpt_path=../../efficientdet/checkpoints/efficientdet-${1} --saved_model_dir=../../efficientdet/checkpoints/efficientdet-${1}/saved_model \
	  --min_score_thresh=0.0 \
	  --tflite_path=../../efficientdet/models/efficientdet-${1}/efficientdet-${1}.tflite

	echo "You can find the converted model in 'models' directory."
	echo "Done"

	exit 0
fi

# For conversion with quantization optimization, representative dataset in the
# format of tfrecords is necessary
if [[ ${TFLITE} == "INT8" && ! -d "tfrecord" ]]; then
	echo "'tfrecord' folder, which is necessary for EfficientDet INT8 quantization, was not found in EfficientDet repository."
	echo "Creating 'tfrecord' folder manually now ...\n"

	if [[ ! -d "annotations" || ! -d "val2017" ]]; then
		echo "Downloading the dataset ..."

		wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
		wget http://images.cocodataset.org/zips/val2017.zip

		unzip -q annotations_trainval2017.zip
		unzip -q val2017.zip
	fi

	python3 -m dataset.create_coco_tfrecord \
	 --image_dir=val2017 \
	 --caption_annotations_file=annotations/captions_val2017.json \
	 --output_file_prefix=tfrecord/val
fi

python3 -m tf2.inspector \
 --mode=export \
 --file_pattern=tfrecord/*.tfrecord \
 --model_name=efficientdet-${1} \
 --model_dir=../../efficientdet/checkpoints/efficientdet-${1} \
 --num_calibration_steps=1000 \
 --saved_model_dir=../../efficientdet/checkpoints/efficientdet-${1}/saved_model \
 --tflite=${TFLITE}

cp ../../efficientdet/checkpoints/efficientdet-${1}/saved_model/${SUFFIX}.tflite ../../efficientdet/models/efficientdet-${1}/efficientdet-${1}-${SUFFIX}.tflite

echo "You can find the converted model in 'models' directory."
echo "Done"
