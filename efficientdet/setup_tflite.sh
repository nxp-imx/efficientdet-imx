#!/bin/bash

# Copyright 2021 NXP

# Script expects argument which version of efficientDet to download (ie "d0") and an optimization string, ie "INT8", "FP16", etc...
# You can also provide a third argument specifying how many outputs there should be. By default, EfficientDet uses 100.

while getopts ":m:q:d:" arg; do
    case "${arg}" in
        m)
            MODEL=${OPTARG}
            ;;
        q)
            OPT=${OPTARG}
            ;;
        d)
            MAX_DETECTIONS=${OPTARG}
            ;;
        *)
			echo "Script options:"
			echo "-m Model version (d0, d2, lite0, lite2, ...)"
			echo "-q Quantization options (INT8, FP32, FP16)"
			echo "-d Maximum number of outputs (default 100)"
			exit -1
			;;
    esac
done

if [[ -z ${MODEL} ]]; then
	echo "Please provide model version (-m option), ie d0, d2, ..."
	echo "Example: ./setup_tflite.sh -m d0"
	exit -1
fi

if [[ -z ${MAX_DETECTIONS} ]]; then
	MAX_DETECTIONS=100
fi

# Update submodule automl containing EfficientDet
cd ..
git submodule update --init --recursive

cd efficientdet/checkpoints

# Download models
# Standard and lite models have slightly different download paths

if [[ ${MODEL} == *"lite"* ]]; then
	wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-${MODEL}.tgz
	tar xf efficientdet-${MODEL}.tgz
	rm efficientdet-${MODEL}.tgz
else
	wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-${MODEL}.tar.gz
	tar xf efficientdet-${MODEL}.tar.gz
	rm efficientdet-${MODEL}.tar.gz
fi

cd ..

# Script requires clean checkpoint, otherwise will fail
if [[ -d checkpoints/efficientdet-${MODEL}/saved_model ]]; then
	rm -r checkpoints/efficientdet-${MODEL}/saved_model
fi

if [[ ! -d models/efficientdet-${MODEL} ]]; then
	mkdir models/efficientdet-${MODEL}
fi

cd ../automl/efficientdet

if [[ ${OPT} == "INT8" ]]; then
	SUFFIX=int8
fi

if [[ ${OPT} == "FP16" ]]; then
	SUFFIX=fp16
fi

if [[ ${OPT} == "FP32" ]]; then
	SUFFIX=fp32
fi

if [[ -z ${OPT} ]]; then
	# No optimization selected, use default conversion
	python3 model_inspect.py --runmode=saved_model --model_name=efficientdet-${MODEL} \
	  --ckpt_path=../../efficientdet/checkpoints/efficientdet-${MODEL} --saved_model_dir=../../efficientdet/checkpoints/efficientdet-${MODEL}/saved_model \
	  --min_score_thresh=0.0 \
	  --max_boxes_to_draw=${MAX_DETECTIONS} \
	  --tflite_path=../../efficientdet/models/efficientdet-${MODEL}/efficientdet-${MODEL}.tflite

	echo "You can find the converted model in 'models' directory."
	echo "Done"

	exit 0
fi

# For conversion with quantization optimization, representative dataset in the
# format of tfrecords is necessary
if [[ ${OPT} == "INT8" && ! -d "tfrecord" ]]; then
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
 --model_name=efficientdet-${MODEL} \
 --model_dir=../../efficientdet/checkpoints/efficientdet-${MODEL} \
 --num_calibration_steps=1000 \
 --saved_model_dir=../../efficientdet/checkpoints/efficientdet-${MODEL}/saved_model \
 --hparams="tflite_max_detections=${MAX_DETECTIONS}" \
 --tflite=${OPT}

cp ../../efficientdet/checkpoints/efficientdet-${MODEL}/saved_model/${SUFFIX}.tflite ../../efficientdet/models/efficientdet-${MODEL}/efficientdet-${MODEL}-${SUFFIX}.tflite

echo "You can find the converted model in 'models' directory."
echo "Done"
