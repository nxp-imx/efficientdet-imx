# Copyright 2022 NXP
# SPDX-License-Identifier: Apache-2.0

import requests
import sys
import tarfile
import zipfile
import os
import shutil
import argparse

python_exec="python"

parser = argparse.ArgumentParser(description='EfficientDet preparation script')

parser.add_argument('--model', help='EfficientDet version (ie. d0, d2, lite0, lite2, ...)', required=True)
parser.add_argument('--max_detections', help='Maximum number of detections', default=100, required=False)
parser.add_argument('--quant', help='Quantization option (INT8, FP16, FP32)', required=False)

args = vars(parser.parse_args())

model 		   = args["model"]
opt   		   = args["quant"]
max_detections = args["max_detections"]

# Update submodule
os.system("git submodule update --init --recursive")

# Download models
# Standard and lite models have slightly different download paths

if "lite" in model:
	req = requests.get("https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-{model}.tgz".format(model=model))

else:
	req = requests.get("https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-{model}.tar.gz".format(model=model))

with open ("checkpoints/tmp.tar.gz", "wb") as w:
	w.write(req.content)
tar = tarfile.open("checkpoints/tmp.tar.gz", "r:gz")
tar.extractall(path="checkpoints")
tar.close()
os.remove("checkpoints/tmp.tar.gz")

# Script requires clean checkpoint, otherwise will fail

if os.path.isdir("checkpoints/efficientdet-{model}/saved_model".format(model=model)):
	shutil.rmtree("checkpoints/efficientdet-{model}/saved_model".format(model=model))

if not os.path.isdir("models/efficientdet-{model}".format(model=model)):
	os.mkdir("models/efficientdet-{model}".format(model=model))

if opt is not None:

	os.chdir("../automl/efficientdet")

	if opt == "INT8" and not os.path.isdir("tfrecord"):
		print("'tfrecord' folder, which is necessary for EfficientDet INT8 quantization, was not found in EfficientDet repository.")
		print("Creating 'tfrecord' folder manually now ...")
		print()

		if not os.path.exists("annotations") or not os.path.exists("val2017"):
			print("Downloading the dataset ...")
			# Download required dataset
			coco_annotations = requests.get("http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
			coco_val2017     = requests.get("http://images.cocodataset.org/zips/val2017.zip")

			with open("annotations_trainval2017.zip", "wb") as ann:
				ann.write(coco_annotations.content)

			with open("val2017.zip", "wb") as val:
				val.write(coco_val2017.content)

			with zipfile.ZipFile("annotations_trainval2017.zip", "r") as unzipAnn:
				unzipAnn.extractall()

			with zipfile.ZipFile("val2017.zip", "r") as unzipVal:
				unzipVal.extractall()

		os.system("{python} -m dataset.create_coco_tfrecord \
	 		--image_dir=val2017 \
	 		--caption_annotations_file=annotations/captions_val2017.json \
	 		--output_file_prefix=tfrecord/val".format(python=python_exec))


	os.system("{python} -m tf2.inspector \
 		--mode=export \
		--file_pattern=tfrecord/*.tfrecord \
		--model_name=efficientdet-{model} \
		--model_dir=../../efficientdet/checkpoints/efficientdet-{model} \
		--num_calibration_steps=1000 \
		--hparams=\"tflite_max_detections={tflite_max_detections}\" \
		--saved_model_dir=../../efficientdet/checkpoints/efficientdet-{model}/saved_model \
		--tflite={tflite}".format(python=python_exec, model=model, tflite=opt, tflite_max_detections=max_detections))

	os.chdir("../../efficientdet")

	shutil.copyfile("checkpoints/efficientdet-{model}/saved_model/{opt}.tflite".format(
		model=model, opt=opt.lower()), 
	"models/efficientdet-{model}/efficientdet-{model}-{opt}.tflite".format(
		model=model, opt=opt.lower()))

else:
	# Convert the model to tflite using original EfficientDet script
	os.system("{python} ../automl/efficientdet/model_inspect.py \
		--runmode=saved_model \
		--model_name=efficientdet-{model} \
		--ckpt_path=checkpoints/efficientdet-{model} \
		--max_boxes_to_draw={max_boxes_to_draw} \
		--saved_model_dir=checkpoints/efficientdet-{model}/saved_model \
		--min_score_thresh=0.0 \
		--tflite_path=models/efficientdet-{model}/efficientdet-{model}.tflite".format(
			python=python_exec, model=model, max_boxes_to_draw=max_detections))

print("You can find the converted model in 'models' directory.")
print("Done")
