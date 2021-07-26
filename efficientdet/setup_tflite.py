# Download model as described in 3)
# https://github.com/google/automl/tree/master/efficientdet

# Script expects argument which version of efficientDet to download (ie "d0")

import requests
import sys
import tarfile
import os
import shutil

model = sys.argv[1]

# Download models
# Standard and lite models have slightly different download paths

if "lite" in model:
	req = requests.get("https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-" + model + ".tgz")

else:
	req = requests.get("https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-" + model + ".tar.gz")

with open ("checkpoints/tmp.tar.gz", "wb") as w:
	w.write(req.content)
tar = tarfile.open("checkpoints/tmp.tar.gz", "r:gz")
tar.extractall(path="checkpoints")
tar.close()
os.remove("checkpoints/tmp.tar.gz")

# Script requires clean checkpoint, otherwise will fail

if os.path.isdir("checkpoints/efficientdet-" + model + "/saved_model"):
	shutil.rmtree("checkpoints/efficientdet-" + model + "/saved_model")

if not os.path.isdir("models/efficientdet-" + model):
	os.mkdir("models/efficientdet-" + model)

# Convert the model to tflite using original EfficientDet script
os.system("python3 ../automl/efficientdet/model_inspect.py --runmode=saved_model --model_name=efficientdet-" + model + " --ckpt_path=checkpoints/efficientdet-" + model + " --saved_model_dir=checkpoints/efficientdet-" + model + "/saved_model --min_score_thresh=0.0 --tflite_path=models/efficientdet-" + model + "/efficientdet-" + model + ".tflite")

