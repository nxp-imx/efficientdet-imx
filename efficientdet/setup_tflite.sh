# Download model as described in 3)
# https://github.com/google/automl/tree/master/efficientdet

# Script expects argument which version of efficientDet to download (ie "d0")

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

python3 ../automl/efficientdet/model_inspect.py --runmode=saved_model --model_name=efficientdet-${1} \
  --ckpt_path=checkpoints/efficientdet-${1} --saved_model_dir=checkpoints/efficientdet-${1}/saved_model \
  --min_score_thresh=0.0 \
  --tflite_path=models/efficientdet-${1}/efficientdet-${1}.tflite
