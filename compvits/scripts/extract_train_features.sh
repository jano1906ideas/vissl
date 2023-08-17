#!/bin/bash

model=$1

echo extract_train_features: $model

dir="logs/nearest_neighbor/train_features/${model}"

python tools/run_distributed_engines.py \
    config=compvits/base \
    +config/compvits/model/trunk=${model} \
    +config/compvits/data/train=in1k \
    engine_name=extract_features \
    config.CHECKPOINT.DIR=$dir \
    config.TEST_MODEL=False \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/${model}.pth \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=model \
