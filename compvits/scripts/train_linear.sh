#!/bin/bash

model=$1
echo train_linear: $model

dir="logs/train_linear/${model}"

if [[ $model == "deitb" ]]; then
    head_cfg=mlp_768_1000
    trunk_cfg=deitb
else
    head_cfg=mlp_emlp_768_1000
    trunk_cfg=vitb
fi

python tools/run_distributed_engines.py \
    config=compvits/base \
    +config/compvits/model/trunk=$trunk_cfg \
    +config/compvits/model/head=$head_cfg \
    +config/compvits/data/train=in1k \
    +config/compvits/data/test=in1k \
    +config/compvits/task=train_linear \
    config.CHECKPOINT.DIR=$dir \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/trunk_only/${model}.pth \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=model \
