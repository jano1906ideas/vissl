#!/bin/bash

model=$1

echo extract_train_features: $model

dir="logs/small/nearest_neighbor/train_features/${model}"

if [[ $model == "deit" ]]; then
    trunk_cfg=deits
    ckpt=small/deit.pth
else
    trunk_cfg=vits
    ckpt=trunk_only/small/${model}.pth
fi

python tools/run_distributed_engines.py \
    config=compvits/base \
    +config/compvits/model/trunk=$trunk_cfg \
    +config/compvits/data/train=in1k \
    engine_name=extract_features \
    config.TEST_MODEL=False \
    config.CHECKPOINT.DIR=$dir \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/${ckpt} \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=model \
