#!/bin/bash

M=$1
model=$2
K=$3
echo test_linear: M$M $model K$K

dir="logs/small/test_linear/M${M}/${model}/K$K"


if [[ $model == "deit" ]]; then
    trunk_cfg=deits
else
    trunk_cfg=vits
fi
head_cfg=mlp_384_1000
ckpt=small/${model}.pth

python tools/run_distributed_engines.py \
    config=compvits/base \
    +config/compvits/model/trunk=$trunk_cfg \
    +config/compvits/model/head=$head_cfg \
    +config/compvits/data/test=in1k \
    +config/compvits/task=test_linear \
    config.CHECKPOINT.DIR=$dir \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/${ckpt} \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=model \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.COMP.NAME=afterK \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.COMP.PARAMS.K=$K \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.SPLIT.NAME=precomputed_masks \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.SPLIT.PARAMS.M=$M \
    