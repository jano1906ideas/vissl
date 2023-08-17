#!/bin/bash

cm=cm_98_98_196_0
model=supervised
K=0

#cm=$1
#model=$2
#K=$3
dir=$4
echo test_linear: $cm $model $K

if [[ $dir == "" ]]; then
    dir="debug/test_linear/${cm}/${model}/K$K"
fi


python tools/run_distributed_engines.py \
    config=compvits/deitb_trunk_head \
    +config/compvits/data/test=in1k \
    +config/compvits/task=test_linear \
    config.TEST_ONLY=True \
    config.CHECKPOINT.DIR=$dir \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/supervised/vitb.pth \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=model \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.COMP.NAME="afterK" \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.COMP.PARAMS.K=$K \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.SPLIT.NAME="precomputed_masks" \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.SPLIT.PARAMS.M=2 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    