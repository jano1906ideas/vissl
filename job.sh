#!/bin/bash

source compvits/scripts/extract_train_features.sh ibot
source compvits/scripts/extract_val_features.sh ibot

models=(dino ibot moco3)
for model in ${models[@]}; do
    mkdir --parents logs/small/linear_head_training/${model}
    python compvits/scripts/train_linear_head.py --train_data_path logs/small/nearest_neighbor/train_features/${model} --val_data_path logs/small/nearest_neighbor/test_features/${model} --embed_dim 384 --output_dir logs/small/linear_head_training/${model}
done
