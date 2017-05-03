#!/usr/bin/env bash

# Train on ImageNet with only 10 classes, by default
# (add "--max_labels 1000" to use the full dataset).

NOISE="u-200"  # feature/noise (z) distribution is a 200-D uniform

# BiGAN objective
OBJECTIVE="--encode_gen_weight 1 --encode_weight 0 --discrim_weight 0 --joint_discrim_weight 1"

# Latent Regressor (LR) objective
# OBJECTIVE="--encode_gen_weight 0 --encode_weight 1 --discrim_weight 1 --joint_discrim_weight 0"

# Joint Latent Regressor (Joint LR) objective
# OBJECTIVE="--encode_gen_weight 0.25 --encode_weight 1 --discrim_weight 1 --joint_discrim_weight 0"

python train_gan.py \
    --encode --encode_normalize \
    --dataset imagenet --raw_size 72 --crop_size 64 \
    --gen_net_size 64 \
    --feat_net_size 64 \
    --encode_net alexnet_group_padpool \
    --megabatch_gb 0.5 \
    --classifier --classifier_deploy \
    --nolog_gain --no_decay_gain \
    --deploy_iters 1000 \
    --disp_samples 400 \
    --max_labels 10 --epochs 200 --decay_epochs 200 \
    --disp_interval 25 --save_interval 25 \
    --noise ${NOISE} \
    ${OBJECTIVE} \
    $@
