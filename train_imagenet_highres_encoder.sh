#!/usr/bin/env bash

NOISE="u-200"  # feature/noise (z) distribution is a 200-D uniform

# BiGAN objective
OBJECTIVE="--encode_gen_weight 1 --encode_weight 0 --discrim_weight 0 --joint_discrim_weight 1"

./train_imagenet.sh \
    --raw_size 128 --crop_size 112 --crop_resize 64 \
    ${OBJECTIVE} \
    $@
