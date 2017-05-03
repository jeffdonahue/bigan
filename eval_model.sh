#!/usr/bin/env bash

set -e
set -x

GPU=0
CWD=$(pwd)
CAFFE_DIR=/raid/jdonahue/philkr-caffe
MAGIC_DIR=~/magic_init
CLASS_DIR=/raid/jdonahue/voc-classification
H5DATA_DIR=${CLASS_DIR}/output

# randomly initialize the fully connected layers (starting from fc6)
REINIT_LAYER=fc6

# use
#     TRAIN_FROM=conv1 ./eval_model.sh
# or
#     TRAIN_FROM=fc8_cls ./eval_model.sh
# to run the conv1 or fc8 experiments.
TRAIN_FROM="${TRAIN_FROM:-fc6}"

# MODEL=100_encode_params
# MODEL_DIR=./exp/imagenet_1000_size72_u-200_bigan/models
MODEL=100_encode_params
MODEL_DIR=./exp/imagenet_1000_size128_resize64_u-200_bigan/models
MODEL_DEF=${CWD}/caffemodels/alexnet_train_nolrn_lrelu.prototxt
REINIT_CAFFEMODEL_PATH=${MODEL_DIR}/${MODEL}.kmeans_reinit_from_${REINIT_LAYER}.zero_fc8.cs.caffemodel
RESIZE=
# RESIZE=256

MODEL_PATH=${MODEL_DIR}/${MODEL}.jl
CAFFEMODEL_PATH=${MODEL_DIR}/${MODEL}.caffemodel

MODEL_PATH=$(realpath ${MODEL_PATH})
CAFFEMODEL_PATH=$(realpath ${CAFFEMODEL_PATH})
REINIT_CAFFEMODEL_PATH=$(realpath ${REINIT_CAFFEMODEL_PATH})

export GLOG_minloglevel=3
export PYTHONPATH=$CAFFE_DIR/python
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/home/jdonahue/cudnn3/cuda/lib64:/opt/intel/lib/intel64:/opt/intel/mkl/lib:/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/lib:/home/jdonahue/anaconda/lib:/home/jdonahue/cudnn4/cuda/lib64

if [ ! -e "${CAFFEMODEL_PATH}" ]; then
    cd ${CAFFE_DIR}
    python ${CWD}/export_params.py ${MODEL_DEF} ${MODEL_PATH} ${CAFFEMODEL_PATH}
fi

if [ ! -e "${REINIT_CAFFEMODEL_PATH}" ]; then
    cd ${CAFFE_DIR}
    python ${MAGIC_DIR}/magic_init.py -l ${CAFFEMODEL_PATH} --zero_from ${REINIT_LAYER} --post_zero_from fc8 -cs --gpu ${GPU} -t kmeans ${MODEL_DEF} ${REINIT_CAFFEMODEL_PATH}
fi

OUTPUT_DIR=$(mktemp -d)
cd ${OUTPUT_DIR}
ln -s ${H5DATA_DIR}/*.hf5 .

cd ${CLASS_DIR}
MEAN=127.5,127.5,127.5
SCALE=0.0078431368
if [ -z "$RESIZE" ]; then
    RESIZE_ARG=""
else
    RESIZE_ARG="--resize ${RESIZE}"
fi
MEAN_ARG="--mean_value ${MEAN}"
SCALE_ARG="--scale ${SCALE}"
python src/train_cls.py ${MODEL_DEF} ${REINIT_CAFFEMODEL_PATH} ${MEAN_ARG} ${SCALE_ARG} --output-dir ${OUTPUT_DIR} --train-from ${TRAIN_FROM} --gpu ${GPU} ${RESIZE_ARG}

cd ${CWD}
