#!/usr/bin/env bash

set -e

# 50D bernoulli; "sigmoid straight-through" estimator
#
# NOISE="bss-50"
# ./train_mnist_bigan.sh --noise ${NOISE} --exp_dir ./exp/perminv_mnist_${NOISE}_bigan
#  100) JD: 0.3113  E: 2.0083  G: 2.0083
# NND/100: 5.91  NND/10: 5.06  NND: 4.56  NNC_e: 88.80%  NNC_e-: 96.46%  CLS_e-: 92.44%  EGr: 6.70  EGr_b: 10.17  EGg: 6.27  EGg_b: 10.18


# 5D uniform + 10-way categorical; categorical learned via REINFORCE
# NOISE="u-5_c-1-10"
# ./train_mnist_bigan.sh --noise ${NOISE} --exp_dir ./exp/perminv_mnist_${NOISE}_bigan
#    0) JD: 0.6932  E: 39.2006  G: 39.2006
# NND/100: 13.56  NND/10: 13.53  NND: 13.52  NNC_e: 34.91%  NNC_e-: 96.81%  CLS_e-: 91.30%  EGr: 13.69  EGr_b: 13.69  EGg: 3.16  EGg_b: 3.15
#    1) JD: 0.4836  E: 55.0668  G: 55.0668
# NND/100: 6.23  NND/10: 6.05  NND: 5.96  NNC_e: 70.02%  NNC_e-: 96.00%  CLS_e-: 89.16%  EGr: 7.46  EGr_b: 9.02  EGg: 3.80  EGg_b: 6.17
#   25) JD: nan  E: nan  G: nan
# NND/100: nan  NND/10: nan  NND: nan  NNC_e: 10.16%  NNC_e-: 10.16%  CLS_e-: 9.71%  EGr: nan  EGr_b: nan  EGg: nan  EGg_b: nan


# 5D uniform + 10-way categorical; categorical learned via REINFORCE with a constant ln(2) baseline
# NOISE="u-5_c-1-10"
# ./train_mnist_bigan.sh --noise ${NOISE} --exp_dir ./exp/perminv_mnist_${NOISE}_bigan --baseline_type 0.69314718056
#    0) JD: 0.6932  E: 0.6932  G: 0.6932
# NND/100: 13.56  NND/10: 13.53  NND: 13.52  NNC_e: 34.91%  NNC_e-: 96.81%  CLS_e-: 91.30%  EGr: 13.69  EGr_b: 13.69  EGg: 3.16  EGg_b: 3.15
#    1) JD: 0.3998  E: 37.0051  G: 37.0051
# NND/100: 6.19  NND/10: 6.04  NND: 5.98  NNC_e: 46.89%  NNC_e-: 92.13%  CLS_e-: 82.37%  EGr: 8.33  EGr_b: 9.00  EGg: 5.84  EGg_b: 6.97
#   25) JD: 0.2745  E: 74.4401  G: 74.4401
# NND/100: 5.82  NND/10: 5.23  NND: 4.91  NNC_e: 67.87%  NNC_e-: 88.85%  CLS_e-: 76.73%  EGr: 7.40  EGr_b: 9.99  EGg: 6.19  EGg_b: 9.72


# 5D uniform + 10-way categorical; categorical learned via "softmax straight-through" estimator
# NOISE="u-5_css-1-10"
# ./train_mnist_bigan.sh --noise ${NOISE} --exp_dir ./exp/perminv_mnist_${NOISE}_bigan
#    0) JD: 0.6932  E: 0.6932  G: 0.6932
# NND/100: 13.56  NND/10: 13.53  NND: 13.52  NNC_e: 34.91%  NNC_e-: 96.81%  CLS_e-: 91.30%  EGr: 13.69  EGr_b: 13.69  EGg: 3.16  EGg_b: 3.15
#    1) JD: 0.4782  E: 1.0622  G: 1.0622
# NND/100: 6.07  NND/10: 5.90  NND: 5.83  NNC_e: 65.32%  NNC_e-: 90.14%  CLS_e-: 85.12%  EGr: 7.65  EGr_b: 9.16  EGg: 4.67  EGg_b: 7.04
#   25) JD: 0.3931  E: 1.6151  G: 1.6151
# NND/100: 5.95  NND/10: 5.54  NND: 5.33  NNC_e: 71.63%  NNC_e-: 88.51%  CLS_e-: 85.61%  EGr: 7.23  EGr_b: 10.20  EGg: 5.39  EGg_b: 10.16



source theanosetup.sh

python train_gan.py --encode \
    --dataset mnist --crop_size 28 \
	--encode_net mnist_mlp \
	--discrim_net mnist_mlp \
	--gen_net deconvnet_mnist_mlp \
	--disp_interval 25 \
	--megabatch_gb 0.5 \
	--encode_normalize \
	--classifier --classifier_deploy \
	--nolog_gain --nogain --nobias --no_decay_gain \
	--encode_gen_weight 1 --encode_weight 0 --discrim_weight 0 --joint_discrim_weight 1 \
	--deploy_iters 1000 \
	--disp_samples 400 --max_outcomes 100 --max_labels 100 \
	--epochs 1000 --decay_epochs 1000 \
	--optimizer adam \
	$TAIL \
	$@ # any args passed to this script
