import sys

from collections import OrderedDict
import copy
import itertools
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.externals import joblib
from time import time

import theano
import theano.tensor as T

import data
from data import rescale

import lib.rng as rng
from lib.lazy_function import LazyFunction as lazy_function

import argparse
parser = argparse.ArgumentParser(description='Train (Bi)GANs')

# RNG
parser.add_argument('--seed', type=int, help='RNG seed')

# Objective (which modules to train? what losses to minimize?)
parser.add_argument('--discrim_weight', type=float, default=1,
    help='Weighting of discrim loss')
parser.add_argument('--joint_discrim_weight', type=float, default=0,
    help='Train joint (BiGAN) discriminator (x, z) with this weight')
parser.add_argument('--encode', action='store_true', help='Learn encoder')
parser.add_argument('--no-encode', action='store_false', dest='encode',
    help='No encoder')
parser.add_argument('--encode_weight', type=float, default=1,
    help='Encoder weighting of actual encode loss')
parser.add_argument('--encode_normalize', action='store_true',
    help='Normalize encoder costs by distribution dimension')
parser.add_argument('--encode_gen_weight', type=float, default=0,
    help='Weight of encoder loss in generator optimization')
parser.add_argument('--encode_kldiv_real', type=float, default=0,
    help='If non-zero, train encoder on real images, with this KL divergence '
         'loss weight')
parser.add_argument('--noise_input_joint_discrim',
    help='Comma-separated list of binary values (0/1) indicating whether each '
         'part of the noise is input to the joint discriminator')

# Noise/latent distribution (Z) settings
parser.add_argument('--noise', default='u-100',
    help='Specifies distribution of noise input to generation net')
parser.add_argument('--noise_weight', type=str,
    help='Comma-separated floats specifying a weights for each noise subdist')
parser.add_argument('--noise_input_weight', action='store_true',
    help='Whether to weight the noise distribution components in the INPUT')

# Adversarial optimization
parser.add_argument('--k', type=float, default=1,
    help='Number of discrim updates per gen update (may be fractional do to '
         'multiple gen updates per discrim update')
parser.add_argument('--no_update_both', action='store_true',
    help=("Don't update both nets together, even if k==1 "
          "(nets always updated separately if k!=1)"))

# Architecture settings
# General
parser.add_argument('--net_fc', type=int, default=0,
    help='Number of FC layers of 2x dimension before/after deconv/conv net')
parser.add_argument('--net_fc_dims', default='',
    help='Comma separated list of integer FC layer sizes (if want different '
         'sizes than given by --net_fc)')
parser.add_argument('--nobn', action='store_true', help='No batch norm')
parser.add_argument('--bn_separate', action='store_true',
    help='Apply batch norm to x, z separately')
parser.add_argument('--nogain', action='store_true', help='No gain')
parser.add_argument('--log_gain', action='store_true', help='Learn log of gain')
parser.add_argument('--nolog_gain', action='store_false', dest="log_gain")
parser.add_argument('--nobias', action='store_true', help='No bias')
parser.add_argument('--no_decay_bias', action='store_true',
    help="Don't apply weight decay to the bias(es), if any")
parser.add_argument('--no_decay_gain', action='store_true',
    help="Don't apply weight decay to the gain(s), if any")

# Deconvolutional / Generator
parser.add_argument('--gen_net',
    help='Name of network architecture for generator')
parser.add_argument('--gen_net_size', type=int,
    help='Base dimension of gen net layers (see net.py for defaults)')
parser.add_argument('--deconv_nonlin', default='ReLU',
    help='Nonlinearity for deconv nets')
parser.add_argument('--deconv_ksize', type=int, default=5,
    help='Kernel size for deconv nets')

# Convolutional (Discriminator or Encoder)
parser.add_argument('--feat_net_size', type=int,
    help='Base dimension of feat net layers (see net.py for defaults)')
parser.add_argument('--conv_nonlin', default='LReLU',
    help='Nonlinearity for conv nets')
parser.add_argument('--net_fc_drop', type=float, default=0,
    help='The dropout probability for the convnet FC layers')
parser.add_argument('--cond_fc', type=int,
    help='Number of FC layers of 2x dimension before/after deconv/conv net')
parser.add_argument('--cond_fc_dims',
    help='Comma separated list of integer FC layer sizes (if want different '
         'sizes than given by --cond_fc)')
parser.add_argument('--cond_fc_drop', type=float,
    help='The dropout probability for the cond FC layers')

# Discriminator
parser.add_argument('--discrim_net',
    help='Name of network architecture for discriminator')
parser.add_argument('--discrim_net_size', type=int,
    help='Base dimension of discrim net layers, overriding --feat_net_size')
parser.add_argument('--minibatch_layer_size',
    help='Specify minibatch layer as comma-separated list of 3 ints, '
         'e.g. 300,50,16 for OpenAI settings')
parser.add_argument('--post_minibatch_layer_dims',
    help='Comma-separated list of ints specifying FC layer dims '
         'applied after the minibatch layer')
parser.add_argument('--cat_inputs', action='store_true',
    help='Concatenate real/gen images input to discrim+encoder '
         '(affects batch norm)')

# Encoder
parser.add_argument('--encode_net',
    help='Name of network architecture for encoder')
parser.add_argument('--encode_nonlin',
    help='Nonlinearity for encoder (conv_nonlin used if not specified)')
parser.add_argument('--encode_net_fc', type=int,
    help='Number of FC layers of 2x dimension before/after encode net')
parser.add_argument('--encode_net_fc_dims', type=str,
    help='Comma separated list of integer FC layer sizes (if want different '
         'sizes than given by --encode_net_fc)')
parser.add_argument('--encode_net_fc_drop', type=float,
    help='The dropout probability for the convnet FC layers')
parser.add_argument('--log_var_bias', type=float, default=-5,
    help='Bias to apply to log variance prediction for GaussianRecon*Distribution')
parser.add_argument('--encode_out_bias', action='store_true',
    help='Encoder output predictor includes a learned bias term')

# Data
parser.add_argument('--dataset', default='mnist',
    help='Dataset (mnist/imagenet)')
parser.add_argument('--raw_size', type=int,
    help='Raw minor edge size (for image datasets)')
parser.add_argument('--crop_size', type=int, default=28,
    help='Crop size (for image datasets)')
parser.add_argument('--crop_resize', type=int,
    help='Size of crop for D in, G out (E still takes crop_size resolution)')
parser.add_argument('--include_labels', type=str,
    help='Comma-separated list of labels to include; e.g., "0,100,130,249)"')
parser.add_argument('--max_labels', type=int,
    help='If set, include only labels 0:(max_labels - 1)')
parser.add_argument('--max_images', type=int,
    help='If set, include at most this many images')
parser.add_argument('--use_test_set', action='store_true',
    help='MNIST: use the full training set (train+val) for training; test set as val')

# Black box optimization
parser.add_argument('--optimizer', default='adam',
    help='Optimization algorithm (adam, sgd, or rms)')
parser.add_argument('--discrim_optimizer',
    help='Discriminator optimization algorithm -- '
         'defaults to optimizer if unspecified')
parser.add_argument('--batch_size', type=int, default=128,
    help='Batch size used for learning')
parser.add_argument('--learning_rate', type=float, default=0.0002,
    help='Initial learning rate')
parser.add_argument('--final_lr_mult', type=float,
    help='Final learning rate factor on the initial LR'
         '(default is 0.01 for exponential decay, 0 for linear decay)')
parser.add_argument('--linear_decay', action='store_true',
    help='Decay learning rate linearly (otherwise, decay exponentially)')
parser.add_argument('--decay', type=float, default=2.5e-5,
    help='L2 weight decay')
parser.add_argument('--discrim_decay', type=float,
    help='L2 weight decay for the discriminator (defaults to --decay)')
parser.add_argument('--sgd_momentum', type=float, default=0.9,
    help='Momentum coeff for SGD')
parser.add_argument('--epochs', type=int, default=50,
    help='Number of epochs to train before beginning LR decay '
         '(doubled if k != 1)')
parser.add_argument('--decay_epochs', type=int, default=50,
    help='Number of epochs to train decay LR to 0 (doubled if k != 1)')

# Visualization / evaluation output
parser.add_argument('--exp_dir', default='exp',
    help='Directory where experiment results are stored')
parser.add_argument('--disp_interval', type=int,
    help='Compute and display test accuracy every disp_interval epochs')
parser.add_argument('--deploy_iters', type=int, default=50,
    help='Number of iterations to perform "deploy" steps (e.g. BN tuning)')
parser.add_argument('--no_disp_one', action='store_true',
    help='Also eval/disp at iter #1, unless --no_disp_one is given')
parser.add_argument('--disp_samples', type=int, default=200,
    help='Max number of samples to display')
parser.add_argument('--megabatch_gb', type=float, default=1.0,
    help='Max GB per megabatch (~16 megabatches are created)')
parser.add_argument('--megabatch_images', type=int, default=10000,
    help='Max images per megabatch (~16 megabatches are created)')
parser.add_argument('--num_val_megabatch', type=int, default=1)
parser.add_argument('--num_train_megabatch', type=int, default=5)
parser.add_argument('--num_sample_megabatch', type=int, default=10)
# Train a linear classifier (logistic regression) on labels as part of
# feature learning evaluation (in addition to kNN classifier)
parser.add_argument('--classifier', action='store_true',
    help='Learn classifier from discrim (and encoder) feats')
parser.add_argument('--classifier_deploy', action='store_true',
    help='Learn classifier at deploy time')

# Model saving / loading
parser.add_argument('--save_interval', type=int,
    help='Save params every N epochs for this N')
parser.add_argument('--resume', type=int,
    help='Epoch from which to resume training and load weights')
parser.add_argument('--weights',
    help='Weights filename prefix (e.g., * in *_discrim_params.jl)')

args = parser.parse_args()
print 'Args: %s' % args

args.net_fc_dims = [int(d) for d in args.net_fc_dims.split(',') if d]
if args.encode_net_fc_dims is not None:
    args.encode_net_fc_dims = \
        [int(d) for d in args.encode_net_fc_dims.split(',') if d]
if args.cond_fc_dims is not None:
    args.cond_fc_dims = \
        [int(d) for d in args.cond_fc_dims.split(',') if d]
if args.minibatch_layer_size is not None:
    args.minibatch_layer_size = \
        [int(d) for d in args.minibatch_layer_size.split(',') if d]
    assert len(args.minibatch_layer_size) == 3
if args.post_minibatch_layer_dims is not None:
    args.post_minibatch_layer_dims = \
        [int(d) for d in args.post_minibatch_layer_dims.split(',') if d]

if args.seed is not None:
    rng.set_seed(args.seed)
py_rng = rng.py_rng
np_rng = rng.np_rng
t_rng = rng.t_rng

from lib import updates
from lib.theano_utils import floatX, sharedX
from lib.data_utils import shuffle, list_shuffle
from lib.metrics import nnc_score, nnd_score

from net import Output, L, Net
from dist import MultiDistribution
import gan

bnkwargs = dict(bnkwargs=dict(
    batch_norm=(not args.nobn),
    gain=(not args.nogain) and (not args.nobn),
    log_gain=args.log_gain,
    bias=(not args.nobias),
))

crop = args.crop_size
if args.crop_resize is None:
    args.crop_resize = crop

dataset = data.Dataset(args, load=False)
ny, nc, inverse_transform = dataset.ny, dataset.nc, dataset.inverse_transform

def transform(X, crop=args.crop_resize):
    # X: uint8-type ndarray [0, 255] (possibly flattened)
    # returns NCHW float array in [0, 1]
    X = floatX(X).reshape(-1, nc, crop, crop)
    return rescale(X, (0, 255), dataset.native_range)
def input_transform(X):
    # X: uint8-type tensor [0, 255]
    # returns a float tensor in native_range
    # ([-1, 1] for ImageNet, [0, 1] for MNIST)
    X = T.cast(X, theano.config.floatX)
    return rescale(X, (0, 255), dataset.native_range)
def gen_transform(gX):
    # X: float tensor in [0, 1] (e.g., output by generator's sigmoid)
    # returns float tensor in native_range
    # ([-1, 1] for ImageNet, [0, 1] for MNIST)
    return rescale(gX, (0, 1), dataset.native_range)

def gen_output_to_enc_input(gX):
    gX = gX.reshape(-1, nc, crop, crop)
    gX = np.round(rescale(gX, dataset.native_range, (0, 255)))
    gX[gX < 0] = 0
    gX[gX > 255] = 255
    return np.array(gX, dtype=np.uint8)

pred_log_var = False
b1 = 0.5          # momentum term of adam
niter = args.epochs       # # of iter at starting learning rate
niter_decay = args.decay_epochs # # of iter to linearly decay learning rate
lr = args.learning_rate       # initial learning rate for adam
if args.final_lr_mult is None:
    args.final_lr_mult = 0 if args.linear_decay else 0.01
final_lr = args.final_lr_mult * lr
assert 0 <= final_lr <= lr
if not args.linear_decay:
    # exponential decay
    assert final_lr > 0, \
        'For exponential decay, final LR must be strictly positive (> 0)'
    log_lr = np.log(lr)
    log_final_lr = np.log(final_lr)

model_dir = '%s/models'%(args.exp_dir,)
samples_dir = '%s/samples'%(args.exp_dir,)
for d in [model_dir, samples_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

if args.noise_weight is not None:
    args.noise_weight = [float(f) for f in args.noise_weight.split(',')]

if args.noise_input_joint_discrim is None:
    args.noise_input_joint_discrim = [True] * len(args.noise.split('_'))
else:
    args.noise_input_joint_discrim = [bool(float(x)) for x in
                                      args.noise_input_joint_discrim.split(',')]
assert len(args.noise_input_joint_discrim) == len(args.noise.split('_'))

dist = MultiDistribution(args.batch_size, args.noise,
    normalize=args.encode_normalize,
    weights=args.noise_weight, weight_embed=args.noise_input_weight)
for d in dist.dists:
    d.log_var_bias = args.log_var_bias

# input placeholders
Xi = T.tensor4(dtype='uint8')
X = Output(input_transform(Xi), (args.batch_size, nc, args.crop_size, args.crop_size))
assert args.crop_resize <= args.crop_size
if args.crop_size == args.crop_resize:
    Xis = Xi
else:
    Xis = T.tensor4(dtype='uint8')
Xs = Output(input_transform(Xis), (args.batch_size, nc, args.crop_resize, args.crop_resize))
Z = dist.placeholders
if args.classifier:
    Y = T.ivector()
    y = Output(Y, shape=(args.batch_size, ), index_max=ny)
else:
    Y = None
    y = None

modules = []
nets = []

# generation
gen_kwargs = dict(args=args, dist=dist, nc=nc, bnkwargs=bnkwargs,
                  gen_transform=gen_transform)
train_gen = gan.Generator(**gen_kwargs)
gX = train_gen.data
gXtest = gan.Generator(source=train_gen.net, mode='test', **gen_kwargs).data

lrt = sharedX(lr)

def get_updater(optimizer, **kwargs):
    opt_map = dict(adam='Adam', sgd='Momentum', rms='RMSprop')
    if optimizer not in opt_map:
        raise ValueError('Unknown optimizer: %s' % (optimizer,))
    if optimizer == 'adam':
        kwargs.update(b1=b1)
    elif optimizer == 'sgd':
        kwargs.update(momentum=args.sgd_momentum)
    opt_func = getattr(updates, opt_map[optimizer])
    return opt_func(**kwargs)

ignored_prefixes = []
if args.no_decay_bias: ignored_prefixes.append('Bias')
if args.no_decay_gain: ignored_prefixes.append('Gain')
reg = updates.Regularizer(l2=args.decay, ignored_prefixes=ignored_prefixes)
updater = get_updater(args.optimizer, lr=lrt, regularizer=reg)

if args.discrim_decay is None:
    args.discrim_decay = args.decay
discrim_reg = updates.Regularizer(l2=args.discrim_decay,
    ignored_prefixes=ignored_prefixes)

if args.discrim_optimizer is None:
    args.discrim_optimizer = args.optimizer
discrim_updater = get_updater(args.discrim_optimizer,
    lr=lrt, regularizer=discrim_reg)

def featurizer(x=None, gx=None, args=args, **kwargs):
    return gan.Featurizer(args, dist, x, gx, y, nc=nc, ny=ny,
        bnkwargs=bnkwargs, updater=updater, **kwargs)
def discrim_featurizer(x=Xs, gx=gX, *a, **k):
    return featurizer(x=x, gx=gx, *a, **k)
gx_enc = gX if args.crop_size == args.crop_resize else None
def encode_featurizer(x=X, gx=gx_enc, *a, **k):
    return featurizer(x=x, gx=gx, *a, **k)

if args.discrim_weight:
    f_discrim = discrim_featurizer(
        discrim_weight=args.discrim_weight,
        net_name=args.discrim_net, net_size=args.discrim_net_size,
        is_discrim=True,
        name='Discriminator')
    d_cost = f_discrim.net.get_loss().mean()
    discrim_params = f_discrim.net.params()
    d_updates = f_discrim.net.get_updates(updater=discrim_updater)
    g_cost = f_discrim.net.get_loss('opp_loss_gen')
    train_gen.net.add_loss(g_cost)
    modules.append(f_discrim)
    nets.append(f_discrim.net)
else:
    f_discrim = None
    d_cost = 0
    discrim_params = []
    d_updates = []

if args.encode:
    f_encoder = encode_featurizer(
        encode_weight=args.encode_weight,
        joint_discrim_weight=args.joint_discrim_weight,
        net_name=args.encode_net,
        name='Encoder')
    modules.append(f_encoder)
    def sample_z_from_x(x, net=None):
        f_in = f_encoder.feats(x)
        z_preds = f_encoder.encoder.preds(f_in).value
        z_sample = dist.logits_to_sample(z_preds)
        is_input = dist.dist_sized_slices(args.noise_input_joint_discrim)
        return [z for z, i in zip(z_sample, is_input) if i]

def disp(x):
    if isinstance(x, (int, float)):
        return x
    return x.mean()

disp_costs = OrderedDict(D=disp(d_cost))  # name -> cost mapping

gen_params = train_gen.net.params()
nets.append(train_gen.net)

if args.joint_discrim_weight:
    assert args.encode
    def joint_discrim(X, Xs, Y, Z_dist, name='JointDiscriminator'):
        joint_discrim_args = copy.deepcopy(args)
        joint_discrim_args.classifier = 0
        eZ = sample_z_from_x(X, net=f_encoder.net)
        gX = train_gen.data
        _, weights = gen_cond_and_weights = Z_dist.embed_data()
        real_cond_and_weights = eZ, weights
        discrim = discrim_featurizer(x=Xs, gx=gX, args=joint_discrim_args,
            extra_cond_gen=gen_cond_and_weights,
            extra_cond_real=real_cond_and_weights,
            discrim_weight=1,
            net_name=args.discrim_net, net_size=args.discrim_net_size,
            is_discrim=True, name=name)
        return discrim
    f_joint_discrim = joint_discrim(X, Xs, Y, dist)
    weight = args.joint_discrim_weight
    f_encoder.net.add_loss(f_joint_discrim.net.get_loss('opp_loss_real'),
                           weight=weight, name='loss_real')
    f_encoder.net.add_loss(f_joint_discrim.net.get_loss('opp_loss_gen'),
                           weight=weight, name='loss_gen')
    modules.append(f_joint_discrim)
    nets.append(f_joint_discrim.net)
    d_updates += f_joint_discrim.net.get_updates(updater=discrim_updater)
    joint_discrim_params = f_joint_discrim.net.params()
    disp_costs.update(JD=disp(f_joint_discrim.net.get_loss()))
else:
    joint_discrim_params = []

if args.encode:
    net = f_encoder.net
    num_loss_terms = sum(l in net.loss for l in ['loss', 'loss_real', 'loss_gen'])
    weight = 1.0 / num_loss_terms
    if 'loss_real' in net.loss:
        net.add_agg_loss_term('loss_real', weight=weight)
    if 'loss_gen' in net.loss:
        net.add_agg_loss_term('loss_gen', weight=weight)
    nets.append(net)
    if args.encode_gen_weight:
        try:
            train_gen.net.add_loss(net.get_loss(),
                                   weight=args.encode_gen_weight)
        except KeyError:
            print 'Warning: encoder had no separate loss to contribute to gen'
        encode_gen_params = net.learnables()
        for k in net.learnable_keys():
            # mark all params unlearnable -- will be learned by gen
            net._params[k] = net._params[k][0], False
        assert not net.learnables()
    else:
        encode_gen_params = []
    encode_params = net.params()
    e_updates = net.get_updates(updater=updater)
    encoder_loss = f_encoder.net.get_loss()
    disp_costs.update(E=disp(encoder_loss))
    e_only_cost = f_encoder.encoder.cost
    if (e_only_cost is not None) and (e_only_cost != encoder_loss):
        disp_costs.update(e=disp(e_only_cost))
else:
    encode_params, encode_gen_params, e_updates = [], [], []

g_updates = []

def set_mode(mode):
    for m in modules:
        m.set_mode(mode)

train_label = (args.classifier and not args.classifier_deploy)
deploy_label = (args.classifier and args.classifier_deploy)

X_discrim_input = [Xis]
X_enc_input = [Xi]

for k, v in disp_costs.iteritems():
    if isinstance(v, (int, float, type(None))):
        del disp_costs[k]

set_mode('test')
if f_discrim is not None:
    discrim_feats = f_discrim.feats(X)
    _discrim_feats = lazy_function(X_discrim_input, discrim_feats.value)
set_mode('train')
if args.classifier:
    if f_discrim is not None:
        discrim_preds = f_discrim.labeler.preds(discrim_feats)
        _discrim_preds = lazy_function([discrim_feats.value],
                                        discrim_preds.value)

set_mode('train')
small_inputs = [Xis] if Xis != Xi else []
inputs = [Xi] + small_inputs
if train_label:
    inputs += [Y]
inputs += Z
deploy_inputs = [Xi] + small_inputs
if deploy_label:
    deploy_inputs += [Y]
deploy_inputs += Z
deploy_updates = []
[deploy_updates.extend(n.get_deploy_updates()) for n in nets]
_deploy_update = lazy_function(deploy_inputs, disp_costs.values(),
                               updates=deploy_updates)
update_both = (not args.no_update_both) and (args.k == 1)
if update_both:
    all_updates = g_updates + d_updates + e_updates
    _train_gd = lazy_function(inputs, [], updates=all_updates,
                              on_unused_input='ignore')
else:
    niter *= 2
    niter_decay *= 2
    # update encoder with generator
    train_d_updates = d_updates
    train_g_updates = g_updates + e_updates
    _train_g = lazy_function(inputs, [], updates=train_g_updates,
                             on_unused_input='ignore')
    _train_d = lazy_function(inputs, [], updates=train_d_updates)
_gen_train = lazy_function(Z, gX.value)
set_mode('test')
_gen = lazy_function(Z, gXtest.value)
_cost = lazy_function(inputs, disp_costs.values(),
                      on_unused_input='ignore')
if args.encode:
    enc_feats = f_encoder.feats(X)
    enc_preds = f_encoder.encoder.preds(enc_feats).value
    _enc = lazy_function(X_enc_input, enc_preds)
    enc_recon_outs = [o.value for o in dist.logits_to_recon(enc_preds)]
    _enc_recon = lazy_function(X_enc_input, enc_recon_outs)
    _enc_sample = lazy_function(X_enc_input,
        [o.value for o in dist.logits_to_sample(enc_preds)])
    _enc_l2distable = lazy_function(X_enc_input, dist.l2distable(enc_preds))
    _enc_feats = lazy_function(X_enc_input, enc_feats.value)
    if args.classifier:
        enc_label_preds = f_encoder.labeler.preds(enc_feats)
        _enc_preds = lazy_function([enc_feats.value],
                                   enc_label_preds.value)

def batch_feats(f, X, nbatch=args.batch_size, wraparound=False):
    """ wraparound=True makes sure all batches have exactly nbatch items,
        wrapping around to the beginning to grab enough extras to fill the
        last batch, if needed."""
    def is_ndarray_like(x):
        if issubclass(type(x), np.ndarray):
            return True
        try:
            x_array = np.array(x)
            return x_array.shape == x.shape
        except:
            return False
    def concat(*args):
        if any(isinstance(a, np.ndarray) for a in args):
            return np.concatenate(args, axis=0)
        out = []
        for a in args:
            out += list(a)
        return out
    if isinstance(X, np.ndarray):
        X = [X]
    assert len(X) > 0
    n = len(X[0])
    for x in X:
        assert len(x) == n
    out = None
    f_returns_ndarray = None
    for start in xrange(0, n, nbatch):
        end = min(start + nbatch, n)
        inputs = [x[start:end] for x in X]
        if wraparound:
            assert len(inputs[0]) > 0
            while len(inputs[0]) < nbatch:
                num_left = nbatch - len(inputs[0])
                inputs = [concat(i, x[:num_left]) for i, x in zip(inputs, X)]
        batch = f(*inputs)
        if f_returns_ndarray is None:
            f_returns_ndarray = is_ndarray_like(batch)
        if f_returns_ndarray:
            batch = [batch]
        if out is None:
            out = [np.zeros((n,) + b.shape[1:], dtype=b.dtype) for b in batch]
        num = end - start
        for o, b in zip(out, batch):
            o[start:end, ...] = b[:num]
    if f_returns_ndarray:
        return out[0]
    return out

def flat(X):
    return X.reshape(len(X), -1)

param_groups = dict(gen=gen_params)

def load_params(weight_prefix=None, resume_epoch=None, groups=param_groups):
    if resume_epoch is not None:
        assert weight_prefix is None
        weight_prefix = '%s/%d' % (model_dir, resume_epoch)
    assert weight_prefix is not None
    for key, param_list in groups.iteritems():
        if len(param_list) == 0: continue
        path = '%s_%s_params.jl' % (weight_prefix, key)
        if not os.path.exists(path):
            raise IOError('param file not found: %s' % path)
        saved_params = joblib.load(path)
        if len(saved_params) != len(param_list):
            raise ValueError(('different param list lengths: '
                              'len(saved)=%d != %d=len(params)')
                               % (len(saved_params), len(param_list)))
        print 'Loading %d params from: %s' % (len(param_list), path)
        for saved, shared in zip(saved_params, param_list):
            if shared.get_value().shape != saved.shape:
                raise ValueError(('shape mismatch: '
                                  'saved.shape=%s != %s=shared.shape')
                                 % (shared.get_value().shape, saved.shape))
            shared.set_value(saved)

if __name__ == '__main__':
    if (args.weights is not None) or (args.resume is not None):
        load_params(weight_prefix=args.weights, resume_epoch=args.resume)
    grid_size = 10, 10
    print 'Sampling z'
    z = dist.sample(num=np.prod(grid_size))
    print 'Computing generator samples G(z)'
    gX = _gen(*z)
    filename = 'gen_samples.png'
    print 'Saving generator samples G(z) to:', filename
    dataset.grid_vis(inverse_transform(gX), grid_size, filename)
