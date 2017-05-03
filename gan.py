#!/usr/bin/env python
from __future__ import division

import theano
import theano.tensor as T

from net import L, Net, Output, get_convnet, get_deconvnet

class LearningModule(object):
    modes = frozenset(('train', 'test'))
    def set_mode(self, mode):
        if mode not in self.modes:
            raise ValueError('Unknown mode %s; should be in: %s'
                             % (mode, self.modes))
        self.mode = mode

class Generator(LearningModule):
    def __init__(self, args, dist, nc, z=None, source=None, mode='train',
                 bnkwargs={}, gen_transform=None):
        N = self.net = Net(source=source, name='Generator')
        self.set_mode(mode)
        h_and_weights = dist.embed_data()
        bn_use_ave = (mode == 'test')
        self.data, _ = get_deconvnet(image_size=args.crop_resize,
                                     name=args.gen_net)(h_and_weights, N=N,
            nout=nc, size=args.gen_net_size, num_fc=args.net_fc,
            fc_dims=args.net_fc_dims, nonlin=args.deconv_nonlin,
            bn_use_ave=bn_use_ave, ksize=args.deconv_ksize, **bnkwargs)
        if gen_transform is not None:
            self.data = Output(gen_transform(self.data.value),
                               shape=self.data.shape)

class Featurizer(LearningModule):
    def __init__(self, args, dist, X, gX, Y,
                 nc=3, ny=None, mode='train', bnkwargs={},
                 discrim_weight=0, encode_weight=0,
                 joint_discrim_weight=0, updater=None,
                 net_name=None, net_size=None,
                 extra_cond_real=None, extra_cond_gen=None,
                 is_discrim=False, name='Featurizer'):
        self.X = X
        self.gX = gX
        self.Y = Y
        self.name = name
        self.args = args
        self.nc = nc
        self.bnkwargs = bnkwargs
        self.set_mode(mode)
        self.net = None
        self.updater = updater
        self.net_name = net_name
        self.net_size = net_size
        self._feats = {}
        self.cond_real = extra_cond_real
        self.cond_gen = extra_cond_gen
        assert (self.cond_real is None) == (self.cond_gen is None)
        self.cond = self.cond_real is not None
        assert ((self.cond_real is None) and (self.cond_gen is None)) or \
               len(self.cond_real) == len(self.cond_gen)
        self.do_encode = bool(encode_weight or joint_discrim_weight)
        self.is_discrim = is_discrim
        if args.cat_inputs and not any(a is None for a in (self.X, self.gX)):
            data_cat = L.Concat(self.X, self.gX, axis=0)
            if self.cond:
                weights = self.cond_real[1]
                assert weights == self.cond_gen[1]
                cond_cat = [L.Concat(zr, zg, axis=0) for zr, zg in
                            zip(self.cond_real[0], self.cond_gen[0])], weights
            else:
                cond_cat = None
            h_cat = self.feats(data_cat, cond=cond_cat)
            h_real, h_gen = L.Slice(h_cat, slice_point=[self.X.value.shape[0]],
                                    axis=0)
        else:
            h_real, h_gen = (None if (x is None) else self.feats(x, cond=c)
                for x, c in [(self.X, self.cond_real), (self.gX, self.cond_gen)])
        assert self.net is not None
        self.h_real, self.h_gen = h_real, h_gen
        if args.classifier and h_real is not None:
            self.labeler = labeler = MultilabelClassifier(self.net, ny)
            loss = labeler.loss(h_real, Y)
            assert labeler.W is not None
            # pop W from the params to be trained separately at 'deploy' time
            for w in reversed(labeler.W):
                assert self.net._params.popitem()[1][0] == w
            if args.classifier_deploy:
                add_updates = self.net.add_deploy_updates
            else:
                add_updates = self.net.add_updates
            add_updates(*updater(labeler.W, loss.mean()))
        if discrim_weight:
            self.add_discrim_loss(h_real, h_gen, weight=discrim_weight)
        if self.do_encode:
            self.encoder = encoder = Encoder(self.net, args, dist, self.Y,
                X=self.X, updater=updater,
                featurizer=self, bias=args.encode_out_bias)
            encoder.gen_cost = g = encoder.real_loss(self.h_real)
            encoder.real_cost = r = encoder.gen_loss(self.h_gen)
            cost_terms = [x.mean() for x in [g, r] if x is not None]
            if cost_terms:
                encoder.cost = sum(cost_terms)
                if encoder.real_cost is not None:
                    self.net.add_loss(encoder.real_cost,
                        weight=1, name='loss_real')
                if encoder.gen_cost is not None:
                    self.net.add_loss(encoder.gen_cost,
                        weight=1, name='loss_gen')
            else:
                encoder.cost = None

    def feats(self, image, cond=None):
        assert self.cond == (cond is not None)
        key = self.mode, image
        if key in self._feats:
            return self._feats[key]
        args = self.args
        if self.net is None:
            N = self.net = Net(name=self.name)
        else:
            N = Net(source=self.net, name=self.name)
        assert isinstance(image, Output)
        fc_drop = 0 if (self.mode == 'test') else (
            args.encode_net_fc_drop if
            (self.do_encode and (args.encode_net_fc_drop is not None))
            else args.net_fc_drop
        )
        fc_dims = (
            args.encode_net_fc_dims if
            (self.do_encode and (args.encode_net_fc_dims is not None))
            else args.net_fc_dims
        )
        num_fc = (
            args.encode_net_fc if
            (self.do_encode and (args.encode_net_fc is not None))
            else args.net_fc
        )
        nonlin = (
            args.encode_nonlin if
            (self.do_encode and (args.encode_nonlin is not None))
            else args.conv_nonlin
        )
        bn_use_ave = (self.mode == 'test')
        net = get_convnet(image_size=args.crop_resize, name=self.net_name)
        kwargs = {}
        kwargs.update(self.bnkwargs)
        if args.cond_fc is not None:
            kwargs.update(cond_num_fc=args.cond_fc)
        if args.cond_fc_dims is not None:
            kwargs.update(cond_fc_dims=args.cond_fc_dims)
        if args.cond_fc_drop is not None:
            kwargs.update(cond_fc_drop=args.cond_fc_drop)
        if self.is_discrim and args.minibatch_layer_size is not None:
            kwargs.update(minibatch_layer_size=args.minibatch_layer_size)
        if self.is_discrim and args.cat_inputs:
            kwargs.update(minibatch_layer_halves=True)
        if self.is_discrim and args.post_minibatch_layer_dims is not None:
            kwargs.update(post_minibatch_layer_dims=args.post_minibatch_layer_dims)
        if self.net_size is None:
            size = args.feat_net_size
        else:
            size = self.net_size
        f, _ = net(image, cond=cond, N=N, size=size,
                   num_fc=num_fc, fc_dims=fc_dims, fc_drop=fc_drop,
                   nonlin=nonlin, bn_use_ave=bn_use_ave,
                   bn_separate=args.bn_separate, **kwargs)
        self._feats[key] = f
        return f

    def add_discrim_loss(self, h_real, h_gen, weight=1, name='discrim'):
        discrim = {}
        assert not hasattr(self, name)
        setattr(self, name, discrim)
        discrim['discrim'] = d = BinaryClassifier(self.net)
        def add_discrim_cost(h_y, prefix=''):
            key = '%sloss' % prefix
            cost = {}
            for name, h, y in h_y:
                if h is None: continue
                loss = d.loss(h, y)
                loss_name = '%s_%s' % (key, name)
                self.net.add_loss(loss, name=loss_name)
                self.net.add_agg_loss_term(loss_name, weight=weight/2, name=key)
        h_y = ('real', h_real, 1), ('gen', h_gen, 0)
        add_discrim_cost(h_y)
        h_y_not = ((n, h, 1 - y) for n, h, y in h_y)
        add_discrim_cost(h_y_not, prefix='opp_')
        return discrim

class LinearPredictor(LearningModule):
    def __init__(self, N, nout, stddev=0, bias=False):
        self.nout = nout
        self.stddev = stddev
        self.bias = bias
        self.W = None
        self.b = None
        self.N = N
        self._preds = {}

    def preds(self, feats):
        key = feats
        if key in self._preds:
            return self._preds[key]
        assert isinstance(feats, Output)
        params_before = len(self.N.params())
        if self.W is None:
            preds = self.N.FC(feats, nout=self.nout, stddev=self.stddev)
        else:
            assert len(self.W) == 1
            W = Output(self.W[0])
            preds = self.N.FCMult(feats, W)
        net_params = self.N.params()
        num_new_params = len(net_params) - params_before
        if self.W is None:
            assert num_new_params >= 1
            self.W = net_params[-num_new_params:]
        else:
            assert num_new_params == 0
        if self.bias:
            params_before = len(self.N.params())
            if self.b is None:
                preds = self.N.Bias(preds)
            else:
                preds = self.N.BiasAdd(preds, self.b, axis=1)
            net_params = self.N.params()
            num_new_params = len(net_params) - params_before
            if self.b is None:
                assert num_new_params == 1
                self.b = Output(net_params[-1], shape=(self.nout,))
            else:
                assert num_new_params == 0
        self._preds[key] = preds
        return preds

class Encoder(LinearPredictor):
    def __init__(self, N, args, dist, y, X=None, updater=None,
                 featurizer=None, bias=False):
        self.N = N
        self.args = args
        self.dist = dist
        self.y = y
        # don't 0 initialize if discriminator conditioned on encoder output
        stddev = None if args.joint_discrim_weight else 0
        super(Encoder, self).__init__(N, nout=dist.recon_dim, stddev=stddev, bias=bias)
        self.encode_gen = bool(args.encode_weight)
        self.encode_real = args.encode_kldiv_real
        self.X = X
        self.updater = updater
        self.featurizer = featurizer

    def dist_recon_error(self, feats):
        assert isinstance(feats, Output)
        # generated image: compute reconstruction cost vs. real latent sample
        z = self.preds(feats).value
        cost = self.dist.weighted_recon_error(z)
        return cost * self.args.encode_weight

    def dist_kldiv_error(self, feats):
        assert isinstance(feats, Output)
        if not self.encode_real:
            return 0
        args, dist = self.args, self.dist
        cost = 0
        if args.encode_kldiv_real:
            cost += args.encode_kldiv_real * dist.kl_divergence(feats.value)
        # normalization by distribution dimension
        if args.encode_normalize:
            cost /= dist.norm_divisor()
        return cost

    def real_loss(self, real_feats):
        error = None
        if self.encode_real:
            error = self.dist_kldiv_error(real_feats)
        return error

    def gen_loss(self, gen_feats):
        error = None
        if self.encode_gen:
            error = self.dist_recon_error(gen_feats)
        return error

class MultilabelClassifier(LinearPredictor):
    def __init__(self, N, num_labels):
        super(MultilabelClassifier, self).__init__(N, nout=num_labels)

    def probs(self, feats):
        preds = self.preds(feats)
        return T.nnet.softmax(preds.value)

    def acc(self, feats, label):
        return (feats.argmax(axis=1) == label.value).mean()

    def loss(self, feats, label):
        probs = self.probs(feats)
        return T.nnet.categorical_crossentropy(probs, label.value)

class BinaryClassifier(LinearPredictor):
    def __init__(self, N):
        super(BinaryClassifier, self).__init__(N, nout=1)

    def probs(self, feats):
        preds = self.preds(feats)
        assert len(preds.shape) == 2
        assert preds.shape[1] == 1
        preds = self.N.Reshape(preds, shape=(-1,))
        return T.nnet.sigmoid(preds.value)

    def loss(self, feats, label):
        probs = self.probs(feats)
        return T.nnet.binary_crossentropy(probs, label)
