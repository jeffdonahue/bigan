from __future__ import division

import sys
sys.path.append('..')

from collections import Counter, OrderedDict
from functools import partial
import itertools

import numpy as np
import theano
import theano.tensor as T

from lib import inits
from lib.rng import py_rng, np_rng, t_rng
from lib.theano_utils import floatX, sharedX

from timeit import Timer

class Output(object):
    def __init__(self, value, shape=None, index_max=None):
        """
        value: A Theano Tensor, shared variable, or constant value.
        shape: May be None (default) if non-symbolic shape is accessible by
             value.get_value().shape (as in a Theano shared variable --
             tried first) or by value.shape (as in a NumPy array).
             Otherwise (e.g., if value is a symbolic Theano tensor), shape
             should be specified as an iterable of ints, where some may be -1
             for don't cares (e.g., batch size).
        index_max: If value is integer-typed, index_max may be used
             to specify its maximum value.  e.g., a batch of N one-hot vectors,
             each representing a word in a 500 word vocabulary, could be
             specified with an integer-typed Tensor with values in
             [0, 1, ..., 499], and index_max=500.
        """
        if isinstance(value, Output):
            raise TypeError("value may not be an Output")
        self.value = value
        if shape is None:
            try:
                shape = value.get_value().shape
            except AttributeError:
                try:
                    shape = value.shape
                    if isinstance(shape, theano.Variable):
                        shape = None
                except AttributeError:
                    pass
        if shape is not None:
            for s in list(shape) + ([] if (index_max is None) else [index_max]):
                assert isinstance(s, int)
                assert s >= 0
            shape = tuple(shape)
            assert len(shape) == value.ndim
        self.shape = shape
        if index_max is not None:
            assert isinstance(value, int) or str(value.dtype).startswith('int'), \
                ('if index_max is given, value must be integer-typed; '
                 'was: %s' % value.dtype)
            assert index_max == int(index_max)
            index_max = int(index_max)
            if index_max < 0:
                raise ValueError('index_max must be non-negative')
        self.index_max = index_max

    def __repr__(self):
        args = self.value, self.shape
        if self.index_max is not None:
            args += self.index_max,
            return 'Output(%s, shape=%s, index_max=%d)' % args
        return 'Output(%s, shape=%s)' % args

reparam = False
exp_reparam = False
def reparameterized_weights(w, g, epsilon=1e-8, nin_axis=None, exp=exp_reparam):
    for axis in nin_axis:
        assert isinstance(axis, int)
        assert 0 <= axis < w.ndim
    norm = T.sqrt(T.sqr(w).sum(axis=nin_axis, keepdims=True) + epsilon)
    if exp: g = T.exp(g)
    g_axes = list(reversed(xrange(g.ndim)))
    dimshuffle_pattern = ['x' if (axis in nin_axis) else g_axes.pop()
                          for axis in range(w.ndim)]
    assert not g_axes
    if 'x' in dimshuffle_pattern:
        g = g.dimshuffle(*dimshuffle_pattern)
    return g * w / norm

def castFloatX(x):
    return T.cast(x, theano.config.floatX)

def align_dims(a, b, axis):
    """Returns a broadcastable version of b which allows for various binary
       operations with a, with axis as the first aligned dimension."""
    extra_dims = a.ndim - (axis + b.ndim)
    if extra_dims < 0:
        raise ValueError('Must have a.ndim >= axis + b.ndim.')
    if extra_dims > 0:
        order = ('x',) * axis + tuple(range(b.ndim)) + ('x',) * extra_dims
        b = b.dimshuffle(*order)
    return b

def bias_add(h, b, axis=1):
    return h + align_dims(h, b, axis)

def scale_mul(h, g, axis=1):
    return h * align_dims(h, g, axis)

def scale_div(h, g, axis=1):
    return h / align_dims(h, g, axis)

class Layer(object):
    def __init__(self, *inputs, **kwargs):
        self.net         = kwargs.pop('net', None)
        if self.net is None:
            self.net = L
        self.name        = kwargs.pop('name', None)
        if self.name is None:
            self.name = '(anonymous) ' + self.__class__.__name__
        self.weight_init = kwargs.pop('weight_init', 0.02)
        if isinstance(inputs, Output):
            # `inputs` may be a single `Output`, or an iterable of them.
            # canonicalize single `Output`s to single-element lists here.
            inputs = [inputs]
        for input in inputs:
            assert isinstance(input, Output)
            assert input.shape is not None
        outs = self.get_output(*inputs, **kwargs)
        if not isinstance(outs, tuple):
            outs = outs,
        outs = list(outs)
        for index, out in enumerate(outs):
            assert isinstance(out, Output)
            if out.shape is None:
                skip_types = theano.compile.sharedvalue.SharedVariable, np.ndarray
                input_dict = {i.value: np.zeros(i.shape, dtype=i.value.dtype)
                              for i in inputs
                              if not isinstance(i.value, skip_types)}
                out_shape = out.value.shape.eval(input_dict)
                outs[index] = Output(out.value, out_shape,
                                     index_max=out.index_max)
        self.output = tuple(outs)
        print '(%s) Creating outputs with shapes: %s' % \
            (self.name, ', '.join(str(o.shape) for o in self.output))
        if len(self.output) == 1:
            self.output = self.output[0]

    def get_output(self, *a, **k):
        """Layer subclasses should implement get_output."""
        raise NotImplementedError

    def add_param(self, value, prefix=None, **kwargs):
        name = '%s/%s' % (self.name, prefix)
        return self.net._add_param(name, value, layer_name=self.name, **kwargs)

    def weights(self, shape, stddev=None, reparameterize=reparam,
                nin_axis=None, exp_reparam=exp_reparam):
        if stddev is None:
            stddev = self.weight_init
        print 'weights: initializing weights with stddev = %f' % stddev
        if stddev == 0:
            value = np.zeros(shape)
        else:
            value = np_rng.normal(loc=0, scale=stddev, size=shape)
        w = self.add_param(value, prefix='w')
        if isinstance(nin_axis, int):
            nin_axis = [nin_axis]
        assert isinstance(nin_axis, list)
        if reparameterize:
            g_shape = [dim for axis, dim in enumerate(shape)
                       if axis not in nin_axis]
            f_init = np.zeros if exp_reparam else np.ones
            g = self.add_param(f_init(g_shape, dtype=theano.config.floatX),
                               prefix='w_scale')
            w = reparameterized_weights(w, g, exp=exp_reparam,
                                        nin_axis=nin_axis)
        return w
    def biases(self, dim):
        return self.add_param(np.zeros(dim), prefix='b')
    def gains(self, dim, init_value=1):
        return self.add_param(init_value * np.ones(dim), prefix='g')
    def bn_count(self):
        return self.add_param(np.zeros(()), prefix='count', learnable=False,
                              dtype='int')
    def bn_mean(self, dim):
        return self.add_param(np.zeros(dim), prefix='mean', learnable=False)
    def bn_var(self, dim):
        return self.add_param(np.zeros(dim), prefix='var', learnable=False)

class Identity(Layer):
    def get_output(self, *h):
        return h

class Reshape(Layer):
    def get_output(self, h, shape=None):
        assert shape is not None, 'shape is required'
        return Output(h.value.reshape(shape), index_max=h.index_max)

class Concat(Layer):
    def get_output(self, *h, **kwargs):
        axis = kwargs.pop('axis', 1)
        if not isinstance(axis, int):
            raise TypeError('Concat axis must be an int, not %s' % type(axis))
        assert len(kwargs) == 0
        if len(h) < 1:
            raise ValueError('Concat: len(h) (= %d) < 1' % (len(h), ))
        if len(h) == 1:
            return h[0]
        assert len(h) >= 2
        num_axes = set(len(hi.shape) for hi in h)
        if len(num_axes) != 1:
            raise ValueError('Concat: inputs have differing ndims: %s'
                             % (num_axes, ))
        num_axes = num_axes.pop()
        if not (0 <= axis < num_axes):
            raise ValueError(('Concat: must have 0 <= axis (= %d) '
                              '< num_axes (= %d)') % (num_axes, axis))
        index_max = set(hi.index_max for hi in h)
        assert len(index_max) == 1
        index_max = index_max.pop()
        for i in xrange(num_axes):
            if i == axis:
                continue
            dims = set(hi.shape[i] for hi in h)
            if len(dims) != 1:
                raise ValueError('Concat: differing axis %d dimensions: %s'
                                 % (i, sorted(dims)))
        out = T.concatenate([hi.value for hi in h], axis=axis)
        return Output(out, index_max=index_max)

class Slice(Layer):
    def get_output(self, h, axis=1, num=None, slice_point=None):
        if not isinstance(axis, int):
            raise TypeError('Slice axis must be an int, not %s' % type(axis))
        shape = tuple(h.shape)
        assert 0 <= axis < len(shape)
        assert (num is None) != (slice_point is None), \
            "Slice: either num or slice_point must be specified (but not both)"
        if slice_point is not None:
            slice_point = list(slice_point)
        in_dim = shape[axis]
        if in_dim < 0:
            in_dim = h.value.shape[axis]
        if num is not None:
            num = int(num)
            size = in_dim // num
            if isinstance(in_dim, (int, float)):
                assert num * size == in_dim, \
                    'Slice: num(=%d) must evenly divide in_dim=(%d)' % (num, in_dim)
            slice_point = [i * size for i in xrange(1, num)]
        if len(slice_point) == 0:
            return Output(h.value, index_max=h.index_max)
        slices = [slice(start, end) for start, end in
                  zip([0] + slice_point, slice_point + [in_dim])]
        pad = [slice(None)] * axis
        return tuple(Output(h.value[pad + [s]], index_max=h.index_max)
                     for s in slices)

class EltwiseSum(Layer):
    def get_output(self, *H):
        assert len(H) > 0
        assert all(h.shape == H[0].shape for h in H)
        if len(H) == 1:
            return H[0]
        return Output(sum(h.value for h in H))

def conv_out_shape(in_shape, nout, ksize, stride, pad):
    assert len(in_shape) == 4
    out_size = ((s - ksize + 2 * pad) // stride + 1 for s in in_shape[2:])
    out_shape = (in_shape[0], nout) + tuple(out_size)
    return out_shape

def get_pad(pad, ksize):
    if pad == 'SAME':
        pad = (ksize - 1) // 2
    return pad

def conv_kwargs(stride, pad):
    assert isinstance(pad, int), 'pad must be an int'
    return dict(subsample=(stride, stride), border_mode=(pad, pad))

class Conv(Layer):
    def get_output(self, h, nout=None, ksize=1, stride=1, pad='SAME', group=1,
                   stddev=None, filter_flip=True):
        if nout is None:
            raise ValueError('nout must be provided')
        h, h_shape = h.value, h.shape
        if len(h_shape) == 2:
            h_shape += 1, 1
            h = h.reshape(*h_shape)
        assert len(h_shape) == 4
        nin = h_shape[1]
        assert nout % group == 0
        assert nin % group == 0
        W = self.weights((nout, nin // group, ksize, ksize),
                         stddev=stddev, nin_axis=[1, 2, 3])
        pad = get_pad(pad, ksize)
        subsample = stride, stride
        outs = []
        for g in xrange(group):
            if group > 1:
                size = nout // group
                w = W[g*size : (g+1)*size]
                size = nin // group
                hi = h[:, g*size : (g+1)*size]
            else:
                w = W
                hi = h
            outs.append(T.nnet.conv2d(hi, w, border_mode=pad,
                subsample=subsample, filter_flip=filter_flip))
        if len(outs) == 1:
            out = outs[0]
        else:
            out = T.concatenate(outs, axis=1)
        return Output(out)

def deconv(h, w, subsample=(1, 1), border_mode=(0, 0), out_dims=None,
           filter_flip=True):
    if out_dims is None:
        out_dims = h.shape[2] * subsample[0], h.shape[3] * subsample[1]
    assert len(out_dims) == 2
    out_shape = (None, None) + out_dims
    op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(imshp=out_shape,
        border_mode=border_mode, subsample=subsample, filter_flip=filter_flip)
    return op(w, h, out_dims)

class Deconv(Layer):
    def get_output(self, h, nout=None, ksize=1, stride=1, pad='SAME',
                   stddev=None):
        if nout is None:
            raise ValueError('nout must be provided')
        h, h_shape = h.value, h.shape
        out_shape_specified = isinstance(nout, tuple)
        pad = get_pad(pad, ksize)
        if out_shape_specified:
            out_shape = (h_shape[0],) + nout
            nout = nout[0]
        else:
            assert isinstance(nout, int)
            out_size = (stride * (s - 1) + ksize + (ksize % 2) - 2 * pad
                        for s in h_shape[2:])
            out_shape = (h_shape[0], nout) + tuple(out_size)
        nin = h_shape[1]
        if len(h_shape) == 2:
            h_shape += 1, 1
            h = h.reshape(*h_shape)
        expected_input_shape = conv_out_shape(in_shape=out_shape,
                nout=nin, ksize=ksize, stride=stride, pad=pad)
        if h_shape != expected_input_shape:
            raise ValueError(('deconv: input shape %s does not match expected '
                              'input shape %s for output shape %s')
                             % (h_shape, expected_input_shape, out_shape))
        W = self.weights((nin, nout, ksize, ksize), stddev=stddev,
                         nin_axis=[0, 2, 3])
        kwargs = conv_kwargs(stride, pad)
        kwargs.update(out_dims=out_shape[2:])
        out = deconv(h, W, **kwargs)
        return Output(out)

class Pool(Layer):
    def get_output(self, h, ksize=1, stride=1, pad='SAME', mode='max'):
        from theano.sandbox.cuda.dnn import dnn_pool
        if mode == 'ave':
            mode = 'average_exc_pad'  # other choice is average_inc_pad
        h, h_shape = h.value, h.shape
        assert len(h_shape) == 4
        pad = get_pad(pad, ksize)
        out = dnn_pool(h, ws=(ksize, ksize), stride=(stride, stride),
                       pad=(pad, pad))
        return Output(out)

class SpatialUpsample(Layer):
    def get_output(self, h, factor=2, axis=[2, 3], use_gpu_upsample=True):
        import theano.sandbox.cuda.dnn as dnn
        assert isinstance(factor, int)
        assert factor >= 1
        h, h_shape = h.value, h.shape
        assert isinstance(axis, list)
        assert all(isinstance(a, int) for a in axis)
        assert 0 <= min(axis)
        assert max(axis) < len(h_shape)
        if use_gpu_upsample:
            """ GPU SpatialUpsample using CuDNN (un)pooling. """
            from unpool_with_grad import dnn_upsample_nearest
            assert axis == [2, 3]
            return Output(dnn_upsample_nearest(h, factor))
        else:
            """ CPU-only SpatialUpsample. Slow. """
            for a in axis:
                h = T.extra_ops.repeat(h, factor, axis=a)
            return Output(h)

class Upconv(Layer):
    """Alternative to deconvolution: explicitly upsample then convolve."""
    def get_output(self, h, stride=1, nout=None, **conv_kwargs):
        N = self.net
        h = N.SpatialUpsample(h, factor=stride)
        if isinstance(nout, tuple):
            nout, height, width = nout
        else:
            height = width = None
        h = N.Conv(h, nout=nout, stride=1, **conv_kwargs)
        if height is not None:
            assert h.shape[2:] == (height, width)
        return h

class FC(Layer):
    def get_output(self, h, nout=None, stddev=None,
                   reparameterize=reparam, exp_reparam=exp_reparam):
        h, h_shape, h_max = h.value, h.shape, h.index_max
        nin = np.prod(h_shape[1:], dtype=np.int) if (h_max is None) else h_max
        out_shape_specified = isinstance(nout, tuple)
        if out_shape_specified:
            out_shape = nout
        else:
            assert isinstance(nout, int)
            out_shape = nout,
        nout = np.prod(out_shape)
        nin_axis = [0]
        W = self.weights((nin, nout), stddev=stddev,
            reparameterize=reparameterize, nin_axis=nin_axis,
            exp_reparam=exp_reparam)
        if h_max is None:
            if h.ndim > 2:
                h = T.flatten(h, 2)
            out = T.dot(h, W)
        else:
            assert nin >= 1, 'FC: h.index_max must be >= 1; was: %s' % (nin,)
            assert h.ndim == 1
            out = W[h]
        return Output(out)

class FCMult(Layer):
    def get_output(self, h, W):
        h, h_shape, h_max = h.value, h.shape, h.index_max
        nin = np.prod(h_shape[1:], dtype=np.int) if (h_max is None) else h_max
        assert nin == W.shape[0]
        W = W.value
        if h_max is None:
            if h.ndim > 2:
                h = T.flatten(h, 2)
            out = T.dot(h, W)
        else:
            assert nin >= 1, 'FC: h.index_max must be >= 1; was: %s' % (nin,)
            assert h.ndim == 1
            out = W[h]
        return Output(out)

class OneHot(FC):
    nin_to_one_hots = {}
    def get_output(self, h, **kwargs):
        nin = h.index_max
        if not isinstance(nin, int):
            raise TypeError('h.index_max must be an integer; was: %s (type %s)'
                            % (nin, type(nin)))
        shape = nin, nin
        if nin not in self.nin_to_one_hots:
            self.nin_to_one_hots[nin] = inits.Identity(scale=1)(shape)
        W = Output(self.nin_to_one_hots[nin], shape=shape)
        return self.net.FCMult(h, W, **kwargs)

class Gain(Layer):
    def get_output(self, h, log_gain=False, axis=1):
        h, h_shape = h.value, h.shape
        init_value = 0 if log_gain else 1
        g = self.gains(h_shape[1], init_value=init_value)
        if log_gain:
            g = T.exp(g)
        out = scale_mul(h, g, axis=axis)
        return Output(out)

class Bias(Layer):
    def get_output(self, h, axis=1):
        b = self.biases(h.shape[axis])
        out = bias_add(h.value, b, axis=axis)
        return Output(out)

class BiasAdd(Layer):
    def get_output(self, h, b, axis=0):
        out = bias_add(h.value, b.value, axis=axis)
        return Output(out)

class BatchNorm(Layer):
    def get_output(self, h, u=None, s=None, use_ave=False, ave_frac=1,
                   epsilon=1e-8, log_var_move_ave=False,
                   var_bias_correction=True, ignore_moment_grads=False):
        no_grad = theano.gradient.disconnected_grad
        def move_ave_update(param, update, log_update=False):
            if log_update:
                new_param = ave_frac * param + T.log(update)
            else:
                new_param = ave_frac * param + update
            self.net.deploy_updates[param] = new_param
        h, h_shape = h.value, h.shape
        assert h.ndim >= 1
        axes = [0] + range(2, h.ndim)
        count = self.bn_count()
        if not use_ave:
            move_ave_update(count, 1)
        if u is None:
            mu = self.bn_mean(h_shape[1])
            if use_ave:
                u = castFloatX(mu / count)
            else:
                u = h.mean(axis=axes)
                move_ave_update(mu, u)
        if ignore_moment_grads:
            u = no_grad(u)
        h = bias_add(h, -u)
        if s is None:
            sigma = self.bn_var(h_shape[1])
            if use_ave:
                s = castFloatX(sigma / count)
                if log_var_move_ave:
                    s = T.exp(s)
            else:
                s = T.sqr(h).mean(axis=axes)
                if var_bias_correction:
                    n = h.shape[0] * T.prod(h.shape[2:])
                    nf = T.cast(n, theano.config.floatX)
                    # undo 1/n normalization; renorm by 1/(n-1) (unbiased var.)
                    s_unbiased = (nf / (nf - 1)) * s
                else:
                    s_unbiased = s
                move_ave_update(sigma, s_unbiased, log_update=log_var_move_ave)
        stdev = T.sqrt(s + epsilon)
        if ignore_moment_grads:
            stdev = no_grad(stdev)
        h = scale_div(h, stdev)
        return Output(h)

class Nonlinearity(Layer):
    def nonlin(self, op, h):
        out = op(h.value)
        return Output(out, h.shape)

class ReLU(Nonlinearity):
    def get_output(self, h):
        def relu(x):
            return (x + abs(x)) / 2
        return self.nonlin(relu, h)

class LReLU(Nonlinearity):
    def get_output(self, h, leak=0.2):
        def lrelu(x):
            return T.nnet.relu(x, alpha=leak)
        return self.nonlin(lrelu, h)

class Sigmoid(Nonlinearity):
    def get_output(self, h):
        return self.nonlin(T.nnet.sigmoid, h)

class Identity(Nonlinearity):
    def get_output(self, h):
        return h

class Scale(Nonlinearity):
    def get_output(self, h, scale=1):
        if scale == 1:
            return h
        return self.nonlin(lambda x: scale * x, h)

class Dropout(Nonlinearity):
    def get_output(self, h, ratio=0.5):
        assert 0 <= ratio < 1
        def op(h):
            if ratio == 0:
                return h
            keep_prob = 1 - ratio
            mask = t_rng.binomial(h.shape, p=keep_prob,
                                  dtype=theano.config.floatX)
            return h * mask / keep_prob
        return self.nonlin(op, h)

class BiReLU(Layer):
    def get_output(self, h, axis=1):
        assert h.value.ndim > axis
        assert h.shape[axis] >= 0
        N = self.net
        neg_h = N.Scale(h, scale=-1)
        return N.Concat(N.ReLU(h), N.ReLU(neg_h), axis=axis)

class L(object):
    layers = {k: v for k, v in globals().iteritems()
              if isinstance(v, type) and issubclass(v, Layer)}
    def __getattr__(self, attr):
        def layer_method(*args, **kwargs):
            return self.layers[attr](*args, **kwargs).output
        return layer_method
L = L()

def checked_update(target_map, source={}, **new_kwargs):
    """
    Inserts the keyval pairs of new_kwargs into target_map (a dict),
    raising a ValueError if target_map already contains any of the keys.

    Returns None, updating target_map in-place.
    """
    for k, v in itertools.chain(source.iteritems(), new_kwargs.iteritems()):
        if k in target_map:
            raise ValueError('checked_update: key exists: %s' % k)
        target_map[k] = v

class Net(object):
    layer_types = {k: v for k, v in globals().iteritems()
                   if isinstance(v, type) and issubclass(v, Layer)}

    def __init__(self, source=None, name=None):
        self.name = name
        self.name_prefix = '' if (name is None) else ('%s/' % name)
        if source is not None:
            assert name == source.name

        """self.loss: maps strings to losses (scalar tensor values)"""
        self.loss = OrderedDict()

        """Support 'aggregate' losses -- weighted sums of other losses."""
        self.is_agg_loss = OrderedDict()
        self.agg_loss_terms = OrderedDict()

        """self.layers: maps layer names (strings) to layers"""
        self.layers = OrderedDict()

        """
        self.updates: *additional* updates to be performed at training time --
        any updates *besides* updates for the learnable params (self.params).
        Update mappings are theano.shared -> theano.tensor, where the latter
        element (the tensor) is the updated value of the shared variable.
        """
        self.updates = OrderedDict()

        """
        self.deploy_updates: updates to be performed at deployment time; e.g.,
        accumulation of batch norm statistics, or updates used to learn an
        independent classifier.  Could often be learned at training time
        instead (by instead putting the updates in self.updates).
        """
        self.deploy_updates = OrderedDict()
        self.layer_count = Counter()
        self.reuse = source is not None
        self._params = OrderedDict()
        self.source_params = source._params if self.reuse else None

    def params(self):
        return [p for p, _ in self._params.itervalues()]

    def learnables(self):
        return [p for p, l in self._params.itervalues() if l]

    def learnable_keys(self):
        return [k for k, (_, l) in self._params.iteritems() if l]

    def add_deploy_updates(self, *args, **kwargs):
        for k in (dict(args), kwargs):
            checked_update(self.deploy_updates, k)

    def add_updates(self, *args, **kwargs):
        for k in (dict(args), kwargs):
            checked_update(self.updates, k)

    def get_updates(self, updater=None, loss='loss', extra_params=[]):
        updates = self.updates.items()
        if updater is not None:
            try:
                loss_value = self.get_loss(loss).mean()
                params = self.learnables() + extra_params
                updates += updater(params, loss_value)
            except KeyError:
                # didn't have a loss, check that we also had no learnables
                assert not self.learnables(), 'had no loss but some learnables'
        return updates

    def get_deploy_updates(self):
        return self.deploy_updates.items()

    def add_loss(self, value, weight=1, name='loss'):
        print 'Adding loss:', (self.name, weight, name)
        if value.ndim > 1:
            raise ValueError('value must be 0 or 1D (not %dD)' % value.ndim)
        if name not in self.is_agg_loss:
            self.is_agg_loss[name] = False
        assert not self.is_agg_loss[name]
        if (name not in self.loss) and (weight == 1):
            # special case where we can just set the loss to value directly
            # maintains tensor equality (==) when possible
            self.loss[name] = value
        else:
            if weight == 0:
                value = T.zeros_like(value, dtype=theano.config.floatX)
                self.loss[name] = value
            else:
                if weight != 1:
                    value *= weight
                if name in self.loss:
                    self.loss[name] += value
                else:
                    self.loss[name] = value

    def add_agg_loss_term(self, term_name, weight=1, name='loss'):
        print 'Adding agg loss:', (self.name, weight, name, term_name)
        if name not in self.is_agg_loss:
            self.is_agg_loss[name] = True
            self.agg_loss_terms[name] = []
        assert self.is_agg_loss[name]
#         assert not self.is_agg_loss[term_name], \
#             'Recursive aggregate losses not supported.'
        assert name != term_name
        self.agg_loss_terms[name].append((term_name, weight))

    def get_loss(self, name='loss'):
        if self.is_agg_loss[name]:
            return sum(w * self.get_loss(k).mean()
                       for k, w in self.agg_loss_terms[name])
        no_grad = theano.gradient.disconnected_grad
        total_loss = self.loss[name]
        assert total_loss.dtype.startswith('float')
        return total_loss

    def _add_layer(self, layer_constructor, *args, **kwargs):
        type_name = layer_constructor.__name__
        self.layer_count[type_name] += 1
        name = '%s%s%d' % (self.name_prefix, type_name,
                           self.layer_count[type_name])
        checked_update(kwargs, net=self, name=name)
        layer = layer_constructor(*args, **kwargs)
        checked_update(self.layers, **{name: layer})
        return layer

    def _add_param(self, name, value, learnable=True, layer_name='',
                   dtype=theano.config.floatX):
        if self.reuse:
            assert name in self.source_params, \
                'param "%s does not exist and self.reuse==True' % name
            param = self.source_params[name][0]
            existing_shape = param.get_value().shape
            if value.shape != existing_shape:
                raise ValueError('Param "%s": incompatible shapes %s vs. %s' %
                                 (name, existing_shape, value.shape))
            print '(%s) Reusing param "%s" with shape: %s' % \
                (layer_name, name, value.shape)
        else:
            print '(%s) Adding param "%s" with shape: %s' % \
                  (layer_name, name, value.shape)
            param = sharedX(value, dtype=dtype, name=name)
        assert name not in self._params, 'param "%s already exists' % name
        self._params[name] = (param, bool(learnable))
        return param

    def __getattr__(self, attr):
        def layer_method(*args, **kwargs):
            return self._add_layer(self.layer_types[attr],
                                   *args, **kwargs).output
        if attr in self.layer_types:
            return layer_method
        raise AttributeError('Unknown attribute: %s' % attr)

def batch_norm(N, h, batch_norm=True, bias=False, gain=False, log_gain=False,
               use_ave=False):
    if batch_norm: h = N.BatchNorm(h, use_ave=use_ave)
    if gain      : h = N.Gain(h, log_gain=log_gain)
    if bias      : h = N.Bias(h)
    return h

def multifc(N, H, nout=None, renormalize_weights=True, **kwargs):
    if isinstance(H, tuple):
        H, weights = H
    else:
        weights = None
    if isinstance(H, Output):
        H = [H]
        if weights is not None:
            weights = [weights]
    if weights is None:
        weights = [1] * len(H)
    assert isinstance(H, list) and isinstance(weights, list)
    assert len(H) == len(weights)
    for h in H:
        assert isinstance(h, Output)
    weights = np.array(weights, dtype=theano.config.floatX)
    if renormalize_weights:
        weights *= len(weights) / np.sum(weights)
    unweighted_outputs = [N.FC(h, nout=nout, **kwargs) for h in H]
    weighted_outputs = [N.Scale(o, scale=w)
                        for o, w in zip(unweighted_outputs, weights)]
    return N.EltwiseSum(*weighted_outputs)

def apply_cond(N, h, cond=None, ksize=1, bn=None, bn_separate=False):
    if cond is not None:
        stddev = 0.02
        if not bn_separate:
            stddev *= ksize ** 2
        b = multifc(N, cond, nout=h.shape[1], stddev=stddev)
        if (bn is not None) and bn_separate:
            b = bn(b)
            h = bn(h)
        h = N.BiasAdd(h, b)
        if (bn is not None) and bn_separate:
            # if X, Y ~ N(0, 1), then std(X+Y) = sqrt(2)
            # compensate by dividing by sqrt(2)
            scale = floatX(1. / np.sqrt(2))
            h = N.Scale(h, scale=scale)
    if (bn is not None) and ((not bn_separate) or (cond is None)):
        h = bn(h)
    return h

kwargs28 = dict(batch_norm=True, bias=False, gain=False)
def deconvnet_28(h, N=None, nout=3, size=None, bn_flat=True,
                 nonlin='ReLU', bnkwargs=kwargs28, bn_use_ave=False,
                 **ignored_kwargs):
    cond = h
    if N is None: N = Net()
    nonlin = getattr(N, nonlin)
    if size is None: size = 64
    def bn(h):
        return batch_norm(N, h, use_ave=bn_use_ave, **bnkwargs)
    def acts(h, ksize=1):
        h = apply_cond(N, h, cond=cond, ksize=ksize)
        h = bn(h)
        h = nonlin(h)
        return h
    h = nonlin(bn(multifc(N, h, nout=1024)))
    shape = size*2, 7, 7
    if bn_flat:
        # Batch normalize, then reshape to image.
        # (Each individual pixel of reshaped image is treated as a separate
        # channel in batch norm. This is what was done in the original code.)
        h = acts(N.FC(h, nout=np.prod(shape)))
        # recompute channel_dim in case it was altered by acts
        channel_dim = np.prod(h.shape[1:]) // np.prod(shape[1:])
        assert channel_dim * np.prod(shape[1:]) == np.prod(h.shape[1:])
        shape = (channel_dim, ) + shape[1:]
        h = N.Reshape(h, shape=((-1, ) + shape))
    else:
        h = acts(N.FC(h, nout=shape))
    h = acts(N.Deconv(h, nout=size*1, ksize=5, stride=2), ksize=5)
    h =      N.Deconv(h, nout=  nout, ksize=5, stride=2)
    h = N.Sigmoid(h) # generate images in [0, 1] range
    return h, N
def deconvnet_mnist_mlp(h, N=None, nout=3, size=None, bn_flat=True,
                        nonlin='ReLU', bnkwargs=kwargs28, bn_use_ave=False,
                        **ignored_kwargs):
    cond = h
    if N is None: N = Net()
    nonlin = getattr(N, nonlin)
    if size is None: size = 64
    def bn(h):
        return batch_norm(N, h, use_ave=bn_use_ave, **bnkwargs)
    def acts(h, ksize=1):
        h = apply_cond(N, h, cond=cond, ksize=ksize, bn=bn)
        h = nonlin(h)
        return h
    h = nonlin(bn(multifc(N, h, nout=size*16)))
    # h = acts(N.FC(h, nout=size*16))
    h = acts(N.FC(h, nout=size*16))
    h = N.FC(h, nout=28*28)
    h = N.Sigmoid(h) # generate images in [0, 1] range
    h = N.Reshape(h, shape=[-1, 1, 28, 28])
    return h, N
def deconvnet_pong_mlp(h, N=None, nout=3, size=None, bn_flat=True,
                        nonlin='ReLU', bnkwargs=kwargs28, bn_use_ave=False,
                        **ignored_kwargs):
    cond = h
    if N is None: N = Net()
    nonlin = getattr(N, nonlin)
    if size is None: size = 64
    def bn(h):
        return batch_norm(N, h, use_ave=bn_use_ave, **bnkwargs)
    def acts(h, ksize=1):
        h = apply_cond(N, h, cond=cond, ksize=ksize, bn=bn)
        h = nonlin(h)
        return h
    h = nonlin(bn(multifc(N, h, nout=size*16)))
    # h = acts(N.FC(h, nout=size*16))
    h = acts(N.FC(h, nout=size*16))
    h = N.FC(h, nout=4*84*84)
    h = N.Sigmoid(h) # generate images in [0, 1] range
    h = N.Reshape(h, shape=[-1, 4, 84, 84])
    return h, N
def min_convnet_28(h, N=None, size=None, nonlin='LReLU', bnkwargs=kwargs28,
                   bn_use_ave=False, fc_drop=0, **ignored_kwargs):
    if N is None: N = Net()
    nonlin = getattr(N, nonlin)
    if size is None: size = 64
    def bn(h):
        return batch_norm(N, h, use_ave=bn_use_ave, **bnkwargs)
    h = nonlin(bn(  N.FC(h, nout=16*size)))
    if fc_drop != 0:
        h = N.Dropout(h, ratio=fc_drop)
    return h, N
def min_deconvnet_28(h, N=None, nout=3, size=None, bn_flat=True, nonlin='ReLU',
                     bnkwargs=kwargs28, bn_use_ave=False, **ignored_kwargs):
    cond = h
    if N is None: N = Net()
    nonlin = getattr(N, nonlin)
    if size is None: size = 64
    def bn(h):
        return batch_norm(N, h, use_ave=bn_use_ave, **bnkwargs)
    h = nonlin(bn(N.FC(h, nout=1024)))
    h = N.FC(h, nout=(nout, 28, 28))
    h = N.Sigmoid(h) # generate images in [0, 1] range
    return h, N
if False:
    convnet_28 = min_convnet_28
    deconvnet_28 = min_deconvnet_28

kwargs64 = dict(batch_norm=True, bias=True, gain=True)

def deconvnet_64(h, N=None, nout=3, size=None, bn_flat=True,
                 nonlin='ReLU', bnkwargs=kwargs64, num_fc=0, fc_dims=[],
                 bn_use_ave=False, num_refine=0, refine_ksize=5,
                 start_size=4, ksize=5, deconv_op='Deconv'):
    cond = h
    if N is None: N = Net()
    nonlin = getattr(N, nonlin)
    if size is None: size = 128
    def acts(h, ksize=1, do_cond=True):
        if do_cond: h = apply_cond(N, h, cond=cond, ksize=ksize)
        h = batch_norm(N, h, use_ave=bn_use_ave, **bnkwargs)
        h = nonlin(h)
        return h
    deconv_op = getattr(N, deconv_op)
    def deconv_acts(h, ksize=ksize, **kwargs):
        return acts(deconv_op(h, ksize=ksize, **kwargs), ksize=ksize)
    # do FCs
    fc_dims = [size*16] * num_fc + fc_dims
    for index, dim in enumerate(fc_dims):
        h = acts(multifc(N, h, nout=dim), do_cond=bool(index))
    # do deconv from 4x4
    ss = start_size
    shape = size*8, ss, ss
    if bn_flat:
        # Batch normalize, then reshape to image.
        # (Each individual pixel of reshaped image is treated as a separate
        # channel in batch norm. This is what was done in the original code.)
        h = acts(multifc(N, h, nout=np.prod(shape)), do_cond=bool(fc_dims))
        # recompute channel_dim in case it was altered by acts
        channel_dim = np.prod(h.shape[1:]) // np.prod(shape[1:])
        assert channel_dim * np.prod(shape[1:]) == np.prod(h.shape[1:])
        shape = (channel_dim, ) + shape[1:]
        h = N.Reshape(h, shape=((-1, ) + shape))
    else:
        h = acts(multifc(N, h, nout=shape), do_cond=bool(fc_dims))
    h = deconv_acts(h, nout=(size*4, ss*2, ss*2), stride=2)
    h = deconv_acts(h, nout=(size*2, ss*4, ss*4), stride=2)
    h = deconv_acts(h, nout=(size*1, ss*8, ss*8), stride=2)
    curnout = (nout if num_refine == 0 else size//2, ss*16, ss*16)
    h = deconv_op(h, nout=curnout, ksize=ksize, stride=2)
    for i in xrange(num_refine):
        h = acts(h, ksize=k)
        is_last = (i == num_refine - 1)
        curnout = nout if is_last else (size//2)
        h = N.Conv(h, nout=curnout, ksize=refine_ksize, stride=1)
    h = N.Sigmoid(h) # generate images in [0, 1] range
    return h, N

def deconvnet_84(*args, **kwargs):
    if 'start_size' not in kwargs:
        kwargs.update(start_size=6)
    if 'nout' not in kwargs:
        kwargs.update(nout=4)
    h, N = deconvnet_64(*args, **kwargs)
    assert h.shape[2:] == (96, 96)
    c = (96 - 84) // 2
    h_cropped = h.value[:, :, c:-c, c:-c]
    h = Output(h_cropped, h.shape[:2] + (84, 84))
    h = N.Bias(h)
    return h, N

def upconvnet_84(*args, **kwargs):
    return deconvnet_84(*args, deconv_op='Upconv', **kwargs)

def upconvnet_64(*a, **k):
    return deconvnet_64(*a, deconv_op='Upconv', **k)

def deconvnet_128(h, N=None, nout=3, size=None, bn_flat=True,
                  nonlin='ReLU', bnkwargs=kwargs64, num_fc=0, fc_dims=[],
                  bn_use_ave=False, num_refine=0, refine_ksize=5,
                  start_size=4, ksize=5, deconv_op='Deconv'):
    cond = h
    if N is None: N = Net()
    nonlin = getattr(N, nonlin)
    if size is None: size = 64
    def acts(h, ksize=1):
        h = apply_cond(N, h, cond=cond, ksize=ksize)
        h = batch_norm(N, h, use_ave=bn_use_ave, **bnkwargs)
        h = nonlin(h)
        return h
    deconv_op = getattr(N, deconv_op)
    def deconv_acts(h, ksize=ksize, **kwargs):
        return acts(deconv_op(h, ksize=ksize, **kwargs), ksize=ksize)
    # architecture from
    # https://github.com/openai/improved-gan/blob/master/imagenet/generator.py#L71
    # do FCs
    fc_dims = [size*8] * num_fc + fc_dims
    for dim in fc_dims:
        h = acts(multifc(N, h, nout=dim))
    # do deconv from 4x4
    ss = start_size
    shape = size*8, ss, ss
    if bn_flat:
        # Batch normalize, then reshape to image.
        # (Each individual pixel of reshaped image is treated as a separate
        # channel in batch norm. This is what was done in the original code.)
        h = acts(multifc(N, h, nout=np.prod(shape)))
        # recompute channel_dim in case it was altered by acts
        channel_dim = np.prod(h.shape[1:]) // np.prod(shape[1:])
        assert channel_dim * np.prod(shape[1:]) == np.prod(h.shape[1:])
        shape = (channel_dim, ) + shape[1:]
        h = N.Reshape(h, shape=((-1, ) + shape))
    else:
        h = acts(multifc(N, h, nout=shape))
    # h \in (size*8, 4, 4)
    h = deconv_acts(h, nout=(size*4, ss* 2, ss* 2), stride=2)  # g_h1 (size*4,   8,   8)
    h = deconv_acts(h, nout=(size*2, ss* 4, ss* 4), stride=2)  # g_h2 (size*2,  16,  16)
    h = deconv_acts(h, nout=(size*1, ss* 8, ss* 8), stride=2)  # g_h3 (size*1,  32,  32)
    h = deconv_acts(h, nout=(size//2, ss*16, ss*16), stride=2)  # g_h4 (size*1,  64,  64)
#     h = deconv_acts(h, nout=(size*1, ss*32, ss*32), stride=2)  # g_h5 (size*1, 128, 128)
#     h = N.Conv(h, nout=nout, ksize=refine_ksize, stride=1)     # g_h5 (     3, 128, 128)
    h = deconv_op(h, nout=(nout, ss*32, ss*32), stride=2)  # g_h5 (size*1, 128, 128)
    if False:  # ignore this stuff for now...
        curnout = nout if (num_refine == 0) else (size//2, ss*32, ss*32)
        h = N.Deconv(h, nout=curnout, ksize=k, stride=2)
        for i in xrange(num_refine):
            h = acts(h, ksize=k)
            is_last = (i == num_refine - 1)
            curnout = nout if is_last else (size//2)
            h = N.Conv(h, nout=curnout, ksize=refine_ksize, stride=1)
    h = N.Sigmoid(h) # generate images in [0, 1] range
    return h, N

def upconvnet_128(*a, **k):
    return deconvnet_128(*a, deconv_op='Upconv', **k)

# 'deepsimgen' generator from
# http://lmb.informatik.uni-freiburg.de/Publications/2016/DB16a/deepsim.pdf
def deepsimgen_deconvnet_128(
        h, N=None, nout=3, size=None, bn_flat=True,
        nonlin='LReLU', bnkwargs=kwargs64, num_fc=0, fc_dims=[],
        bn_use_ave=False, num_refine=0, refine_ksize=5,
        start_size=4, ksize=4):
    cond = h
    if N is None: N = Net()
    nonlin = getattr(N, nonlin)
    if size is None: size = 64
    def acts(h, ksize=1, do_cond=True):
        if do_cond: h = apply_cond(N, h, cond=cond, ksize=ksize)
        h = batch_norm(N, h, use_ave=bn_use_ave, **bnkwargs)
        h = nonlin(h)
        return h
    def deconv_acts(h, ksize=ksize, **kwargs):
        return acts(N.Deconv(h, ksize=ksize, **kwargs),
                    # ksize=ksize, do_cond=False)
                    ksize=ksize, do_cond=True)
    def conv_acts(h, ksize=ksize, **kwargs):
        return acts(N.Conv(h, ksize=ksize, **kwargs), ksize=ksize)
    # do FCs
    assert len(fc_dims) == 0 and num_fc == 0
    # fc_dims = [2048] * 2
    fc_dims = [2048] * 1
    for dim in fc_dims:
        h = acts(multifc(N, h, nout=dim), do_cond=False)
    # do deconv from 4x4
    ss = start_size
    shape = 256, ss, ss
    if bn_flat:
        # Batch normalize, then reshape to image.
        # (Each individual pixel of reshaped image is treated as a separate
        # channel in batch norm. This is what was done in the original code.)
        h = acts(multifc(N, h, nout=np.prod(shape)))
        # recompute channel_dim in case it was altered by acts
        channel_dim = np.prod(h.shape[1:]) // np.prod(shape[1:])
        assert channel_dim * np.prod(shape[1:]) == np.prod(h.shape[1:])
        shape = (channel_dim, ) + shape[1:]
        h = N.Reshape(h, shape=((-1, ) + shape))
    else:
        h = acts(multifc(N, h, nout=shape))
    conv_ksize = 3                                                      #  start -> (256,   4,   4)
    h = deconv_acts(h, nout=(4*size,  2*ss,  2*ss), stride=2, ksize=4)  # uconv1 -> (256,   8,   8)
    h =   conv_acts(h, nout=(8*size), ksize=conv_ksize)                 #  conv1 -> (512,   8,   8)
    h = deconv_acts(h, nout=(4*size,  4*ss,  4*ss), stride=2, ksize=4)  # uconv2 -> (256,  16,  16)
    h =   conv_acts(h, nout=(4*size), ksize=conv_ksize)                 #  conv2 -> (256,  16,  16)
    h = deconv_acts(h, nout=(2*size,  8*ss,  8*ss), stride=2, ksize=4)  # uconv3 -> (128,  32,  32)
    h =   conv_acts(h, nout=(2*size), ksize=conv_ksize)                 #  conv3 -> (128,  32,  32)
    h = deconv_acts(h, nout=(  size, 16*ss, 16*ss), stride=2, ksize=4)  # uconv4 -> ( 64,  64,  64)
    curnout = nout if (num_refine == 0) else (size//2)
    h = N.Deconv(h, nout=(curnout, 32*ss, 32*ss), stride=2, ksize=4)    # uconv5 -> (  3, 128, 128)
    for i in xrange(num_refine):
        h = acts(h, ksize=k)
        is_last = (i == num_refine - 1)
        curnout = nout if is_last else (size//2)
        h = N.Conv(h, nout=curnout, ksize=refine_ksize, stride=1)
    h = N.Sigmoid(h) # generate images in [0, 1] range
    return h, N
# deconvnet_128 = deepsimgen_deconvnet_128

deconvnet_96 = partial(deconvnet_128, start_size=3)
deconvnet_96 = partial(deconvnet_64, start_size=6)

def convnet(h, N=None, cond=None, arch=None, size=None, nonlin='LReLU',
            num_fc=0, fc_dims=[], fc_drop=0,
            cond_num_fc=0, cond_fc_dims=[], cond_fc_drop=0,
            minibatch_layer_halves=False, minibatch_layer_size=None,
            post_minibatch_layer_dims=[], bnkwargs=kwargs64,
            bn_separate=False, bn_use_ave=False):
    if N is None: N = Net()
    nonlin = getattr(N, nonlin)
    def bn(h):
        return batch_norm(N, h, use_ave=bn_use_ave, **bnkwargs)
    def acts(h, ksize=1, do_cond=True):
        if do_cond: h = apply_cond(N, h, cond=cond, ksize=ksize,
                                   bn=bn, bn_separate=bn_separate)
        h = nonlin(h)
        return h
    def conv_acts(h, ksize=1, **kwargs):
        return acts(N.Conv(h, ksize=ksize, **kwargs), ksize=ksize)
    if cond is not None:
        cond_fc_dims = [1024] * cond_num_fc + cond_fc_dims
        hcond = cond
        for dim in cond_fc_dims:
            hcond = multifc(N, hcond, nout=dim)
            hcond = acts(hcond, do_cond=False)
            if cond_fc_drop != 0:
                hcond = N.Dropout(hcond, ratio=cond_fc_drop)
        cond = hcond
    if arch == 'convnet_28':
        if size is None: size = 64
        h = nonlin(N.Conv(h, nout=size* 1, ksize=5, stride=2))
        h =     conv_acts(h, nout=size* 2, ksize=5, stride=2)
        h =   acts(  N.FC(h, nout=size*16))
    elif arch == 'mnist_mlp':
        if size is None: size = 64
        h = nonlin(N.FC(h, nout=size*16))
        # h =   acts(N.FC(h, nout=size*16))
        h =   acts(N.FC(h, nout=size*16))
    elif arch == 'atari_pad':
        if size is None: size = 16
        h = nonlin(N.Conv(h, nout=size*1, ksize=8, stride=4, pad=4))
        h =     conv_acts(h, nout=size*2, ksize=4, stride=2, pad=2)
        h =   acts(  N.FC(h, nout=size*16))
    elif arch == 'atari':
        if size is None: size = 16
        h = nonlin(N.Conv(h, nout=size*1, ksize=8, stride=4, pad=0))
        h =     conv_acts(h, nout=size*2, ksize=4, stride=2, pad=0)
        h =   acts(  N.FC(h, nout=size*16))
    elif arch == 'atari_bnc1':
        if size is None: size = 16
        h = conv_acts(h, nout=size*1, ksize=8, stride=4, pad=0)
        h = conv_acts(h, nout=size*2, ksize=4, stride=2, pad=0)
        h = acts(N.FC(h, nout=size*16))
    elif arch == 'convnet_64':
        if size is None: size = 128
        h = nonlin(N.Conv(h, nout=size*1, ksize=5, stride=2))
        h =     conv_acts(h, nout=size*2, ksize=5, stride=2)
        h =     conv_acts(h, nout=size*4, ksize=5, stride=2)
        h =     conv_acts(h, nout=size*8, ksize=5, stride=2)
    elif arch == 'convnet_64_k4':
        if size is None: size = 128
        h = nonlin(N.Conv(h, nout=size*1, ksize=4, stride=2))
        h =     conv_acts(h, nout=size*2, ksize=4, stride=2)
        h =     conv_acts(h, nout=size*4, ksize=4, stride=2)
        h =     conv_acts(h, nout=size*8, ksize=4, stride=2)
    elif arch in ('convnet_96', 'convnet_128'):
        if size is None: size = 128
        h = nonlin(N.Conv(h, nout=size*1, ksize=5, stride=2))
        h =     conv_acts(h, nout=size*2, ksize=5, stride=2)
        h =     conv_acts(h, nout=size*4, ksize=5, stride=2)
        h =     conv_acts(h, nout=size*6, ksize=5, stride=2)
        h =     conv_acts(h, nout=size*8, ksize=5, stride=2)
    elif arch == 'openai_impgan_discrim':
        if size is None: size = 64
        # from https://github.com/openai/improved-gan/blob/master/imagenet/discriminator.py#L49
        h = conv_acts(h, nout=size   , ksize=3, stride=2)  # d_h0_conv
        h = conv_acts(h, nout=size*2 , ksize=3, stride=2)  # d_h1_conv
        h = conv_acts(h, nout=size*4 , ksize=3, stride=2)  # d_h2_conv
        h = conv_acts(h, nout=size*4 , ksize=3, stride=1)  # d_h3_conv
        h = conv_acts(h, nout=size*4 , ksize=3, stride=1)  # d_h4_conv
        h = conv_acts(h, nout=size*8 , ksize=3, stride=2)  # d_h5_conv
        h = conv_acts(h, nout=size*8 , ksize=3, stride=1)  # d_h6_conv
        h = acts(N.FC(h, nout=size*40))                    # d_h7
    elif arch == 'alexnet_group_padpool':
        # slightly broken version of architecture which pads conv1 & pool inputs
        h = nonlin(N.Conv(h, nout= 96, ksize=11, stride=4))                 # conv1
        h =        N.Pool(h,           ksize= 3, stride=2)                  # pool1
        h =     conv_acts(h, nout=256, ksize= 5, stride=1, pad=2, group=2)  # conv2
        h =        N.Pool(h,           ksize= 3, stride=2)                  # pool2
        h =     conv_acts(h, nout=384, ksize= 3, stride=1, pad=1)           # conv3
        h =     conv_acts(h, nout=384, ksize= 3, stride=1, pad=1, group=2)  # conv4
        h =     conv_acts(h, nout=256, ksize= 3, stride=1, pad=1, group=2)  # conv5
    elif arch == 'alexnet_group':
        h = nonlin(N.Conv(h, nout= 96, ksize=11, stride=4, pad=0))          # conv1
        h =        N.Pool(h,           ksize= 3, stride=2, pad=0)           # pool1
        h =     conv_acts(h, nout=256, ksize= 5, stride=1, pad=2, group=2)  # conv2
        h =        N.Pool(h,           ksize= 3, stride=2, pad=0)           # pool2
        h =     conv_acts(h, nout=384, ksize= 3, stride=1, pad=1)           # conv3
        h =     conv_acts(h, nout=384, ksize= 3, stride=1, pad=1, group=2)  # conv4
        h =     conv_acts(h, nout=256, ksize= 3, stride=1, pad=1, group=2)  # conv5
    elif arch == 'alexnet_group_bnc1':
        h =     conv_acts(h, nout= 96, ksize=11, stride=4, pad=0)           # conv1
        h =        N.Pool(h,           ksize= 3, stride=2, pad=0)           # pool1
        h =     conv_acts(h, nout=256, ksize= 5, stride=1, pad=2, group=2)  # conv2
        h =        N.Pool(h,           ksize= 3, stride=2, pad=0)           # pool2
        h =     conv_acts(h, nout=384, ksize= 3, stride=1, pad=1)           # conv3
        h =     conv_acts(h, nout=384, ksize= 3, stride=1, pad=1, group=2)  # conv4
        h =     conv_acts(h, nout=256, ksize= 3, stride=1, pad=1, group=2)  # conv5
    elif arch == 'alexnet':
        h = nonlin(N.Conv(h, nout= 96, ksize=11, stride=4, pad=0)) # conv1
        h =        N.Pool(h,           ksize= 3, stride=2, pad=0)  # pool1
        h =     conv_acts(h, nout=256, ksize= 5, stride=1, pad=2)  # conv2
        h =        N.Pool(h,           ksize= 3, stride=2, pad=0)  # pool2
        h =     conv_acts(h, nout=384, ksize= 3, stride=1, pad=1)  # conv3
        h =     conv_acts(h, nout=384, ksize= 3, stride=1, pad=1)  # conv4
        h =     conv_acts(h, nout=256, ksize= 3, stride=1, pad=1)  # conv5
    elif arch == 'alexnet_group_plus_convnet_64':
        hin = h
        # alexnet
        h = nonlin(N.Conv(hin, nout= 96, ksize=11, stride=4))               # conv1
        h =        N.Pool(h,           ksize= 3, stride=2)                  # pool1
        h =     conv_acts(h, nout=256, ksize= 5, stride=1, pad=2, group=2)  # conv2
        h =        N.Pool(h,           ksize= 3, stride=2)                  # pool2
        h =     conv_acts(h, nout=384, ksize= 3, stride=1, pad=1)           # conv3
        h =     conv_acts(h, nout=384, ksize= 3, stride=1, pad=1, group=2)  # conv4
        h =     conv_acts(h, nout=256, ksize= 3, stride=1, pad=1, group=2)  # conv5
        halexnet = h
        # convnet_64
        if size is None: size = 128
        h = nonlin(N.Conv(hin, nout=size*1, ksize=5, stride=2))
        h =     conv_acts(h, nout=size*2, ksize=5, stride=2)
        h =     conv_acts(h, nout=size*4, ksize=5, stride=2)
        h =     conv_acts(h, nout=size*8, ksize=5, stride=2)
        hstandard = h
        # concat
        h = N.Concat(halexnet, hstandard)
    elif arch == 'zfnet':
        h = nonlin(N.Conv(h, nout= 96, ksize=7, stride=2, pad=3)) # conv1
        h =        N.Pool(h,           ksize=3, stride=2)         # pool1
        h =     conv_acts(h, nout= 96, ksize=5, stride=2, pad=2)  # conv2
        h =        N.Pool(h,           ksize=3, stride=2)         # pool2
        h =     conv_acts(h, nout=384, ksize=3, stride=1, pad=1)  # conv3
        h =     conv_acts(h, nout=384, ksize=3, stride=1, pad=1)  # conv4
        h =     conv_acts(h, nout=256, ksize=3, stride=1, pad=1)  # conv5
    elif arch == 'smallnet':
        # designed for 64x64 input
        h = nonlin(N.Conv(h, nout= 64, ksize=5, stride=2, pad=2))  # conv1 (32)
        h =     conv_acts(h, nout= 64, ksize=1, stride=1)          # conv1.1
        h =     conv_acts(h, nout=128, ksize=5, stride=2, pad=2)   # conv2 (16)
        h =     conv_acts(h, nout=128, ksize=1, stride=1)          # conv2.1
        h =     conv_acts(h, nout=192, ksize=5, stride=2, pad=2)   # conv3 (8)
        h =     conv_acts(h, nout=192, ksize=1, stride=1)          # conv3.1
        h =     conv_acts(h, nout=256, ksize=5, stride=2, pad=2)   # conv4 (4)
        h =     conv_acts(h, nout=256, ksize=1, stride=1)          # conv4.1
    else:
        raise ValueError('Unknown architecture: %s' % arch)
    fc_dims = [4096] * num_fc + fc_dims
    for dim in fc_dims:
        h = acts(N.FC(h, nout=dim))
        if fc_drop != 0:
            h = N.Dropout(h, ratio=fc_drop)
    if minibatch_layer_size is not None:
        assert len(minibatch_layer_size) == 3
        assert all(isinstance(n, int) and n > 0 for n in minibatch_layer_size)
        B, C, D = minibatch_layer_size
        total_size = B * C
        in_dim = np.prod(h.shape[1:])
        if len(h.shape) != 2:
            assert len(h.shape) > 2
            h = N.Reshape(h, shape=(-1, in_dim))
        hin = h
        h = N.FC(h, nout=total_size)  # -> N x (B*C)
        h = N.Reshape(h, shape=(-1, B, C))
        if minibatch_layer_halves:
            h_parts = N.Slice(h, num=2, axis=0)
        else:
            h_parts = h,
        diff_sums = []
        for h in h_parts:
            diff_sum = None
            hv = h.value
            for offset in xrange(1, D+1):
                hoff = T.concatenate([hv[offset:], hv[:offset]], axis=0)
                diff = abs(hv - hoff).sum(axis=2) # -> N x B (summed over C)
                diff = T.exp(-diff)
                if diff_sum is None:
                    diff_sum = diff
                else:
                    diff_sum += diff
            diff_sums.append(diff_sum)
        if len(diff_sums) == 1:
            diff_sum = diff_sums[0]
        else:
            diff_sum = T.concatenate(diff_sums, axis=0)
        diff_mean = diff_sum / D
        h = Output(T.concatenate([hin.value, diff_mean], axis=1),
                   shape=(hin.shape[0], in_dim + B))
    for dim in post_minibatch_layer_dims:
        h = acts(N.FC(h, nout=dim))
        if fc_drop != 0:
            h = N.Dropout(h, ratio=fc_drop)
    return h, N

def get_convnet(image_size=None, name=None):
    if name is None:
        assert image_size is not None
        name = 'convnet_%d' % image_size
    return partial(convnet, arch=name)

def get_deconvnet(image_size=None, name=None):
    if name is None:
        assert image_size is not None
        name = 'deconvnet_%d' % image_size
    return globals()[name]

def test_batch_norm(thresh=1e-8):
    b = 3
    nb = 50
    n = b * nb
    dim = 5
    shape = n, dim, 7, 4
    data = floatX(10 + 5 * np.random.rand(*shape))

    z = Output(T.tensor4(), shape=shape)
    N = Net()
    znormed = N.BatchNorm(z)
    f = theano.function([z.value], znormed.value,
                        updates=N.deploy_updates.items())

    outputs = [f(data[i:(i+b)]) for i in xrange(0, n, b)]
    output = np.concatenate(outputs, axis=0)
    thresh = 1e-6
    for i in xrange(dim):
        d, o = data[:, i], output[:, i]
        print 'Input: (mean, std) = (%f, %f)' % (d.mean(), d.std())
        print 'Output: (mean, std) = (%f, %f)' % (o.mean(), o.std())
        assert np.abs(o.mean()) < thresh
        assert np.abs(np.log(o.std())) < thresh

    ztest = Output(T.tensor4(), shape=znormed.shape)
    N2 = Net(source=N)
    znormedtest = N2.BatchNorm(ztest, use_ave=True)
    ftest = theano.function([ztest.value], znormedtest.value)
    # check that batching of inputs doesn't matter with use_ave=True
    output_batches = [ftest(data[i:(i+b)]) for i in xrange(0, n, b)]
    outputs = np.concatenate(output_batches, axis=0)
    output = ftest(data)
    assert np.all(outputs == output)
    thresh = 1e-2
    for i in xrange(dim):
        d, o = data[:, i], output[:, i]
        print 'Input: (mean, std) = (%f, %f)' % (d.mean(), d.std())
        print 'Output: (mean, std) = (%f, %f)' % (o.mean(), o.std())
        assert np.abs(o.mean()) < thresh
        assert np.abs(np.log(o.std())) < thresh

def test_dropout():
    n = 1000
    dim = 5
    shape = n, dim, 7, 4
    data = floatX(10 + 5 * np.random.rand(*shape))
    z = Output(T.tensor4(), shape=shape)
    zdropped = L.Dropout(z)
    ztest = L.Dropout(z, ratio=0)
    f = theano.function([z.value], zdropped.value)
    assert (data == 0).sum() == 0

    dropped_data = f(data)
    num_dropped = (dropped_data == 0).sum()
    proportion_dropped = num_dropped / float(data.size)
    assert np.abs(proportion_dropped - 0.5) < 1e-2
    kept_inds = np.where(dropped_data != 0)
    assert np.max(np.abs(data[kept_inds] * 2 - dropped_data[kept_inds])) < 1e-4

    dropped_data2 = f(data)
    num_dropped = (dropped_data * dropped_data2 == 0).sum()
    proportion_dropped = num_dropped / float(data.size)
    assert np.abs(proportion_dropped - 0.75) < 1e-2

    ftest = theano.function([z.value], ztest.value)
    test_data = ftest(data)
    num_dropped = (test_data == 0).sum()
    assert num_dropped == 0

def test_multifc(n=5, b=100, d_in=500, d_out=1000, n_trials=1000):
    x = [Output(T.matrix(), shape=(b, d_in)) for _ in xrange(n)]
    x_in = [xi.value for xi in x]
    x_sample = [np.asarray(np.random.rand(*xi.shape), dtype=xi.value.dtype)
                for xi in x]
    # method A: concat then multiply
    N_a = Net(name='A')
    x_cat = N_a.Concat(*x, axis=1)
    y_a = N_a.FC(x_cat, nout=d_out)
    f_a = theano.function(x_in, y_a.value)
    time_a = Timer(partial(f_a, *x_sample))
    # method B: multiply each one then sum results
    N_b = Net(name='B')
    ys = [N_b.FC(xi, nout=d_out) for xi in x]
    y_b = N_b.EltwiseSum(*ys)
    f_b = theano.function(x_in, y_b.value)
    time_b = Timer(partial(f_b, *x_sample))
    # time them
    print 'Time A:', time_a.timeit(number=n_trials)
    print 'Time B:', time_b.timeit(number=n_trials)

if __name__ == '__main__':
    test_multifc()
    test_dropout()
    test_batch_norm()
    images = Output(sharedX(np.zeros([128, 3, 28, 28])))
    D = min_convnet_28(images)[0]
    print 'D shape =', D.shape
    latents = Output(sharedX(np.zeros([128, 100])))
    G = min_deconvnet_28(latents)[0]
    print 'G shape =', G.shape
    birelu_G = L.BiReLU(G)
    print 'birelu_G shape =', birelu_G.shape
