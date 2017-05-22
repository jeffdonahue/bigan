from __future__ import division

import itertools
from operator import itemgetter

import numpy as np
import theano
import theano.tensor as T

from lib import inits
from lib.rng import t_rng
from lib.theano_utils import floatX, sharedX

import net
from net import L, Output

def castFloatX(x):
  return T.cast(x, theano.config.floatX)

def concat(x):
  x = list(x)
  if len(x) == 0:
    return None
  if len(x) == 1:
    return x[0]
  return T.concatenate(x, axis=1)

class Distribution(object):
  placeholders = []

  def __init__(self):
    raise NotImplementedError()

  def sample(self, num=None):
    """Returns samples from the distribution of arbitrary type T."""
    raise NotImplementedError()

  def recon_error(self, recon_logits):
    """Returns a computed reconstruction loss given 'recon_logits',
       a predicted reconstruction directly output from a linear layer."""
    raise NotImplementedError()

  def logits_to_recon(self, recon_logits):
    """Returns the nearest possible `sample()` output from `recon_logits`."""
    return [Output(recon_logits, (self.num, self.recon_dim))]

  def logits_to_sample(self, recon_logits):
    """Returns a sample output from `recon_logits`."""
    return self.logits_to_recon(recon_logits), \
        T.zeros([recon_logits.shape[0]], dtype=theano.config.floatX)

  def kl_divergence(self, recon_logits):
    raise NotImplementedError()

  def l2distable(self, recon):
    return recon

  def norm_divisor(self):
    return self.sample_dim

class UniformDistribution(Distribution):
  nickname = 'u'
  default_limits = (-1, +1)

  def __init__(self, num, definition, limits=default_limits,
               internal_rng=False, name=None):
    assert len(limits) == 2
    assert limits[1] > limits[0]
    self.limits = tuple(float(l) for l in limits)
    self.span = limits[1] - limits[0]
    if len(definition) != 1:
      raise ValueError('definition should have 1 parameter (dim), not %d'
                       % len(definition))
    try:
      dim = int(definition[0])
    except ValueError:
      raise ValueError('non-integer dim: %s' % dim)
    self.recon_dim = self.sample_dim = dim
    self.num = num
    self.rangekw = dict(low=self.limits[0], high=self.limits[1])
    if internal_rng:
      self.placeholders = [t_rng.uniform(size=(num, dim), **self.rangekw)]
    else:
      self.placeholders = [T.matrix()]
    self.flat_data = [Output(self.placeholders[0], shape=(self.num, dim))]

  def sample(self, num=None):
    if num is None: num = self.num
    return [floatX(np.random.uniform(size=(num, self.sample_dim),
                                     **self.rangekw))]

  def recon_error(self, recon):
    sample = self.flat_data[0].value
    if self.limits[0] != 0:
      sample -= self.limits[0]
    if self.span != 1:
      sample /= self.span
    recon = T.nnet.sigmoid(recon)
    axes = range(1, recon.ndim)
    return T.nnet.binary_crossentropy(recon, sample).sum(axis=axes)

  def _scale_and_shift(self, recon, input_range):
    # Shift and scale recon from the given input_range to self.limits.
    # input range  (input_range): x, y
    # output range (self.limits): a, b
    # input point z, output point c
    # t := (recon - x) / (y - x) -> t \in [0, 1]
    # c := t * (b - a) + a
    #    = ((recon - x) / (y - x)) * (b - a) + a
    #    = ((b - a) / (y - x)) * (recon - x) + a
    x, y = input_range
    a, b = self.limits
    if x != 0:
      recon -= x
    scale = (b - a) / (y - x)
    if scale != 1:
      recon *= scale
    if a != 0:
      recon += a
    return recon

  def logits_to_recon(self, recon_logits):
    shape = self.num, self.recon_dim
    recon = T.nnet.sigmoid(recon_logits)
    limits = 0, 1
    return [Output(self._scale_and_shift(recon, limits), shape)]

class UniformDistributionL2Error(UniformDistribution):
  nickname = 'ul'

  def recon_error(self, recon):
    sample = self.flat_data[0].value
    axes = range(1, recon.ndim)
    return 0.5 * ((recon - sample) ** 2).sum(axis=axes)

  def logits_to_recon(self, recon_logits):
    shape = self.num, self.recon_dim
    return [Output(recon_logits, shape)]

class UniformDistributionClampedL2Error(UniformDistributionL2Error):
  nickname = 'ulc'

  def logits_to_recon(self, recon_logits):
    def blend(mask, true, false):
      return mask * true + (1 - mask) * false
    shape = self.num, self.recon_dim
    # clamp recon in the dist. range
    recon = recon_logits
    recon = blend(recon < self.limits[0], self.limits[0], recon)
    recon = blend(recon > self.limits[1], self.limits[1], recon)
    return [Output(recon, shape)]

class UniformDistributionTanHL2Error(UniformDistributionL2Error):
  nickname = 'ut'

  def recon_error(self, recon):
    if self.limits != (-1, 1):
      raise NotImplementedError
    recon = T.tanh(recon)
    return super(UniformDistributionTanHL2Error, self).recon_error(recon)

  def logits_to_recon(self, recon_logits):
    shape = self.num, self.recon_dim
    recon = T.tanh(recon_logits)
    limits = -1, 1
    return [Output(self._scale_and_shift(recon, limits), shape)]

class GaussianDistribution(Distribution):
  nickname = 'g'

  def __init__(self, num, definition, mean=0, stdev=None, internal_rng=False):
    self.mean = mean
    if len(definition) != 1:
      raise ValueError('definition should have 1 parameter (dim), not %d'
                       % len(definition))
    try:
      dim = int(definition[0])
    except ValueError:
      raise ValueError('non-integer dim: %s' % dim)
    if stdev is None:
      var = 2 * np.log(2)
      stdev = var ** 0.5
    else:
      var = stdev ** 2
    self.var, self.stdev = (floatX(x) for x in (var, stdev))
    self.recon_dim = self.sample_dim = dim
    self.num = num
    if internal_rng:
      self.placeholders = [t_rng.normal(size=(num, dim),
                                        avg=mean, std=self.stdev)]
    else:
      self.placeholders = [T.matrix()]
    self.flat_data = [Output(self.placeholders[0], shape=(num, dim))]

  def sample(self, num=None):
    if num is None: num = self.num
    return [floatX(np.random.normal(loc=self.mean, scale=self.stdev,
                                    size=(num, self.sample_dim)))]

  def recon_error(self, recon):
    axes = range(1, recon.ndim)
    return 0.5 * ((recon - self.flat_data[0].value) ** 2).sum(axis=axes)

  def kl_divergence(self, recon):
    assert self.mean == 0, 'not implemented for non-zero mean'
    return 0.5 * (recon ** 2).sum()

class GaussianReconVarDistribution(GaussianDistribution):
  nickname = 'gv'

  def __init__(self, *args, **kwargs):
    super(GaussianReconVarDistribution, self).__init__(*args, **kwargs)
    self.slice_point = self.recon_dim
    self.recon_dim *= 2
    self.log_var_bias = 0

  def recon_error(self, recon_logits):
    sample = self.flat_data[0].value
    recon_mean = recon_logits[:, :self.slice_point]
    recon_log_var = recon_logits[:, self.slice_point:]
    if self.log_var_bias != 0:
      recon_log_var += self.log_var_bias
    recon_var = T.exp(recon_log_var)
    # compute the negative log likelihood of the sample under recon_logits
    nll = (recon_log_var + (T.sqr(recon_mean - sample) / recon_var)) / 2
    return nll.sum(axis=1)

  def kl_divergence(self, recon_logits):
    assert self.mean == 0, 'not implemented for non-zero mean'
    recon_mean = recon_logits[:, :self.slice_point]
    recon_log_var = recon_logits[:, self.slice_point:]
    if self.log_var_bias != 0:
      recon_log_var += self.log_var_bias
    recon_var = T.exp(recon_log_var)
    mean_term = (T.sqr(recon_mean) / recon_var).sum(axis=1)
    var_term = recon_var.sum(axis=1)
    log_var_term = recon_log_var.sum(axis=1)
    return (mean_term + var_term - log_var_term) / 2

  def logits_to_recon(self, recon_logits):
    recon_mean = recon_logits[:, :self.slice_point]
    return [Output(recon_mean, (self.num, self.sample_dim))]

  def logits_to_sample(self, recon_logits):
    recon_mean = recon_logits[:, :self.slice_point]
    recon_log_var = recon_logits[:, self.slice_point:]
    if self.log_var_bias != 0:
      recon_log_var += self.log_var_bias
    recon_logstd = recon_log_var / 2
    recon_std = T.exp(recon_logstd)
    standard_sample = t_rng.normal(size=recon_mean.shape)
    sample = recon_mean + standard_sample * recon_std
    sample = [Output(sample, (self.num, self.sample_dim))]
    # return zeros as logprob since we reparameterized
    return sample, T.zeros([recon_logits.shape[0]], dtype=theano.config.floatX)

class GaussianReconVarSampleDistribution(GaussianReconVarDistribution):
  nickname = 'gvs'

  def logits_to_sample(self, recon_logits):
    recon_mean = recon_logits[:, :self.slice_point]
    recon_log_var = recon_logits[:, self.slice_point:]
    if self.log_var_bias != 0:
      recon_log_var += self.log_var_bias
    recon_var = T.exp(recon_log_var)
    recon_logstd = recon_log_var / 2
    recon_std = T.exp(recon_logstd)
    standard_sample = t_rng.normal(size=recon_mean.shape)
    sample = recon_mean + standard_sample * recon_std
    sample = theano.gradient.disconnected_grad(sample)
    sample_log_prob = recon_log_var + T.sqr(sample - recon_mean) / recon_var
    sample_log_prob /= floatX(-2)
    sample = [Output(sample, (self.num, self.sample_dim))]
    return sample, sample_log_prob.sum(axis=range(1, sample_log_prob.ndim))

class CategoricalDistribution(Distribution):
  nickname = 'c'

  def __init__(self, num, definition, internal_rng=False):
    assert not internal_rng, 'not implemented'
    if len(definition) != 2:
      raise ValueError('definition should have 2 parameters (dim, cats), not %d'
                       % len(definition))
    dim, cats = definition
    try:
      self.dim = int(dim)
    except ValueError:
      raise ValueError('non-integer dim: %s' % dim)
    try:
      self.cats = int(cats)
    except ValueError:
      raise ValueError('non-integer #cats: %s' % cats)
    if self.cats < 2:
      raise ValueError('#cats must be >= 2, not %d' % cats)
    self.recon_dim = self.sample_dim = self.dim * self.cats
    # don't use uint types or int8/int16;
    # T.nnet.categorical_crossentropy doesn't accept them
    self.tcast = None
    for t in self.allowed_types():
      self.dtype = t
      self.np_index_type = getattr(np, t)
      if self.cats < np.iinfo(self.np_index_type).max:
        self.tcast = lambda x: T.cast(x, t)
        break
    if self.tcast is None:
      raise ValueError('#cats too large: %d' % self.cats)
    self.num = num
    # set normalization divisor so result has same initial expected entropy
    # as 1D Bernoulli
    self.norm_div = self.dim * (np.log(self.cats) / np.log(2))
    # placeholders: shape (N, D) of integer indices
    self.placeholders = [T.vector(dtype=self.dtype) for _ in xrange(self.dim)]
    self.flat_data = [Output(p, shape=(self.num,), index_max=self.cats)
                      for p in self.placeholders]

  def allowed_types(self):
    return 'int32', 'int64'

  def sample(self, num=None):
    if num is None:
      num = self.num
    return [np.random.randint(self.cats, size=num, dtype=self.np_index_type)
            for i in xrange(self.dim)]

  def recon_error(self, recon, sample=None):
    if sample is None:
      sample = self.placeholders
    # reshape from [num, dim*cats] to [num, dim, cats]
    assert recon.ndim == 2
    recon = recon.reshape([-1, self.dim, self.cats])
    recon = [recon[:, i] for i in xrange(self.dim)]
    assert len(sample) == len(recon) == self.dim
    error = 0
    for s, r in zip(sample, recon):
      if not any(s.dtype.startswith(t) for t in ['int', 'uint']):
        raise TypeError('sample must be int indices')
      assert s.ndim == 1
      r = T.nnet.softmax(r)
      crossent = T.nnet.categorical_crossentropy(r, s)
      error += crossent.sum(axis=range(1, crossent.ndim))
    return error

  def _to_recon(self, data):
    full = self.tcast(data)
    p = Output(full, shape=(self.num, self.dim), index_max=self.cats)
    if self.dim == 1:
      slices = [p]
    else:
      slices = L.Slice(p, axis=1, num=self.dim)
    slices = [L.Reshape(s, shape=[-1]) for s in slices]
    return slices

  def logits_to_recon(self, recon_logits):
    data = recon_logits.reshape([-1, self.dim, self.cats]).argmax(axis=2)
    return self._to_recon(data)

  def logits_to_sample(self, recon_logits):
    probs = T.nnet.softmax(recon_logits.reshape([-1, self.cats]))
    sample = t_rng.multinomial(pvals=probs).argmax(axis=1)
    sample = theano.gradient.disconnected_grad(sample)
    sample = sample.reshape([-1, self.dim])
    log_prob = castFloatX(T.nnet.categorical_crossentropy(probs, sample))
    return self._to_recon(sample), log_prob

  def kl_divergence(self, recon):
    log_probs = T.nnet.logsoftmax(recon.reshape([-1, self.cats]))
    return log_probs.mean(axis=1).reshape([-1, self.dim]).sum(axis=1)

  def l2distable(self, recon):
    # recon.shape == (N, DC); reshape to (N, D, C)
    recon = recon.reshape([-1, self.dim, self.cats])
    # argmax along axis 2 -> shape (N, D)
    recon = recon.argmax(axis=2)
    # flatten from (N, D) to (ND)
    recon = recon.reshape([-1])
    # self.one_hots has shape(C, C), index by recon -> shape (ND, C)
    recon = castFloatX(L.OneHot(Output(recon, shape=(self.num,),
                                       index_max=self.cats)).value)
    # reshape from (ND, C) back to (N, DC)
    return recon.reshape([-1, self.dim * self.cats])

  def norm_divisor(self):
    return self.norm_div

class CategoricalFlatDistribution(CategoricalDistribution):
  nickname = 'cf'

  def __init__(self, *args, **kwargs):
    super(CategoricalFlatDistribution, self).__init__(*args, **kwargs)
    holder = T.matrix()
    self.placeholders = [holder]
    self.flat_data = [Output(holder, shape=(self.num, self.sample_dim))]
    self.one_hots = np.identity(self.cats, dtype=theano.config.floatX)

  def allowed_types(self):
    return 'int8', 'int16', 'int32', 'int64'

  def sample(self, num=None):
    if num is None:
      num = self.num
    total_num = num * self.dim
    sample_indices = np.random.randint(self.cats, size=total_num,
                                       dtype=self.np_index_type)
    one_hot_sample = self.one_hots[sample_indices]
    return [one_hot_sample.reshape([num, self.sample_dim])]

  def _to_recon(self, data):
    data = castFloatX(data)
    return [Output(data, shape=(self.num, self.sample_dim))]

  def logits_to_recon(self, recon_logits):
    shape = self.num, self.dim, self.cats
    argmaxes = recon_logits.reshape(shape).argmax(axis=2)
    shape = self.num * self.dim,
    flat_argmaxes = argmaxes.reshape(shape)
    flat_argmaxes_out = Output(flat_argmaxes, shape=shape, index_max=self.cats)
    one_hot_recon = L.OneHot(flat_argmaxes_out)
    recon = L.Reshape(one_hot_recon, shape=[-1, self.sample_dim]).value
    return self._to_recon(recon)

  def logits_to_sample(self, recon_logits):
    recon_logits = recon_logits.reshape([-1, self.cats])
    probs = T.nnet.softmax(recon_logits)
    sample_one_hot = t_rng.multinomial(pvals=probs)
    sample_one_hot = theano.gradient.disconnected_grad(sample_one_hot)
    log_prob = castFloatX(T.nnet.categorical_crossentropy(probs, sample_one_hot))
    assert log_prob.ndim == 1
    log_prob = log_prob.reshape([-1, self.dim]).sum(axis=1)
    sample_one_hot = sample_one_hot.reshape([-1, self.sample_dim])
    return self._to_recon(sample_one_hot), log_prob

class CategoricalTemperatureDistribution(CategoricalDistribution):
  nickname = 'ct'

  def __init__(self, *args, **kwargs):
    super(CategoricalTemperatureDistribution, self).__init__(*args, **kwargs)
    self.recon_dim += self.dim
    self.log_temp_bias = 0

  def logits_to_sample(self, recon_logits):
    recon_logits, log_temps = recon_logits[:, :-self.dim], recon_logits[:, -self.dim:]
    if self.log_temp_bias != 0:
      log_temps += self.log_temp_bias
    temps = T.exp(log_temps).dimshuffle(0, 1, 'x')
    recon_logits = recon_logits.reshape([-1, self.dim, self.cats])
    softmax_input = temps * recon_logits
    return super(CategoricalTemperatureDistribution, self).logits_to_sample(softmax_input)

  def logits_to_recon(self, recon_logits):
    recon_logits = recon_logits[:, :-self.dim]
    return super(CategoricalTemperatureDistribution, self).logits_to_recon(recon_logits)

  def l2distable(self, recon):
    recon = recon[:, :-self.dim]
    return super(CategoricalTemperatureDistribution, self).l2distable(recon)

class PseudoGradientOp(theano.Op):
  """
  An operation on two inputs with the same shape: a and b.

  The output y is b itself, but when computing the gradient,
  we pretend dy/db = 0 and dy/da = 1.
  """
  __props__ = ()
  view_map = {0: [1]}

  def make_node(self, *args):
    assert len(args) == 2
    inputs = [theano.tensor.as_tensor_variable(a) for a in args]
    return theano.Apply(self, inputs, [inputs[1].type()])

  def connection_pattern(self, node):
    return [[1], [0]]

  def perform(self, node, inputs, output_storage):
    output_storage[0][0] = inputs[1]

  def infer_shape(self, node, i0_shapes):
    return [i0_shapes[1]]

  def grad(self, inputs, output_grads):
    return output_grads[0], theano.gradient.DisconnectedType()()

pseudo_gradient = PseudoGradientOp()

class CategoricalStraightThroughDistribution(CategoricalFlatDistribution):
  nickname = 'cs'

  def _op_input(self, recon_logits, probs):
    return recon_logits

  def _op_result(self, recon_logits, probs, op_input):
    return t_rng.multinomial(pvals=probs)

  def logits_to_sample(self, recon_logits):
    recon_logits = recon_logits.reshape([-1, self.cats])
    probs = T.nnet.softmax(recon_logits)
    op_input = self._op_input(recon_logits, probs)
    op_result = self._op_result(recon_logits, probs, op_input)
    sample = pseudo_gradient(op_input, op_result)
    sample = sample.reshape([-1, self.sample_dim])
    return self._to_recon(sample), \
        T.zeros([sample.shape[0]], dtype=theano.config.floatX)

class CategoricalSoftmaxStraightThroughDistribution(
    CategoricalStraightThroughDistribution):
  nickname = 'css'

  def _op_input(self, recon_logits, probs):
    return probs

class CategoricalHardThresholdDistribution(
    CategoricalStraightThroughDistribution):
  nickname = 'ch'

  def _op_result(self, recon_logits, probs, op_input):
    shape = self.num, self.dim, self.cats
    argmaxes = recon_logits.reshape(shape).argmax(axis=2)
    shape = self.num * self.dim,
    flat_argmaxes = argmaxes.reshape(shape)
    flat_argmaxes_out = Output(flat_argmaxes, shape=shape, index_max=self.cats)
    one_hot_recon = L.OneHot(flat_argmaxes_out)
    return L.Reshape(one_hot_recon, shape=[-1, self.sample_dim]).value

class CategoricalSoftmaxHardThresholdDistribution(
    CategoricalHardThresholdDistribution):
  nickname = 'chs'

  def _op_input(self, recon_logits, probs):
    return probs

def get_uniform_sample(shape):
  """ Use 1-U rather than U, since unifrom returns samples in [0, 1),
      but we want samples in (0, 1] as we'll be taking the log."""
  return 1 - t_rng.uniform(size=shape)

def get_neg_log_uniform_sample(shape):
  return -T.log(get_uniform_sample(shape))

def get_gumbel_sample(shape):
  return -T.log(get_neg_log_uniform_sample(shape))

class CategoricalGumbelSoftmaxStraightThroughDistribution(
    CategoricalStraightThroughDistribution):
  nickname = 'cgs'

  def __init__(self, *a, **k):
    self.temperature = 1
    return super(CategoricalGumbelSoftmaxStraightThroughDistribution,
                 self).__init__(*a, **k)

  def _op_input(self, recon_logits, probs):
    """ Stabler implementation? """
    logits = probs / get_neg_log_uniform_sample(probs.shape)
    if self.temperature != 1:
      logits **= (1 / self.temperature)
    return logits / T.sum(logits, axis=1, keepdims=True)
    """
    Alternate implementation, directly from Gumbel-Softmax paper (Jang et al.).
    """
    log_probs = T.nnet.logsoftmax(recon_logits)
    gumbel_sample = get_gumbel_sample(recon_logits.shape)
    logits = (log_probs + gumbel_sample).reshape([-1, self.cats])
    if self.temperature != 1:
      logits /= self.temperature
    return T.nnet.softmax(logits)

  def _op_result(self, recon_logits, probs, op_input):
    sample = op_input.argmax(axis=1).reshape([-1])
    sample_out = Output(sample, shape=(self.num * self.dim,),
                        index_max=self.cats)
    sample_one_hot = L.OneHot(sample_out)
    return L.Reshape(sample_one_hot, shape=[-1, self.cats]).value

class CategoricalGumbelSoftmaxSoftDistribution(
    CategoricalGumbelSoftmaxStraightThroughDistribution):
  nickname = 'cgsoft'

  def _op_result(self, recon_logits, probs, op_input):
    return op_input

class BernoulliDistribution(CategoricalDistribution):
  nickname = 'b'

  def __init__(self, num, definition, values=[0, 1], internal_rng=False):
    assert not internal_rng, 'not implemented'
    if len(definition) != 1:
      raise ValueError('definition should have 1 parameter (dim), not %d'
                       % len(definition))
    dim = definition[0]
    try:
      self.dim = int(dim)
    except ValueError:
      raise ValueError('non-integer dim: %s' % dim)
    self.cats = 2
    self.recon_dim = self.sample_dim = self.dim
    self.dtype = 'int8'
    self.np_index_type = getattr(np, self.dtype)
    self.tcast = lambda x: T.cast(x, self.dtype)
    self.num = num
    self.placeholders = [T.matrix()]
    assert len(values) == 2
    assert values[1] > values[0]
    self.values = [float(v) for v in values]
    self.value_range = self.values[1] - self.values[0]
    self.float_data = self.placeholders[0]
    if self.values == [0, 1]:
      flat_data = self.float_data
    else:
      iinfo = np.iinfo(self.np_index_type)
      if self.value_range != 1:
        assert iinfo.min <= 0 < self.value_range <= iinfo.max
      if self.values[0] != 0:
        assert iinfo.min <= self.values[0] < self.values[1] <= iinfo.max
      flat_data = self._rescale_binary_sample(self.placeholders[0])
    self.flat_data = [Output(flat_data, shape=(self.num, self.dim))]

  def _rescale_binary_sample(self, sample):
    if self.value_range != 1:
      sample *= self.value_range
    if self.values[0] != 0:
      sample += self.values[0]
    return sample

  def _descale_binary_sample(self, sample):
    if self.values[0] != 0:
      sample -= self.values[0]
    if self.value_range != 1:
      sample /= self.value_range
    return sample

  def sample(self, num=None):
    if num is None:
      num = self.num
    shape = num, self.dim
    sample = np.asarray(np.random.randint(2, size=shape, dtype=np.bool),
                        dtype=theano.config.floatX)
    return [self._rescale_binary_sample(sample)]

  def recon_error(self, recon, sample=None):
    if sample is None:
      sample = self.float_data
    sample = self._descale_binary_sample(sample)
    # sigmoid: labels are scalars in [0, 1]
    recon = T.nnet.sigmoid(recon)
    axes = range(1, recon.ndim)
    return T.nnet.binary_crossentropy(recon, sample).sum(axis=axes)

  def logits_to_recon(self, recon_logits):
    recon = castFloatX(recon_logits >= 0)
    rescaled_recon = self._rescale_binary_sample(recon)
    return [Output(rescaled_recon, (self.num, self.dim))]

  def logits_to_sample(self, recon_logits):
    probs = T.nnet.sigmoid(recon_logits)
    sample = castFloatX(probs > t_rng.uniform(probs.shape))
    log_prob = T.nnet.binary_crossentropy(probs, sample).sum(axis=1)
    rescaled_sample = self._rescale_binary_sample(sample)
    return [Output(rescaled_sample, (self.num, self.dim))], log_prob

  def kl_divergence(self, recon):
    even_dist = np.ones_like(recon) / self.cats
    return self.recon_error(recon, sample=even_dist)

  def l2distable(self, recon):
    # recon.shape == (N, D)
    return castFloatX(recon >= 0)

  def norm_divisor(self):
    return self.dim

use_stable_sigmoid = True
if use_stable_sigmoid:
  """
  Theano's native sigmoid gradient is computed as:

      dx := dy * y * (1 - y)
          = (dy * y) * (1 - y)

  In this version, we reverse the multiplication order to first multiply
  y by (1-y) before multiplication with dy:

      dx := dy * (y * (1 - y))

  This gives different results due to the non-associativity of floating point
  multiplication.
  """
  class ScalarSigmoidStableGrad(T.nnet.sigm.ScalarSigmoid):
      def grad(self, inp, grads):
          x, = inp
          gz, = grads
          y = scalar_sigmoid(x)
          rval = gz * (y * (1.0 - y))
          assert rval.type.dtype.find('float') != -1
          return [rval]
  scalar_sigmoid = ScalarSigmoidStableGrad(theano.scalar.upgrade_to_float,
                                           name='scalar_sigmoid')
  sigmoid = T.elemwise.Elemwise(scalar_sigmoid, name='sigmoid')
else:
  sigmoid = T.nnet.sigmoid

class BernoulliStraightThroughDistribution(BernoulliDistribution):
  nickname = 'bs'

  def _op_input(self, recon_logits, probs):
    return recon_logits

  def _op_result(self, recon_logits, probs, op_input):
    rand = t_rng.uniform(recon_logits.shape)
    return castFloatX(probs > rand)

  def logits_to_sample(self, recon_logits):
    probs = sigmoid(recon_logits)
    op_input = self._op_input(recon_logits, probs)
    op_result = self._op_result(recon_logits, probs, op_input)
    sample = pseudo_gradient(op_input, op_result)
    rescaled_sample = self._rescale_binary_sample(sample)
    return [Output(rescaled_sample, (self.num, self.dim))], \
        T.zeros([recon_logits.shape[0]], dtype=theano.config.floatX)

class BernoulliSigmoidStraightThroughDistribution(
    BernoulliStraightThroughDistribution):
  nickname = 'bss'

  def _op_input(self, recon_logits, probs):
    return probs

class BernoulliHardThresholdDistribution(
    BernoulliStraightThroughDistribution):
  nickname = 'bh'

  def _op_result(self, recon_logits, probs, op_input):
    return castFloatX(recon_logits > 0)

class BernoulliSigmoidHardThresholdDistribution(
    BernoulliHardThresholdDistribution):
  nickname = 'bhs'

  def _op_input(self, recon_logits, probs):
    return probs

def get_binary_gumbel_sample(shape):
  sample = get_gumbel_sample((2,) + tuple(shape))
  return sample[0] - sample[1]

class BernoulliGumbelSigmoidStraightThroughDistribution(
    BernoulliStraightThroughDistribution):
  nickname = 'bgs'

  def __init__(self, *a, **k):
    self.temperature = 1
    return super(BernoulliGumbelSigmoidStraightThroughDistribution,
                 self).__init__(*a, **k)

  def _op_input(self, recon_logits, probs):
    """ Stabler implementation? """
    if False:  # disable for now, seems broken somehow?
      nlu_sample = get_neg_log_uniform_sample((2,) + tuple(recon_logits.shape))
      pos = (1 + T.exp( recon_logits)) * nlu_sample[0]
      neg = (1 + T.exp(-recon_logits)) * nlu_sample[1]
      if self.temperature != 1:
        inv_temp = 1 / self.temperature
        pos, neg = (a ** inv_temp for a in (pos, neg))
      return pos / (pos + neg)
    """ Straightforward implementation """
    gumbel_sample = get_binary_gumbel_sample(probs.shape)
    logits = recon_logits + gumbel_sample
    if self.temperature != 1:
      logits /= self.temperature
    return sigmoid(logits)

  def _op_result(self, recon_logits, probs, op_input):
    return castFloatX(op_input > 0.5)

class BernoulliGumbelSigmoidSoftDistribution(
    BernoulliGumbelSigmoidStraightThroughDistribution):
  nickname = 'bgsoft'

  def _op_result(self, recon_logits, probs, op_input):
    return op_input

def chain(args):
  return list(itertools.chain(*args))

class MultiDistribution(Distribution):
  known_dists = frozenset(d for d in globals().values() if
                          type(d) == type and issubclass(d, Distribution) and
                          hasattr(d, 'nickname'))
  dist_nickname_to_type = {d.nickname: d for d in known_dists}
  assert len(known_dists) == len(dist_nickname_to_type), 'duplicate nicknames'

  def __init__(self, num, definition, uniform_error='scel',
               weights=None, normalize=True, weight_embed=True,
               internal_rng=False):
    self.internal_rng = internal_rng
    self.num = num
    self.dists = self._parse_def(definition)
    self.sample_dims = [d.sample_dim for d in self.dists]
    self.sample_dim = sum(self.sample_dims)
    self.recon_dims = [d.recon_dim for d in self.dists]
    self.recon_dim = sum(self.recon_dims)
    if not internal_rng:
      def cat_placeholders(a, b):
        bp = b.placeholders
        if not isinstance(bp, list):
          bp = [bp]
        return a + bp
      self.placeholders = reduce(cat_placeholders, self.dists, [])

    if weights is None:
      weights = [1] * len(self.dists)
    assert len(weights) == len(self.dists), \
        'weights length must be the # of dists'
    assert not any(w < 0 for w in weights), \
        'weights must have no negative entries'
    self.weights = weights
    if normalize:
      # divide each weight by corresponding norm_divisor; rescale to sum to 1
      sum_weights = sum(weights)
      assert sum_weights > 0, 'weights must have at least one nonzero entry'
      self.weights = [float(w) / sum_weights / d.norm_divisor()
                      for d, w in zip(self.dists, weights)]
    self.weight_embed = weight_embed
    self.multisample = 1
    self.enable_multisample = [True] * len(self.dists)

  def _parse_def(self, definition):
    """Returns a list of Distributions from definition,
       the string specification."""
    dists = []
    for dist_def in definition.split('_'):
      dist_def = dist_def.strip()
      if not dist_def: continue
      params = dist_def.split('-')
      dist_nickname, dist_params = params[0], params[1:]
      try:
        dist_type = self.dist_nickname_to_type[dist_nickname]
      except KeyError:
        mapping = self.dist_nickname_to_type.items()
        mapping.sort(key=itemgetter(0))
        known_nicknames = "\n".join('\t{}\t{}'.format(nick, t.__name__)
                                    for nick, t in mapping)
        e = 'Unknown Distribution nickname "{}". Known Distributions:\n{}'
        print e.format(dist_nickname, known_nicknames)
        raise
      dist = dist_type(self.num, dist_params, internal_rng=self.internal_rng)
      dists.append(dist)
    return dists

  def sample(self, num=None):
    return chain([d.sample(num=num) for d in self.dists])

  def sample_feed_dict(self, sample=None):
    if sample is None:
      sample = self.sample()
    return dict(zip(self.placeholders, sample))

  def recon_slice(self, recon):
    recon_slices = []
    offset = 0
    for dim in self.recon_dims:
      next_offset = offset + dim
      recon_slices.append(recon[:, offset : next_offset])
      offset = next_offset
    return recon_slices

  def _apply_slices(self, agg, op_name, recon, skip_unimplemented=False):
    results = []
    for d, r in zip(self.dists, self.recon_slice(recon)):
      try:
        op = getattr(d, op_name)
      except NotImplementedError:
        if skip_unimplemented:
          print 'Warning: op "%s" not implemented for ' \
                'distribution: %s; skipping' % (op_name, d)
          continue
        else:
          raise
      results.append(op(r))
    return agg(results)

  def _cat_slices(self, *a, **k):
    def cat(slices):
      return concat(slices)
    return self._apply_slices(cat, *a, **k)

  def _list_slices(self, *a, **k):
    def identity(x): return x
    return self._apply_slices(identity, *a, **k)

  def _sum_slices(self, *a, **k):
    return self._apply_slices(sum, *a, **k)

  def _weighted_sum_slices(self, *a, **k):
    def weighted_sum(args):
      assert len(self.weights) == len(args)
      return sum([(a if w==1 else w*a) for a, w in zip(args, self.weights)])
    return self._apply_slices(weighted_sum, *a, **k)

  def recon_error(self, recon):
    return self._sum_slices('recon_error', recon)

  def weighted_recon_error(self, recon):
    return self._weighted_sum_slices('recon_error', recon)

  def l2distable(self, recon):
    return self._cat_slices('l2distable', recon)

  def kl_divergence(self, recon):
    return self._sum_slices('kl_divergence', recon)

  def logits_to_recon(self, recon_logits):
    """
    logits_to_recon converts a net's linear predictions (recon_logits)
    to the corresponding max-likelihood sample
    (with the same type and shapes as the output of self.sample).
    """
    return chain(self._list_slices('logits_to_recon', recon_logits))

  def logits_to_sample(self, recon_logits):
    """
    logits_to_recon converts a net's linear predictions (recon_logits)
    to a sample from the distribution implied by them
    (with the same type and shapes as the output of self.sample).
    """
    assert isinstance(self.multisample, int)
    assert self.multisample > 0
    assert len(self.enable_multisample) == len(self.dists)
    multisamples = []
    log_probs = []
    for sample_index in xrange(self.multisample):
      outs = self._list_slices('logits_to_sample', recon_logits)
      if sample_index == 0:
        first_outs = outs
      else:
        outs = [o if e else f
                for e, o, f in zip(self.enable_multisample, outs, first_outs)]
      multisamples.append(chain([o for o, _ in outs]))
      log_probs.append(sum(lp for _, lp in outs))
    if self.multisample == 1:
      return multisamples[0], log_probs[0]
    catted_samples = []
    for multisample_comp in zip(*multisamples):
      catted_samples.append(L.Concat(*multisample_comp, axis=0))
    catted_log_probs = T.concatenate(log_probs, axis=0)
    return catted_samples, catted_log_probs

  def norm_divisor(self):
    return sum(d.norm_divisor() for d in self.dists)

  def embed_data(self, include_dists=None):
    if include_dists is None:
      include_dists = [True] * len(self.dists)
    assert len(include_dists) == len(self.dists)
    weights = self.weights if self.weight_embed else ([1] * len(self.weights))
    dists = zip(include_dists, self.dists, weights)
    h = chain([d.flat_data for i, d, _ in dists if i])
    return h, self.dist_sized_slices(weights)

  def dist_sized_slices(self, input_list):
    assert len(input_list) == len(self.dists)
    return chain([[i] * len(d.flat_data)
                 for i, d in zip(input_list, self.dists)])
