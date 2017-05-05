import theano
import theano.sandbox.cuda.dnn as dnn

""" Trick Theano into thinking this is part of the DNN module (dnn.py).
    Otherwise won't find C code files expected to be in same directory. """
__file__ = dnn.__file__

class UnpoolWithGrad(dnn.GpuDnnPoolGrad):
    def connection_pattern(self, node):
        return [[0], [0], [1], [0], [0], [0]]

    def grad(self, inputs, grads):
        out, inp, inp_grad, ws, stride, pad = inputs
        mode = self.mode
        if mode not in ("average_inc_pad", "average_exc_pad"):
            raise NotImplementedError("Unsupported pooling mode for grad.")
        g_out = dnn.dnn_pool(grads[0], ws, stride=stride, pad=pad, mode=mode)
        def d(): return theano.gradient.DisconnectedType()()
        return d(), d(), g_out, d(), d(), d()

def dnn_upsample_nearest(img, factor):
    assert img.ndim == 4
    if isinstance(factor, int):
        factor = (factor, factor)
    assert (len(factor) == 2) and all(isinstance(f, int) for f in factor)
    img = dnn.gpu_contiguous(img)
    s = img.shape
    pool_in_shape = list(s[:2]) + [f * si for f, si in zip(factor, s[2:])]
    pool_in = dnn.gpu_alloc_empty(*pool_in_shape)
    pool_out = dnn.gpu_alloc_empty(*s)
    stride = factor
    pad = (0, 0)
    ret = UnpoolWithGrad(mode="average_inc_pad")(
        pool_in, pool_out, img, factor, stride, pad)
    window_elem = theano.tensor.prod(factor).astype(ret.dtype)
    return dnn.as_cuda_ndarray_variable(ret * window_elem)

if __name__ == '__main__':
    import numpy as np
    def upsample_23(x):
        return dnn_upsample_nearest(x, [2, 3])
    x_input = np.asarray(np.random.randn(3, 4, 5, 6),
                         dtype=theano.config.floatX)
    rng = np.random.RandomState()
    theano.tensor.verify_grad(upsample_23, [x_input], rng=rng)
    print 'Gradient check passed!'

    x = theano.tensor.tensor4()
    y = dnn_upsample_nearest(x, [2, 3])
    upsample2x = theano.function([x], y)

    x_input = np.array([
        [[[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]],
        [[[11, 12, 13],
          [14, 15, 16],
          [17, 18, 19]]],
    ], dtype=theano.config.floatX)
    result = upsample2x(x_input)
    print result
