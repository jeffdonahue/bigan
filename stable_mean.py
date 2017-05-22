import numpy as np

def canonical_axis(x, axis):
    def is_int_ndarray(x):
        if not isinstance(x, np.ndarray):
            return False
        dtype = str(x.dtype)
        prefixes = 'int', 'uint'
        return any(dtype.startswith(p) for p in prefixes)
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int) or (is_int_ndarray(axis) and axis.ndim == 0):
        axis = [int(axis)]
    else:
        axis = list(axis)
    if len(axis) != len(set(axis)):
        raise ValueError('duplicate axis')
    for i, a in enumerate(axis):
        if not isinstance(a, int):
            raise TypeError('non-integer axis')
        if (-x.ndim <= a <= -1):
            axis[i] = a = a + x.ndim
        if not (0 <= a < x.ndim):
            raise ValueError('out of range axis')
    axis.sort()
    return axis

def big_mean(x, axis=None):
    """
    Computes the mean of x by first dividing it by n, then computing the sum.
    This is less efficient than dividing the sum by n -- what numpy does --
    but may avoid computing a mean of +/-inf if computing the sum of x results
    in overflow:

        x = np.finfo('float64').max / 2
        x_tiled = np.tile(x, 10)
        print x                 # 8.98846567431e+307
        print np.mean(x_tiled)  # inf
        print big_mean(x_tiled) # 8.98846567431e+307

    On the other hand, the sum-then-divide method is more precise in the regime
    of small values:

        x = np.nextafter(0, 1) * 2
        x_tiled = np.tile(x, 10)
        print x                 # 9.8813129168249309e-324
        print np.mean(x_tiled)  # 9.8813129168249309e-324
        print big_mean(x_tiled) # 0.0
    """
    axis = canonical_axis(x, axis)
    x = np.array(x)
    x /= float(np.asarray([x.shape[a] for a in axis]).prod())
    return x.sum(axis=tuple(axis))

def flat_stable_mean(x):
    """
    Partition x, a flat ndarray, into two parts: a "high" part and "low" part.
    Use `big_mean` to compute the mean of the high part,
    and the standard numpy `mean` to compute the mean of the low part.
    The final mean is a weighted average of the two part means.
    """
    assert isinstance(x, np.ndarray) and x.ndim == 1
    x = x[np.argsort(abs(x))]
    n = len(x)
    threshold = np.finfo(x.dtype).max / (n + 1)
    try:
        first_high_ind = np.where(abs(x) > threshold)[0][0]
    except IndexError:
        first_high_ind = n
    if n == 0 or first_high_ind > 0:
        low_mean = x[:first_high_ind].mean()
        if first_high_ind == n:
            return low_mean
    high_mean = big_mean(x[first_high_ind:])
    if first_high_ind == 0:
        return high_mean
    low_weight = float(first_high_ind) / n
    return low_weight * low_mean + (1 - low_weight) * high_mean

def stable_mean(x, axis=None):
    x = np.asarray(x)
    axis = canonical_axis(x, axis)
    keep_axis = [i for i in xrange(x.ndim) if i not in axis]
    out_shape = np.array([x.shape[i] for i in keep_axis], dtype=np.int)
    x = x.transpose(keep_axis + axis).reshape(out_shape.prod(), -1)
    out = np.array([flat_stable_mean(xi) for xi in x])
    if len(out_shape):
        return out.reshape(out_shape)
    return out[0]

if __name__ == '__main__':
    import itertools

    def approx_equal(a, b):
        return np.all(np.sign(a) == np.sign(b)) and \
               np.all(np.isinf(a) == np.isinf(b)) and \
               np.all(np.isclose(a, b))

    # basic tests
    data = np.random.randn(3, 4, 5)

    sm = stable_mean(data)
    npm = np.mean(data)
    assert approx_equal(sm, npm)

    sm = stable_mean(data, axis=0)
    npm = np.mean(data, axis=0)
    assert approx_equal(sm, npm)

    sm = stable_mean(data, axis=(-1,))
    npm = np.mean(data, axis=(-1,))
    assert approx_equal(sm, npm)

    sm = stable_mean(data, axis=(0, 2))
    npm = np.mean(data, axis=(0, 2))
    assert approx_equal(sm, npm)

    sm = stable_mean(data, axis=(-1, 1))
    npm = np.mean(data, axis=(-1, 1))
    assert approx_equal(sm, npm)

    def get_large_neg_value(t):
        return np.finfo(t).min / 2

    def get_large_value(t):
        return np.finfo(t).max / 2

    def get_normal_value(t):
        x = np.asarray(np.random.randn(1000), dtype=t)
        return x, x[np.argsort(abs(x))].mean()

    def get_small_value(t):
        start = np.array(0, dtype=t)
        target = np.array(1, dtype=t)
        return np.nextafter(start, target) * 2

    n = 10
    f_means = [
        ('NumPy mean', np.mean),
        ('big_mean', big_mean),
        ('stable_mean', stable_mean)
    ]
    f_values = get_large_value, get_large_neg_value, \
        get_normal_value, get_small_value
    types = 'float32', 'float64', 'float128'
    num_tested = 0
    num_passed = {name: 0 for name, _ in f_means}
    name_pad = max(len(n) for n, _ in f_means)
    debug = False
    for f, t in itertools.product(f_values, types):
        x = f(t)
        if isinstance(x, tuple):
            x_in, true_mean = x
        else:
            true_mean = x
            x_in = np.asarray([x] * n, dtype=t)
        print 'True mean:', true_mean
        num_tested += 1
        for name, f_mean in f_means:
            computed_mean = f_mean(x_in)
            passed = approx_equal(computed_mean, true_mean)
            passed_str = 'pass' if passed else 'fail'
            print '\t{:{pad}}: {} ({})'.format(
                name, str(computed_mean), passed_str, pad=name_pad)
            num_passed[name] += passed
            if debug and (not passed) and name == 'stable_mean':
                import pdb; pdb.set_trace()
    print 'Summary:'
    for name, _ in f_means:
        num = num_passed[name]
        pass_percent = 100.0 * num / num_tested
        print '\t{:{pad}}: {}/{} ({}%) passed'.format(
            name, num, num_tested, pass_percent, pad=name_pad)
