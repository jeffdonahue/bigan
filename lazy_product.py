try:
    range = xrange # Python 2
except NameError:
    pass # Python 3

def lazy_product(*iter_funcs, **kwargs):
    """
    If f1, f2, ..., are functions which have no (required) arguments and
    return iterables, then
        lazy_product(f1, f2, ..., repeat=k)
    is equivalent to
        itertools.product(f1(), f2(), ..., repeat=k);
    but much faster in certain cases.

    For example, let f have the following definition:

        def f(n):
            def func():
                return xrange(n)
            return func

    Then, this code:
        itertools.product(*[f(N)() for _ in xrange(M)], repeat=K)
    takes O(NMK) time and memory to execute, whereas
        lazy_product(*[f(N) for _ in xrange(M)], repeat=K)
    is equivalent, and takes just O(MK) time and memory.
    (Of course, iterating over either result is exactly N^(MK) steps, and each
    step takes O(1) time; the only difference between itertools.product and
    lazy_product is at the time of initialization of the iterable -- the first
    call to next(x), where x is the product result: x = itertools.product(...)
    or x = lazy_product(...).)

    The speed/memory overhead results from itertools.product saving the full
    result of xrange(m) as a list (or similar data structure) in memory.  This
    is necessary as itertools.product takes iterables as input, and, in general,
    it is not possible to "reset" an iterator, so all of its values instead need
    to be stored.  So, the input to lazy_product is an iterable of *functions*
    returning iterables, rather than the iterables themselves, allowing for
    repeated iteration over each iterable (by calling iter_func again when we
    reach the end of the iterable that iter_func created on the previous call).

    Inputs:

      - iter_funcs: functions that create and return an iterable

      - kwargs: a dict which is either empty or contains only the key `repeat`,
        with an integer value.  In Python 3, the function definition could
        (much more clearly) be written as
            def lazy_product(*iter_funcs, repeat=1)
        and the first two lines of ugly parsing code could be dropped.

    Returns:
        an iterator over the Cartesian product of the iterables returned
        by the elements of iter_funcs -- equivalent to:
            return itertools.product(*(f() for f in iter_funcs), **kwargs)
    """
    repeat = kwargs.pop('repeat', 1)
    if kwargs: raise ValueError('unknown kwargs: %s' % kwargs.keys())
    iters = [iter(f()) for _ in range(repeat) for f in iter_funcs]
    values = [next(i) for i in iters]
    while True:
        yield tuple(values)
        for index in reversed(range(-1, len(iters))):
            if index < 0: return
            try:
                values[index] = next(iters[index])
                break
            except StopIteration:
                iters[index] = iter(iter_funcs[index % len(iter_funcs)]())
                values[index] = next(iters[index])

from functools import partial
def lazy_product_func(*a, **k):
    return partial(lazy_product, *a, **k)
def range_func(*a, **k):
    return partial(range, *a, **k)
xrange_func = range_func

if __name__ == '__main__':
    import itertools
    def test_equivalence(*iter_funcs, **kwargs):
        lazy_result = lazy_product(*iter_funcs, **kwargs)
        iters = (f() for f in iter_funcs)
        itertools_result = itertools.product(*iters, **kwargs)
        return list(lazy_result) == list(itertools_result)
    assert test_equivalence()
    assert test_equivalence(repeat=0)
    assert test_equivalence(repeat=1)
    assert test_equivalence(repeat=2)
    assert test_equivalence(range_func(0))
    assert test_equivalence(range_func(0), repeat=2)
    assert test_equivalence(range_func(2))
    assert test_equivalence(range_func(2), repeat=2)
    assert test_equivalence(range_func(2), range_func(3))
    assert test_equivalence(range_func(2), range_func(0), range_func(3))
    assert test_equivalence(range_func(2), range_func(0), range_func(3),
                            repeat=2)
    assert test_equivalence(range_func(2), range_func(3), repeat=2)
    assert test_equivalence(range_func(2), range_func(3), repeat=2)
    assert test_equivalence(range_func(3), range_func(2, 7), repeat=0)
    assert test_equivalence(range_func(3), range_func(2, 7), repeat=4)
    print('Test passed!')
