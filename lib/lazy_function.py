import theano

class LazyFunction(object):
    def __init__(self, *args, **kwargs):
        self.function = None
        self.args = args
        self.kwargs = kwargs
    def __call__(self, *args, **kwargs):
        if self.function is None:
            print(self.args)
            print(self.kwargs)
            self.function = theano.function(*self.args, **self.kwargs)
            del self.args
            del self.kwargs
        return self.function(*args, **kwargs)
