import theano
import theano.tensor as T
import numpy as np

from theano_utils import shared0s, floatX, sharedX

from metrics import l2norm

def clip_norm(g, c, n):
    if c > 0:
        g = T.switch(T.ge(n, c), g*c/n, g)
    return g

def clip_norms(gs, c):
    norm = T.sqrt(sum([T.sum(g**2) for g in gs]))
    return [clip_norm(g, c, norm) for g in gs]

class Regularizer(object):

    def __init__(self, l1=0., l2=0., maxnorm=0., l2norm=False, frobnorm=False,
                 ignored_prefixes=[]):
        ignored_prefixes = set(ignored_prefixes)
        self.__dict__.update(locals())

    def max_norm(self, p, maxnorm):
        if maxnorm > 0:
            norms = T.sqrt(T.sum(T.sqr(p), axis=0))
            desired = T.clip(norms, 0, maxnorm)
            p = p * (desired/ (1e-7 + norms))
        return p

    def l2_norm(self, p):
        return p/l2norm(p, axis=0)

    def frob_norm(self, p, nrows):
        return (p/T.sqrt(T.sum(T.sqr(p))))*T.sqrt(nrows)

    def ignored(self, p):
        return (p.name is not None) and \
            (any(p.name.startswith(pre) for pre in self.ignored_prefixes) or
             any((('/' + pre) in p.name) for pre in self.ignored_prefixes))

    def gradient_regularize(self, p, g):
        if self.ignored(p):
            return g
        if self.l2 != 0:
            g += p * self.l2
        if self.l1 != 0:
            g += T.sgn(p) * self.l1
        return g

    def weight_regularize(self, p):
        if self.ignored(p):
            return p
        p = self.max_norm(p, self.maxnorm)
        if self.l2norm:
            p = self.l2_norm(p)
        if self.frobnorm > 0:
            p = self.frob_norm(p, self.frobnorm)
        return p


class Update(object):

    def __init__(self, regularizer=Regularizer(), clipnorm=0.):
        self.__dict__.update(locals())

    def __call__(self, params, grads):
        raise NotImplementedError

class SGD(Update):

    def __init__(self, lr=0.01, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            updated_p = p - self.lr * g
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates

class Momentum(Update):

    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            value = p.get_value() * 0.
            if p.dtype == theano.config.floatX:
                value = floatX(value)
            m = theano.shared(value)
            v = (self.momentum * m) - (self.lr * g)
            updates.append((m, v))

            updated_p = p + v
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates


class NAG(Update):

    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p, g in zip(params, grads):
            g = self.regularizer.gradient_regularize(p, g)
            m = theano.shared(p.get_value() * 0.)
            v = (self.momentum * m) - (self.lr * g)

            updated_p = p + self.momentum * v - self.lr * g
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((m,v))
            updates.append((p, updated_p))
        return updates


class RMSprop(Update):

    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            acc = theano.shared(p.get_value() * 0.)
            acc_new = self.rho * acc + (1 - self.rho) * g ** 2
            updates.append((acc, acc_new))

            updated_p = p - self.lr * (g / T.sqrt(acc_new + self.epsilon))
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates


class Adam(Update):

    def __init__(self, lr=0.001, b1=0.9, b2=0.999, e=1e-8, l=1-1e-8, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())  

    def __call__(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)  
        t = theano.shared(floatX(0.))
        b1_t = self.b1*self.l**t
        tp1 = t + 1.
     
        for p, g in zip(params, grads):
            g = self.regularizer.gradient_regularize(p, g)
            value = p.get_value() * 0.
            if p.dtype == theano.config.floatX:
                value = floatX(value)
            m = theano.shared(value)
            v = theano.shared(value)
     
            m_t = b1_t*m + (1 - b1_t)*g
            v_t = self.b2*v + (1 - self.b2)*g**2
            m_c = m_t / (1-self.b1**tp1)
            v_c = v_t / (1-self.b2**tp1)
            p_t = p - (self.lr * m_c) / (T.sqrt(v_c) + self.e)
            p_t = self.regularizer.weight_regularize(p_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t) )
        updates.append((t, tp1))
        return updates


class Adagrad(Update):

    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            acc = theano.shared(p.get_value() * 0.)
            acc_t = acc + g ** 2
            updates.append((acc, acc_t))

            p_t = p - (self.lr / T.sqrt(acc_t + self.epsilon)) * g
            p_t = self.regularizer.weight_regularize(p_t)
            updates.append((p, p_t))
        return updates  


class Adadelta(Update):

    def __init__(self, lr=0.5, rho=0.95, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)

            acc = theano.shared(p.get_value() * 0.)
            acc_delta = theano.shared(p.get_value() * 0.)
            acc_new = self.rho * acc + (1 - self.rho) * g ** 2
            updates.append((acc,acc_new))

            update = g * T.sqrt(acc_delta + self.epsilon) / T.sqrt(acc_new + self.epsilon)
            updated_p = p - self.lr * update
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))

            acc_delta_new = self.rho * acc_delta + (1 - self.rho) * update ** 2
            updates.append((acc_delta,acc_delta_new))
        return updates


class NoUpdate(Update):

    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = []
        for p in params:
            updates.append((p, p))
        return updates
