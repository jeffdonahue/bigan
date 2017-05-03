import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import gc
import time

from theano_utils import floatX

from sklearn import metrics
from sklearn.linear_model import LogisticRegression as LR

def l2norm(x, axis=1, e=1e-8, keepdims=True):
    return T.sqrt(T.sum(T.sqr(x), axis=axis, keepdims=keepdims) + e)

def cosine(x, y):
    d = T.dot(x, y.T)
    d /= l2norm(x).dimshuffle(0, 'x')
    d /= l2norm(y).dimshuffle('x', 0)
    return d

def euclidean(x, y, e=1e-8):
    xx = T.sqr(T.sqrt((x*x).sum(axis=1) + e))
    yy = T.sqr(T.sqrt((y*y).sum(axis=1) + e))
    dist = T.dot(x, y.T)
    dist *= -2
    dist += xx.dimshuffle(0, 'x')
    dist += yy.dimshuffle('x', 0)
    dist = T.sqrt(dist)
    return dist

def cv_reg_lr(trX, trY, vaX, vaY, Cs=[0.01, 0.05, 0.1, 0.5, 1., 5., 10., 50., 100.], return_score=True):
    tr_accs = []
    va_accs = []
    models = []
    for C in Cs:
        model = LR(C=C)
        model.fit(trX, trY)
        tr_pred = model.predict(trX)
        va_pred = model.predict(vaX)
        tr_acc = metrics.accuracy_score(trY, tr_pred)
        va_acc = metrics.accuracy_score(vaY, va_pred)
        if not return_score:
            print '%.4f %.4f %.4f'%(C, tr_acc, va_acc)
        tr_accs.append(tr_acc)
        va_accs.append(va_acc)
        models.append(model)
    best = np.argmax(va_accs)
    if return_score: return 100*va_accs[best]
    print 'best model C: %.4f tr_acc: %.4f va_acc: %.4f'%(Cs[best], tr_accs[best], va_accs[best])
    return models[best]

def gpu_nnc_predict(trX, trY, teX, metric='cosine', batch_size=4096):
    if metric == 'cosine':
        metric_fn = cosine_dist
    else:
        metric_fn = euclid_dist
    idxs = []
    for i in range(0, len(teX), batch_size):
        mb_dists = []
        mb_idxs = []
        for j in range(0, len(trX), batch_size):
            dist = metric_fn(floatX(teX[i:i+batch_size]), floatX(trX[j:j+batch_size]))
            if metric == 'cosine':
                mb_dists.append(np.max(dist, axis=1))
                mb_idxs.append(j+np.argmax(dist, axis=1))
            else:
                mb_dists.append(np.min(dist, axis=1))
                mb_idxs.append(j+np.argmin(dist, axis=1))                
        mb_idxs = np.asarray(mb_idxs)
        mb_dists = np.asarray(mb_dists)
        if metric == 'cosine':
            i = mb_idxs[np.argmax(mb_dists, axis=0), np.arange(mb_idxs.shape[1])]
        else:
            i = mb_idxs[np.argmin(mb_dists, axis=0), np.arange(mb_idxs.shape[1])]
        idxs.append(i)
    idxs = np.concatenate(idxs, axis=0)
    nearest = trY[idxs]
    return nearest

def gpu_nnd_score(trX, teX, metric='cosine', batch_size=4096):
    if metric == 'cosine':
        metric_fn = cosine_dist
    else:
        metric_fn = euclid_dist
    dists = []
    for i in range(0, len(teX), batch_size):
        mb_dists = []
        for j in range(0, len(trX), batch_size):
            dist = metric_fn(floatX(teX[i:i+batch_size]), floatX(trX[j:j+batch_size]))
            if metric == 'cosine':
                mb_dists.append(np.max(dist, axis=1))
            else:
                mb_dists.append(np.min(dist, axis=1))         
        mb_dists = np.asarray(mb_dists)
        if metric == 'cosine':
            d = np.max(mb_dists, axis=0)
        else:
            d = np.min(mb_dists, axis=0)
        dists.append(d)
    dists = np.concatenate(dists, axis=0)
    return float(np.mean(dists))

A = T.matrix()
B = T.matrix()

ed = euclidean(A, B)
cd = cosine(A, B)

cosine_dist = theano.function([A, B], cd)
euclid_dist = theano.function([A, B], ed)

def nnc_score(trX, trY, teX, teY, metric='euclidean'):
    pred = gpu_nnc_predict(trX, trY, teX, metric=metric)
    acc = metrics.accuracy_score(teY, pred)
    return acc*100. 

def nnd_score(trX, teX, metric='euclidean'):
    return gpu_nnd_score(trX, teX, metric=metric)
