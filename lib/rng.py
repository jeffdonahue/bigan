import numpy
from numpy.random import RandomState
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import random
from random import Random

seed = 42

random.seed(seed+1)
numpy.random.seed(seed+1)

py_rng = Random(seed)
np_rng = RandomState(seed)
t_rng = RandomStreams(seed)

def set_seed(n):
    global seed, py_rng, np_rng, t_rng
    
    seed = n
    py_rng = Random(seed)
    np_rng = RandomState(seed)
    t_rng = RandomStreams(seed)
