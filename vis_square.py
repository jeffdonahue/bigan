import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.externals import joblib
import sys

assert len(sys.argv) == 3

# def vis_square(data, output_file, padsize=1, padval=0, copy=True, normalize_individually=True):
def vis_square(data, output_file, padsize=1, padval=0, copy=True, normalize_individually=False):
    if copy:
        data = data.copy()
    def normalize(data):
        data -= data.min()
        data /= data.max()
    if normalize_individually:
        for datum in data:
            normalize(datum)
    else:
        normalize(data)
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imsave(output_file, data)

_, input_file, output_prefix = sys.argv
data = joblib.load(input_file)
for i, d in enumerate(data):
    if len(d.shape) != 4 or d.shape[1] != 3:
        continue
    head, tail = os.path.splitext(output_prefix)
    filename = '%s.%d%s' % (head, i, tail)
    print 'Saving params %d with shape %s: %s' % (i, d.shape, filename)
    d = d.transpose((0, 2, 3, 1))
    vis_square(d, filename)
