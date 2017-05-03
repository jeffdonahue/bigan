from __future__ import division

import caffe
import numpy as np
from sklearn.externals import joblib

BN_EPS = 1e-8

def get_caffenet(model_filename):
    return caffe.Net(model_filename, caffe.TEST)

def load_weights(weights_filename):
    return joblib.load(weights_filename)

def check_caffe_weights(weights):
    # check that each caffe layer has 1 or 2 weights
    # (the filters/weights and maybe a bias)
    shapes = [tuple(w.shape) for w in weights]
    assert len(shapes) in (1, 2)
    assert 2 <= len(shapes[0]) <= 4
    dim = shapes[0][0]
    has_bias = (len(shapes) == 2)
    group = {}
    group['weight'] = weights[:1]
    if len(shapes) == 2:
        bias = shapes[1]
        assert len(bias) == 1 and bias[0] == dim
        group['shift'] = weights[1:]
    return group

def check_theano_weights(weights):
    # theano weights may have any of these structures:
    # len(shapes) ==
    #   1: (filters/weights)
    #   2: (filters/weights, biases)
    #   3: (filters/weights, gains, biases)
    #   4: (filters/weights, BN count, BN mean, BN var)
    #   5: (filters/weights, BN count, BN mean, BN var, biases)
    #   6: (filters/weights, BN count, BN mean, BN var, gains, biases)
    shapes = [w.shape for w in weights]
    assert 1 <= len(shapes) <= 6
    assert 2 <= len(shapes[0]) <= 4
    if len(shapes[0]) == 4:
        dim = shapes[0][0]
    elif len(shapes[0]) == 2:
        dim = shapes[0][1]
        weights[0] = weights[0].T
    else:
        raise ValueError('Unknown ndims: %d' % len(shapes[0]))
    group = {}
    group['weight'] = weights[:1]
    offset = 1
    if len(shapes) >= 4:
        # has BN
        count, mean, var = shapes[1:4]
        assert len(count) == 0
        assert len(mean)  == 1 and mean[0] == dim
        assert len(var)   == 1 and var[0]  == dim
        group['bn'] = weights[1:4]
        offset = 4
    if len(shapes) - offset >= 1:
        bias = shapes[-1]
        assert len(bias) == 1 and bias[0] == dim
        group['shift'] = weights[-1:]
    if len(shapes) - offset >= 2:
        gain = shapes[-2]
        assert len(gain) == 1 and gain[0] == dim
        group['scale'] = weights[-2:-1]
    return group

def transplant_weights(weights, caffenet, flip_filters=True, reverse_3ch=True):
    weight_inds = [i for i, w in enumerate(weights) if len(w.shape) >= 2]
    weights = [weights[start:end]
               for start, end in zip([0] + weight_inds, weight_inds + [None])
               if (end is None or end > start)]
    weights_index = 0
    mismatched = None
    num_layers = 0
    for (name, caffe_weights), theano_weights in \
            zip(caffenet.params.items(), weights):
        caffe_weights = check_caffe_weights(caffe_weights)
        group = theano_weights = check_theano_weights(theano_weights)
        if len(theano_weights) > 1 and len(caffe_weights) == 1:
            print ('Layer "%s" did not match: '
                   'Theano had bias; Caffe layer had only weights') % name
            mismatched = name
            break
        weights = caffe_weights['weight'][0]
        source_weights = group['weight'][0]
        if tuple(weights.shape) != source_weights.shape:
            print ('Layer "%s" did not match: '
                   'weight.shape = %s != %s = source_weight.shape') \
                   % (name, tuple(weights.shape), source_weights.shape)
            mismatched = name
            break
        source_params = caffenet.params[name]
        scale = 1
        if 'shift' in caffe_weights:
            assert len(caffe_weights['shift']) == 1
            shift = caffe_weights['shift'][0].data.copy()
        else:
            shift = 0
        if 'bn' in group:
            bn_params = group['bn']
            assert len(bn_params) == 3
            inv_scale_factor, mean, var = [p.copy() for p in bn_params]
            mean, var = [p / inv_scale_factor for p in (mean, var)]
            stdev = (var + BN_EPS) ** 0.5
            scale /= stdev
            shift -= mean
            shift /= stdev
            print "Merging BN into conv:", name
        if 'scale' in group:
            assert len(group['scale']) == 1
            scale_param = group['scale'][0].copy()
            scale *= scale_param
            shift *= scale_param
            print "Merging scale into conv:", name
        if 'shift' in group:
            assert len(group['shift']) == 1
            shift += group['shift'][0].copy()
            print "Merging shift into conv:", name
        if isinstance(scale, np.ndarray):
            weights.data[...] = (source_weights.T * scale).T
        else:
            print "Directly transplanting weights: %s" % name
            assert scale == 1
            weights.data[...] = source_weights[...]
        if flip_filters and len(weights.shape) == 4:
            weights.data[...] = weights.data[:, :, ::-1, ::-1]
        if reverse_3ch and weights.shape[1] == 3:
            print 'Reversing 3 channel inputs for weights:', name
            weights.data[...] = weights.data[:, ::-1]
        if isinstance(shift, np.ndarray):
            assert 'shift' in caffe_weights, 'need bias'
            bias = caffe_weights['shift'][0]
            assert shift.shape == tuple(bias.shape)
            bias.data[...] = shift[...]
            if reverse_3ch and bias.data.shape[0] == 3:
                print 'Reversing 3 channel output biases:', name
                bias.data[...] = bias.data[::-1]
        elif 'shift' in caffe_weights:
            print "Zero initializing biases: %s" % name
            caffe_weights['shift'][0].data[...] = 0
        num_layers += 1
    print 'Transplanted weights of %d layers' % num_layers
    if mismatched is not None:
        print 'Warning: mismatch starting at layer:', mismatched

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert train_gan.py output to caffemodel')
    parser.add_argument('model', help='(*.prototxt) Caffe model specification')
    parser.add_argument('weights',
        help='(*.jl) weights file saved by train_gan.py')
    parser.add_argument('output', help='(*.caffemodel) output Caffe model file')
    args = parser.parse_args()
    weights = load_weights(args.weights)
    caffenet = get_caffenet(args.model)
    transplant_weights(weights, caffenet)
    print 'Saving transplanted caffenet to:', args.output
    caffenet.save(args.output)
