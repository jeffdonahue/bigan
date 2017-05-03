from __future__ import division

import sys
sys.path.append('..')

from lib.rng import py_rng, np_rng
from lib.data_utils import shuffle, list_shuffle
from lib.vis import grayscale_grid_vis, color_grid_vis, rgba_grid_vis

import functools
from multiprocessing import Pool
import numpy as np
import scipy.misc
import os
from PIL import Image

class DataProvider(object):
    def __init__(self):
        raise NotImplementedError

    def get_batch(self):
        raise NotImplementedError

    def get_data(self, num):
        num_batches = int(np.ceil(num / self.batch_size))
        total_num = num_batches * self.batch_size
        out_data = out_label = None
        index = 0
        for _ in xrange(num_batches):
            out_batch = self.get_batch()
            if out_data is None:
                bd, bl = out_batch
                out_data = [np.zeros((total_num,) + b.shape[1:], dtype=b.dtype)
                            for b in bd]
                out_label = np.zeros((total_num,) + bl.shape[1:], dtype=bl.dtype)
            end_index = index + self.batch_size
            s = slice(index, end_index)
            for od, ob in zip(out_data, out_batch[0]):
                od[s] = ob
            out_label[s] = out_batch[1]
            index = end_index
        if num != total_num:
            out_data = [d[:num] for d in out_data]
            out_label = out_label[:num]
        return out_data, out_label

def random_crop(size, crop):
    """
    `crop` is a length K iterable of ints specifying crop sizes.

    `image` is ND array with the first K axes being spatial,
    each of which must have dimension >= the corresponding element of crop.

    Example:
        - crop = (160, 120)
        - image = numpy ND array with shape (180, 140, 3)
        -> returns a random crop of shape (3, 160, 120) from image
           (or (160, 120, 3) if not hwc2chw), with range [0, 1]
    """
    return [np_rng.randint(s - c + 1) for s, c in zip(size, crop)]

def center_crop(size, crop):
    return [(s - c + 1) // 2 for s, c in zip(size, crop)]

def pil_random_crop(image, crop):
    offset = [np_rng.randint(s - c + 1) for s, c in zip(image.size, crop)]
    # PIL crop args are left, top, right, bottom
    return image.crop((offset[1]          , offset[0]          ,
                       offset[1] + crop[1], offset[0] + crop[0]))

def get_image(path, root, minor_size=None, crop_shapes=[],
              crop_random=True, image_format='RGB', hwc2chw=True):
    full_path = os.path.join(root, path)
    image = Image.open(full_path)
    if minor_size is not None:
        orig_minor_size = min(image.size)
        scale_factor = minor_size / orig_minor_size
        new_size = tuple(int(round(scale_factor * s))
                         for s in image.size)
        image.draft(None, new_size)
        image = image.resize(new_size, Image.ANTIALIAS)
    image = image.convert(image_format)
    if crop_shapes:
        crop = crop_shapes[0]
        if crop_random:
            start = random_crop(image.size, crop)
        else:
            start = center_crop(image.size, crop)
        box = start[0], start[1], start[0] + crop[0], start[1] + crop[1]
        image = image.crop(box)
    downsampled_images = [np.array(image.resize(s, Image.ANTIALIAS))
                          for s in crop_shapes[1:]]
    images = [np.array(image)] + downsampled_images
    if hwc2chw:
        images = [i.transpose(2, 0, 1) for i in images]
    return images

class ImageDataProvider(DataProvider):
    crop_modes = frozenset(('random', 'center'))

    """
    root: the root directory for all image paths
    data: a list of (image path, label) tuples, where image paths are relative
          to root
    batch_size: number of labeled images in each output batch
    minor_size: the edge size to resize the smaller edge of images to,
                or None to keep the original size
    crop_size: the size of both edges of the square crop
    """
    def __init__(self, root, data, batch_size, minor_size, crop_size,
                 crop_mode='random', num_workers=1, labels=None,
                 max_images=None):
        assert crop_mode in self.crop_modes
        if labels is not None:
            label_set = set(labels)
            assert len(labels) > 0
            labels = list(label_set)
            labels.sort()
            print 'Keeping %d labels: %s' % (len(labels), labels)
            # remap the N labels to the 0:(N-1) range
            label_to_index = {l: i for i, l in enumerate(labels)}
            data = [(image, label_to_index[l]) for image, l in data
                     if l in label_set]
        if (max_images is not None) and (len(data) > max_images):
            print 'Shrinking dataset from %d images to %d images' % \
                (len(data), max_images)
            data = list_shuffle(data)
            data = data[:max_images]
        self.__dict__.update({k: v for k, v in locals().iteritems()
                              if k != 'self'})
        self.batch_indices = np.arange(batch_size)
        self.reset_data()
        self.pool = Pool(num_workers)
        self.map_result = None
        # sort crop sizes largest to smallest
        crop_size = list(reversed(sorted(crop_size)))
        self.image_shapes = [(3, c, c) for c in crop_size]
        self.crop_shapes = [(c, c) for c in crop_size]
        self.out_data = [np.zeros((batch_size, ) + s, dtype=np.uint8)
                         for s in self.image_shapes]
        self.out_label = np.zeros(batch_size, dtype=np.int32)
        kwargs = dict(root=self.root, minor_size=self.minor_size,
                      crop_shapes=self.crop_shapes,
                      crop_random=(crop_mode == 'random'))
        self.get_image = functools.partial(get_image, **kwargs)
        self.start_prefetch()

    def reset_data(self):
        self.data = list_shuffle(self.data)
        self.index = 0

    def get_next_batch_input(self):
        image_paths = []
        labels = []
        num_needed = self.batch_size
        while num_needed > 0:
            end_index = self.index + num_needed
            data = self.data[self.index : end_index]
            image_paths += [d[0] for d in data]
            labels      += [d[1] for d in data]
            self.index = end_index
            if self.index >= len(self.data):
                self.reset_data()
            num_needed = self.batch_size - len(labels)
        return image_paths, labels

    def start_prefetch(self):
        image_paths, self.labels = self.get_next_batch_input()
        self.map_result = self.pool.map_async(self.get_image, image_paths)

    def get_prefetch(self):
        self.map_result.wait()
        return self.map_result.get()

    def get_batch(self):
        prefetch_data = self.get_prefetch()
        # self.out_data is a length K list of NxCxHxW ndarrays
        # prefetch_data is a length N list of length K lists of CxHxW ndarrays
        for index, r in enumerate(self.out_data):
            for batch_index in xrange(r.shape[0]):
                r[batch_index, ...] = prefetch_data[batch_index][index]
        self.out_label[...] = self.labels
        self.start_prefetch()
        return self.out_data, self.out_label

def labeled_image_set(filename, shuffle=True):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        f, l = line.split()
        data.append((f, int(l)))
    if shuffle:
        data = list_shuffle(data)
    return data

def imagenet_data_providers(batch_size, minor_size, crop_size,
        root='data/imagenet', num_test=10000, labels=None, max_images=None):
    if isinstance(crop_size, int): crop_size = [crop_size]
    sets = ('train', 'val')
    if num_test is None:
        sets += ('test',)
        test_split = 'test'
    else:
        test_split = 'val'
    join = os.path.join
    data = {s: labeled_image_set(join(root, '%s.txt' % s)) for s in sets}
    if num_test is not None:
        # make test set from val
        assert 0 <= num_test <= len(data['val'])
        data['test'] = data['val'][:num_test]
        data['val']  = data['val'][num_test:]
    dir_and_split = [('train', 'train'), ('val', 'val'), (test_split, 'test')]
    provider_kwargs = dict(batch_size=batch_size,
                           minor_size=minor_size, crop_size=crop_size,
                           labels=labels, max_images=max_images)
    def provider(subdir, split):
        crop_mode = 'random' if (split == 'train') else 'center'
        return ImageDataProvider(root=join(root, subdir), data=data[split],
                                 crop_mode=crop_mode, **provider_kwargs)
    return {s: provider(d, s) for d, s in dir_and_split}

voc_data_providers = functools.partial(imagenet_data_providers, root='data/voc')
robot_data_providers = functools.partial(imagenet_data_providers, root='data/robot')
nexar_data_providers = functools.partial(imagenet_data_providers, root='data/nexar')
tinyvidbeach_data_providers = functools.partial(imagenet_data_providers, root='data/tinyvidbeach')
tinyvidgolf_data_providers = functools.partial(imagenet_data_providers, root='data/tinyvidgolf')
cityscapes_data_providers = functools.partial(imagenet_data_providers, root='data/cityscapes', num_test=None)

class MemoryDataProvider(DataProvider):
    def __init__(self, data, label, batch_size,
                 image_shape=None, crop_size=None):
        for x in [data, label]:
            assert isinstance(x, np.ndarray)
        assert len(crop_size) == 1
        crop_size = crop_size[0]
        (self.data, self.labels, self.batch_size, self.crop_size,
         self.image_shape) = data, label, batch_size, crop_size, image_shape
        self.reset_data()

    def reset_data(self):
        self.data, self.labels = shuffle(self.data, self.labels)
        self.index = 0

    def get_next_batch_input(self):
        images = labels = None
        num_needed = self.batch_size
        while num_needed > 0:
            end_index = self.index + num_needed
            next_images = self.data[self.index : end_index]
            next_labels = self.labels[self.index : end_index]
            if images is None:
                images = next_images
                labels = next_labels
            else:
                images = np.concatenate([images, next_images], axis=0)
                labels = np.concatenate([labels, next_labels], axis=0)
            self.index = end_index
            if self.index >= len(self.data):
                self.reset_data()
            num_needed = self.batch_size - len(labels)
        return images, labels

    def get_batch(self):
        images, labels = self.get_next_batch_input()
        if self.image_shape is not None:
            images = images.reshape((-1, ) + self.image_shape)
        return [images], labels

def mnist_data_providers(batch_size, crop_size=[], use_test_set=False):
    if isinstance(crop_size, int): crop_size = [crop_size]
    from load import mnist_with_valid_set
    trX, vaX, teX, trY, vaY, teY = mnist_with_valid_set()
    if use_test_set:
        trX = np.concatenate([trX, vaX], axis=0)
        trY = np.concatenate([trY, vaY], axis=0)
        vaX = teX
        vaY = teY
    shape = 1, 28, 28
    return {
        'train': MemoryDataProvider(trX, trY, batch_size,
                                    crop_size=crop_size, image_shape=shape),
        'val'  : MemoryDataProvider(vaX, vaY, batch_size,
                                    crop_size=crop_size, image_shape=shape),
        'test' : MemoryDataProvider(teX, teY, batch_size,
                                    crop_size=crop_size, image_shape=shape),
    }

def pong_data_providers(batch_size, crop_size=[],
        filename='./data/atari/Pong-100000-dqn-dec.h5'):
    import h5py
    from atari_pick_splits import pick_train_val
    with h5py.File(filename) as f:
        data = np.array(f['S'])
    _, train_idx, _, val_idx = pick_train_val(filename)
    labels = np.zeros(len(data), dtype='int32')
    shape = 4, 84, 84
    return {s: MemoryDataProvider(data[i], labels[i], batch_size,
                                  crop_size=crop_size, image_shape=shape)
            for s, i in [('train', train_idx),
                         ('val', val_idx),
                         ('test', val_idx)]}

spaceinv_data_providers = functools.partial(pong_data_providers,
    filename='./data/atari/SpaceInvaders-100000-dqn-dec.h5')
seaquest_data_providers = functools.partial(pong_data_providers,
    filename='./data/atari/Seaquest-100000-dqn-dec.h5')
qbert_data_providers = functools.partial(pong_data_providers,
    filename='./data/atari/Qbert-100000-dqn-dec.h5')
breakout_data_providers = functools.partial(pong_data_providers,
    filename='./data/atari/Breakout-100000-dqn-dec.h5')
beamrider_data_providers = functools.partial(pong_data_providers,
    filename='./data/atari/BeamRider-100000-dqn-dec.h5')

def rescale(X, orig, new, in_place=False):
    assert len(orig) == len(new) == 2
    (a, b), (x, y) = ([float(b) for b in r] for r in (orig, new))
    assert b > a and y > x
    if (a, b) == (x, y):
        return X
    if not in_place:
        X = X.copy()
                  # X \in [a, b]
    # X -= a      # X \in [0, b-a]
    if a != 0:
        X -= a
    # X /= b - a  # X \in [0, 1]
    # X *= y - x  # X \in [0, y-x]
    scale = (y - x) / (b - a)
    if scale != 1:
        X *= scale
    # X += x      # X \in [x, y]
    if x != 0:
        X += x
    return X

class Dataset(object):
    def __init__(self, args):
        crop_resize = (args.crop_size if (args.crop_resize is None)
                       else args.crop_resize)
        crop_sizes = list(set((args.crop_size, crop_resize)))
        if args.dataset == 'mnist':
            assert args.raw_size is None
            assert args.crop_size in (None, 28)
            args.crop_size = 28
            self.nc, self.ny = 1, 10
            self.num_vis_samples = 20
            self.native_range = 0, 1
            def inverse_transform(X, crop=args.crop_resize):
                # X (NCHW) \in [-1, 1] -> [0, 1]
                # returns NHW float array in [0, 1]
                return X.reshape(-1, crop, crop)
            self.grid_vis = grayscale_grid_vis
            providers = mnist_data_providers(args.batch_size, crop_size=crop_sizes,
                                             use_test_set=args.use_test_set)
        elif args.dataset in ('pong', 'spaceinv', 'seaquest',
                              'qbert', 'breakout', 'beamrider'):
            if args.dataset == 'pong':
                from data import pong_data_providers as f_data_providers
            elif args.dataset == 'spaceinv':
                from data import spaceinv_data_providers as f_data_providers
            elif args.dataset == 'seaquest':
                from data import seaquest_data_providers as f_data_providers
            elif args.dataset == 'qbert':
                from data import qbert_data_providers as f_data_providers
            elif args.dataset == 'breakout':
                from data import breakout_data_providers as f_data_providers
            elif args.dataset == 'beamrider':
                from data import beamrider_data_providers as f_data_providers
            else:
                raise ValueError
            assert args.raw_size is None
            assert args.crop_size in (None, 84)
            args.crop_size = args.crop_resize = 84
            self.nc, self.ny = 4, 1
            self.num_vis_samples = 20
            self.native_range = 0, 1
            def inverse_transform(X, crop=args.crop_resize):
                return X.reshape(-1, nc, crop, crop).transpose(0, 2, 3, 1)
            self.grid_vis = rgba_grid_vis
            providers = f_data_providers(args.batch_size, crop_size=crop_sizes)
        else:
            if args.dataset == 'imagenet':
                from data import imagenet_data_providers as data_providers
                root = './data/imagenet'
                num_test = 10000
                real_num_labels = 1000
            elif args.dataset == 'voc':
                from data import voc_data_providers as data_providers
                root = './data/voc'
                num_test = 1
                real_num_labels = 1
            elif args.dataset == 'cityscapes':
                from data import cityscapes_data_providers as data_providers
                root = './data/cityscapes'
                num_test = None
                real_num_labels = 1
            elif args.dataset == 'robot':
                from data import robot_data_providers as data_providers
                root = './data/robot'
                num_test = None
                real_num_labels = 1
            elif args.dataset == 'nexar':
                from data import nexar_data_providers as data_providers
                root = './data/nexar'
                num_test = None
                real_num_labels = 1
            elif args.dataset == 'tinyvidbeach':
                from data import tinyvidbeach_data_providers as data_providers
                root = './data/tinyvidbeach'
                num_test = None
                real_num_labels = 1
            elif args.dataset == 'tinyvidgolf':
                from data import tinyvidgolf_data_providers as data_providers
                root = './data/tinyvidgolf'
                num_test = None
                real_num_labels = 1
            else:
                raise ValueError('Unknown dataset: %s' % (args.dataset,))
            assert args.raw_size is not None
            assert args.crop_size is not None
            self.num_vis_samples = 2
            presized_root = root + str(args.raw_size)
            if args.raw_size and os.path.exists(presized_root):
                root = presized_root
                print 'Using pre-sized data: %s' % root
            else:
                print 'Pre-sized data not found (%s); using original data: %s' % \
                    (presized_root, root)
            if args.include_labels:
                labels = list(set([int(l)
                                   for l in args.include_labels.split(',') if l]))
                labels.sort()
            if args.max_labels and args.include_labels:
                max_labels = args.max_labels
                args.max_labels = None
                if len(labels) > max_labels:
                    print ('Warning: both --max_labels and --include_labels '
                           'specified and len(include_labels)=%d > max_labels=%d; '
                           'truncating labels') % (len(labels), max_labels)
                    labels = labels[:max_labels]
            self.nc = 3
            if args.max_labels:
                labels = range(args.max_labels)
                self.ny = args.max_labels
            elif args.include_labels:
                self.ny = len(labels)
            else:
                labels = None  # use all labels
                self.ny = real_num_labels
            providers = data_providers(args.batch_size, args.raw_size,
                crop_sizes, root=root, num_test=num_test, labels=labels,
                max_images=args.max_images)
            self.native_range = -1, 1
            def inverse_transform(X, crop=args.crop_resize):
                # X (NCHW) \in [-1, 1] -> [0, 1]
                # returns NHWC float array in [0, 1]
                X = X.reshape(-1, self.nc, crop, crop).transpose(0, 2, 3, 1)
                return rescale(X, self.native_range, (0, 1))
            self.grid_vis = color_grid_vis
        all_providers = [providers[k] if k in providers else None
                         for k in ('train', 'val', 'test')]
        self.ntrain, self.nval, self.ntest = [0 if p is None else len(p.data)
                                              for p in all_providers]
        self.train_provider, self.val_provider, self.test_provider = all_providers
        self.inverse_transform = inverse_transform
