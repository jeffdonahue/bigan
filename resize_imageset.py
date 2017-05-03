#!/usr/bin/env python

from __future__ import division

import functools
import itertools
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    print 'module tqdm not found; progress will not be displayed'
    def tqdm(gen, *a, **k): return gen

def file_list(root, directory='.', recursive=False):
    join = os.path.join
    directory_path = join(root, directory)
    filenames = os.listdir(directory_path)
    out = []
    for filename in filenames:
        path = join(directory_path, filename)
        relative_path = join(directory, filename)
        if not os.path.isdir(path):
            out.append(relative_path)
        elif recursive:
            out += file_list(root, directory=relative_path, recursive=True)
    return out

def makedirs(root, file_list):
    dirs = set(os.path.dirname(os.path.join(root, f)) for f in file_list)
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def resize_image(minor_size, in_root, out_root, path,
                 overwrite=False, format=None, check=False, verbose=False,
                 strict=False, use_plt=False):
    out_full_path = os.path.join(out_root, path)
    if format is None:
        is_png = (os.path.splitext(path)[1] == 'png')
    else:
        out_full_path = '%s.%s' % (os.path.splitext(out_full_path)[0], format)
        is_png = (format.lower() == 'png')
    if (not overwrite) and os.path.exists(out_full_path):
        if check:
            try:
                image = Image.open(out_full_path)
                image.load()
                assert min(image.size) == minor_size
                return
            except Exception as e:
                print 'Got exception on existing image, removing:', out_full_path
                print e.message
                if e.message == 'Decompressed Data Too Large':
                    print 'Trying to save with PLT'
                    use_plt = True
                os.remove(out_full_path)
        else:
            return
    full_path = os.path.join(in_root, path)
    try:
        image = Image.open(full_path)
    except IOError:
        print 'IOError on open:', full_path
        if strict:
            raise
        return
    size = image.size
    orig_minor_size = min(size)
    scale_factor = minor_size / orig_minor_size
    new_size = tuple(int(round(scale_factor * s)) for s in size)
    image.draft(None, new_size)
    try:
        image = image.resize(new_size, Image.ANTIALIAS)
    except IOError:
        print 'IOError on resize:', full_path
        if strict:
            raise
        return
    if is_png:
        image = image.convert('RGB')
    if verbose:
        print 'Resized image (%s) from %s to %s; saving to file: %s' % \
            (full_path, size, image.size, out_full_path)
    if use_plt:
        plt.imsave(out_full_path, np.array(image, dtype=np.uint8))
    else:
        image.save(out_full_path, quality=95)

class FakePool(object):
    def imap_unordered(self, function, iterable, chunksize=None):
        return itertools.imap(function, iterable)

def resize_directory(minor_size, in_root, out_root, pool=FakePool(),
                     verbose=False, **kwargs):
    if verbose: print 'Finding files in input dir: ', in_root
    files = file_list(in_root, recursive=args.recursive)
    if verbose:
        print 'Done. Found %d files:' % len(files)
        k = 10
        print '\n'.join('\t'+f for f in
                        (files[:k] + (['...'] if len(files) > k else [])))
        print 'Replicating subdir structure in output dir:', out_root
    makedirs(out_root, files)
    if verbose: print 'Done.'
    resize = functools.partial(resize_image, minor_size,
                               in_root, out_root, **kwargs)
    if verbose: print 'Resizing %d images.' % len(files)
    iterator = pool.imap_unordered(resize, files, chunksize=10)
    for _ in tqdm(iterator, total=len(files)): pass
    if verbose: print 'Done.'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=(
        'Resizes a directory of images, with each image resized proportionally '
        'such that the shorter edge has a length (# of pixels) specified as '
        '`size`.'

        'Attempts to load all files as images; skips any file that fails to '
        'do so.'

        'Resumes from previous partially completed resizes. '
        'Use -c to check existing files; otherwise, assume any existing '
        'files are valid and opens neither the target, nor the source file.'

        'Use `-r` to recurse into subdirectories '
        '(duplicating the same subdirectory structure in the '
        'output directory).'

        'Use, e.g., `-j 2` to parallelize over 2 threads.'))
    parser.add_argument('-c', '--check', action='store_true',
        help='Check existing images; remove and replace if invalid')
    parser.add_argument('-f', '--format',
        help='Output format, e.g. "png" or "jpg" '
             '(if unspecified, use input format)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet')
    parser.add_argument('-j', '--jobs', type=int, default=0,
        help='Number of processes to do the resizing (0 for serial processing)')
    parser.add_argument('-r', '--recursive', action='store_true',
        help='Recursively explore subdirectories of the input directory')
    parser.add_argument('-o', '--overwrite', action='store_true',
        help='Overwrite any existing images (otherwise, skip)')
    parser.add_argument('size', type=int,
        help='The minor edge size of the output')
    parser.add_argument('input_directory',
        help='Directory containing input images')
    parser.add_argument('output_directory',
        help='Directory to be filled with output images')
    args = parser.parse_args()
    if not args.quiet: print 'Opening pool with %d workers' % args.jobs
    if args.jobs == 0:
        pool = FakePool()
    elif args.jobs >= 1:
        from multiprocessing import Pool
        pool = Pool(args.jobs)
    else:
        raise ValueError('--jobs (-j) must be non-negative')
    resize_directory(args.size, args.input_directory, args.output_directory,
                     pool=pool, overwrite=args.overwrite, format=args.format,
                     check=args.check, verbose=(not args.quiet))
