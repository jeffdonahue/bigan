#!/usr/bin/env python

import numpy as np

def convert(image, tile=None, horizontal=True):
    image = np.asarray(image)
    if len(image.shape) == 2:
        image = image[..., np.newaxis]
    assert len(image.shape) == 3

    if tile is None:
        tile = tuple(image.shape[:2])
    elif isinstance(tile, int):
        tile = (tile, tile)
    assert len(tile) == 2 and all(isinstance(t, int) and t>0 for t in tile)

    h, w, c = image.shape
    out_h, out_w = h, w

    tile_h, tile_w = tile
    out_tile_h, out_tile_w = tile

    if horizontal:
        out_tile_w *= c
        out_w *= c
    else:
        out_tile_h *= c
        out_h *= c

    out_shape = (h, w*c) if horizontal else (h*c, w)
    out_image = np.zeros(out_shape, dtype=image.dtype)
    for h_start, out_h_start in \
            zip(range(0, h, tile_h), range(0, out_h, out_tile_h)):
        h_end = h_start + tile_h
        out_h_end = out_h_start + out_tile_h
        for w_start, out_w_start in \
                zip(range(0, w, tile_w), range(0, out_w, out_tile_w)):
            w_end = w_start + tile_w
            out_w_end = out_w_start + out_tile_w
            in_tile = image[h_start:h_end, w_start:w_end]
            out_tile = out_image[out_h_start:out_h_end, out_w_start:out_w_end]
            if horizontal:
                out_tile[...] = in_tile.transpose(0, 2, 1).reshape(tile_h, -1)
            else:
                out_tile[...] = in_tile.transpose(2, 0, 1).reshape(-1, tile_w)
    return out_image

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=
        'Convert multichannel (e.g. RGB) image to grayscale per-channel images')
    parser.add_argument('in_filename', help='Path to input image')
    parser.add_argument('out_filename', help='Path to output image')
    parser.add_argument('--horizontal', '-H', action='store_true',
        help='Tile channels horizontally')
    parser.add_argument('--vertical', '-V', action='store_false',
        dest='horizontal', help='Tile channels vertically (default)')
    parser.set_defaults(horizontal=False)
    parser.add_argument('--tile', '-t',
        help='If specified, assume the input image contains tiles of this size, '
             'and split the channels into "subtiles". '
             'Specified as either an int for square tiles (e.g., "64") '
             'or pair of ints specified as "HxW" ("64x48").')

    args = parser.parse_args()

    if args.tile is not None:
        args.tile = [int(s) for s in args.tile.split('x')]
        if len(args.tile) == 1:
            args.tile = args.tile[0]

    from scipy.misc import imread, imsave
    image = imread(args.in_filename)
    out_image = convert(image, tile=args.tile, horizontal=args.horizontal)
    imsave(args.out_filename, out_image)

    def shpstr(x): return 'x'.join(str(n) for n in x.shape)
    print 'Flattened {} input to {}. Saved to:'.format(
        shpstr(image), shpstr(out_image))
    print args.out_filename
