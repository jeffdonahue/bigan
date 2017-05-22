#!/usr/bin/env python

import numpy as np

def convert(image):
    image = np.asarray(image)
    if len(image.shape) == 2:
        image = image[..., np.newaxis]
    assert len(image.shape) == 3
    return image.transpose(2, 0, 1).copy()

def save_gif(filename, frames, delay=20, ext='.gif'):
    import os
    from scipy.misc import imsave
    import shutil
    from subprocess import check_call
    from tempfile import mkdtemp
    frames = np.asarray(frames)
    assert 3 <= len(frames.shape) <= 4
    # Create a temp directory containing a png of each frame.
    frame_dir = mkdtemp()
    frame_filenames = []
    for i, frame in enumerate(frames):
        frame_filename = os.path.join(frame_dir, 'frame_{}.png'.format(i))
        frame_filenames.append(frame_filename)
        imsave(frame_filename, frame)
    # Run `convert` to convert the png frames to a gif.
    needs_move = not filename.endswith(ext)
    output_filename = filename+ext if needs_move else filename
    cmd = (['convert', '-delay', str(delay), '-loop', '0'] +
           frame_filenames + [output_filename])
    check_call(cmd)
    # Move gif to `filename` if needed.
    if needs_move:
        shutil.move(output_filename, filename)
    # Remove temp directory.
    shutil.rmtree(frame_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=
        'Convert multichannel (e.g. RGB) image to grayscale gif with a frame per channel')
    parser.add_argument('in_filename', help='Path to input image')
    parser.add_argument('out_filename', help='Path to output image')
    parser.add_argument('--delay', '-d', type=float, default=20,
                        help='Time (ms) between gif frames')

    args = parser.parse_args()

    from scipy.misc import imread
    image = imread(args.in_filename)
    out_frames = convert(image)
    save_gif(args.out_filename, out_frames, delay=args.delay)

    def shpstr(x): return 'x'.join(str(n) for n in x.shape)
    print 'Saved to:', args.out_filename
