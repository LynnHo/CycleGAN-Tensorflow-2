from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc


def _to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    # transform images from [-1.0, 1.0] to [min_value, max_value] of dtype
    assert \
        np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)


def _im2uint(images):
    # transform images from [-1.0, 1.0] to uint8
    return _to_range(images, 0, 255, np.uint8)


def imread(path, mode='RGB'):
    """Read an image.

    read an image into [-1.0, 1.0] of float64

    `mode` can be one of the following strings:

    * 'L' (8 - bit pixels, black and white)
    * 'P' (8 - bit pixels, mapped to any other mode using a color palette)
    * 'RGB' (3x8 - bit pixels, true color)
    * 'RGBA' (4x8 - bit pixels, true color with transparency mask)
    * 'CMYK' (4x8 - bit pixels, color separation)
    * 'YCbCr' (3x8 - bit pixels, color video format)
    * 'I' (32 - bit signed integer pixels)
    * 'F' (32 - bit floating point pixels)
    """
    return scipy.misc.imread(path, mode=mode) / 127.5 - 1


def imwrite(image, path):
    # save an [-1.0, 1.0] image
    return scipy.misc.imsave(path, _to_range(image, 0, 255, np.uint8))


def imresize(image, size, interp='bilinear'):
    """Resize an image.

    Resize an [-1.0, 1.0] image.

    size: int, float or tuple
        * int - Percentage of current size.
        * float - Fraction of current size.
        * tuple - Size of the output image.

    interp: str, optional
        Interpolation to use for re - sizing('nearest', 'lanczos', 'bilinear', 'bicubic'
        or 'cubic').
    """
    # scipy.misc.imresize should deal with uint8 image, or it would cause some problem (scale the image to [0, 255])
    return (scipy.misc.imresize(_im2uint(image), size, interp=interp) / 127.5 - 1).astype(image.dtype)


def immerge(images, row, col):
    """Merge images.

    merge images into an image with (row * h) * (col * w)

    `images` is in shape of N * H * W(* C=1 or 3)
    """
    if images.ndim == 4:
        c = images.shape[3]
    elif images.ndim == 3:
        c = 1

    h, w = images.shape[1], images.shape[2]
    if c > 1:
        img = np.zeros((h * row, w * col, c))
    else:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    return img
