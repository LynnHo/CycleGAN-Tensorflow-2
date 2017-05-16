"""
Some codes are modified from https://github.com/Newmu/dcgan_code

These functions are all based on [-1.0, 1.0] image
"""

from __future__ import absolute_import, division, print_function
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    """
    transform images from [-1.0, 1.0] to [min_value, max_value] of dtype
    """
    assert \
        np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)


def im2uint(images):
    """ transform images from [-1.0, 1.0] to uint8 """
    return to_range(images, 0, 255, np.uint8)


def im2float(images):
    """ transform images from [-1.0, 1.0] to [0.0, 1.0] """
    return to_range(images, 0.0, 1.0)


def float2uint(images):
    """ transform images from [0, 1.0] to uint8 """
    assert \
        np.min(images) >= 0.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [0.0, 1.0]!'
    return (images * 255).astype(np.uint8)


def uint2float(images):
    """ transform images from uint8 to [0.0, 1.0] of float64 """
    assert images.dtype == np.uint8, 'The input image type should be uint8!'
    return images / 255.0


def imread(path, mode='RGB'):
    """
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


def read_images(path_list, mode='RGB'):
    """ read a list of images into [-1.0, 1.0] and return the numpy array batch in shape of N * H * W (* C) """
    images = [imread(path, mode) for path in path_list]
    return np.array(images)


def imwrite(image, path):
    """ save an [-1.0, 1.0] image """
    return scipy.misc.imsave(path, to_range(image, 0, 255, np.uint8))


def imshow(image):
    """ show a [-1.0, 1.0] image """
    plt.imshow(to_range(image), cmap=plt.gray())


def rgb2gray(images):
    if images.ndim == 4 or images.ndim == 3:
        assert images.shape[-1] == 3, 'Channel size should be 3!'
    else:
        raise Exception('Wrong dimensions!')

    return (images[..., 0] * 0.299 + images[..., 1] * 0.587 + images[..., 2] * 0.114).astype(images.dtype)


def imresize(image, size, interp='bilinear'):
    """
    Resize an [-1.0, 1.0] image.

    size : int, float or tuple
        * int   - Percentage of current size.
        * float - Fraction of current size.
        * tuple - Size of the output image.

    interp : str, optional
        Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear', 'bicubic'
        or 'cubic').
    """

    # scipy.misc.imresize should deal with uint8 image, or it would cause some problem (scale the image to [0, 255])
    return (scipy.misc.imresize(im2uint(image), size, interp=interp) / 127.5 - 1).astype(image.dtype)


def immerge(images, row, col):
    """
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


def center_crop(x, crop_h, crop_w=None):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return x[j:j + crop_h, i:i + crop_w]


def transform(image, resize_h=None, resize_w=None, interp='bilinear', crop_h=None, crop_w=None, crop_fn=center_crop):
    """
    transform a [-1.0, 1.0] image with crop and resize

    'crop_fn' should be in the form of crop_fn(x, crop_h, crop_w=None)
    """
    if crop_h is not None:
        image = crop_fn(image, crop_h, crop_w)

    if resize_h is not None:
        if resize_w is None:
            resize_w = resize_h
        imresize(image, [resize_h, resize_w], interp='bilinear')

    return image


def imread_transform(path, mode='RGB', resize_h=None, resize_w=None, interp='bilinear',
                     crop_h=None, crop_w=None, crop_fn=center_crop):
    """
    read and transform an image into [-1.0, 1.0] of float64

    `mode` can be one of the following strings:

    * 'L' (8 - bit pixels, black and white)
    * 'P' (8 - bit pixels, mapped to any other mode using a color palette)
    * 'RGB' (3x8 - bit pixels, true color)
    * 'RGBA' (4x8 - bit pixels, true color with transparency mask)
    * 'CMYK' (4x8 - bit pixels, color separation)
    * 'YCbCr' (3x8 - bit pixels, color video format)
    * 'I' (32 - bit signed integer pixels)
    * 'F' (32 - bit floating point pixels)

    'crop_fn' should be in the form of crop_fn(x, crop_h, crop_w=None)
    """
    return transform(imread(path, mode), resize_h, resize_w, interp, crop_h, crop_w, crop_fn)


def read_transform_images(path_list, mode='RGB', resize_h=None, resize_w=None, interp='bilinear',
                          crop_h=None, crop_w=None, crop_fn=center_crop):
    """ read and transform a list images into [-1.0, 1.0] of float64 and return the numpy array batch in shape of N * H * W (* C) """
    images = [imread_transform(path, mode, resize_h, resize_w, interp, crop_h, crop_w, crop_fn) for path in path_list]
    return np.array(images)


if __name__ == '__main__':
    pass
