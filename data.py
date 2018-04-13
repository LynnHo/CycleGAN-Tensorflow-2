from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ImageData:

    def __init__(self,
                 session,
                 image_paths,
                 batch_size,
                 load_size=286,
                 crop_size=256,
                 channels=3,
                 prefetch_batch=2,
                 drop_remainder=True,
                 num_threads=16,
                 shuffle=True,
                 buffer_size=4096,
                 repeat=-1):

        self._sess = session
        self._img_batch = ImageData._image_batch(image_paths,
                                                 batch_size,
                                                 load_size,
                                                 crop_size,
                                                 channels,
                                                 prefetch_batch,
                                                 drop_remainder,
                                                 num_threads,
                                                 shuffle,
                                                 buffer_size,
                                                 repeat)
        self._img_num = len(image_paths)

    def __len__(self):
        return self._img_num

    def batch(self):
        return self._sess.run(self._img_batch)

    @staticmethod
    def _image_batch(image_paths,
                     batch_size,
                     load_size=286,
                     crop_size=256,
                     channels=3,
                     prefetch_batch=2,
                     drop_remainder=True,
                     num_threads=16,
                     shuffle=True,
                     buffer_size=4096,
                     repeat=-1):
        def _parse_func(path):
            img = tf.read_file(path)
            img = tf.image.decode_jpeg(img, channels=channels)
            img = tf.image.random_flip_left_right(img)
            img = tf.image.resize_images(img, [load_size, load_size])
            img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
            img = tf.random_crop(img, [crop_size, crop_size, channels])
            img = img * 2 - 1
            return img

        dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        dataset = dataset.map(_parse_func, num_parallel_calls=num_threads)

        if shuffle:
            dataset = dataset.shuffle(buffer_size)

        if drop_remainder:
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        else:
            dataset = dataset.batch(batch_size)

        dataset = dataset.repeat(repeat).prefetch(prefetch_batch)

        return dataset.make_one_shot_iterator().get_next()
