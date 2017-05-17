from __future__ import absolute_import, division, print_function

import utils
import models
import argparse
import numpy as np
import tensorflow as tf
import image_utils as im

from PIL import Image
from images2gif import writeGif


def gif_frames(gif, to_numpy=True):
    """
    convert a PIL gif to a list of RGB images

    to_numpy = True -> return numpy ([-1.0, 1.0], float64) images
    to_numpy = False -> return PIL ([0, 255], uint8) images
    """
    def iter_frame(gif):
        try:
            i = 0
            while 1:
                gif.seek(i)
                imframe = gif.copy()
                if i == 0:
                    palette = imframe.getpalette()
                else:
                    imframe.putpalette(palette)
                imframe = imframe.convert('RGB')
                yield imframe
                i += 1
        except EOFError:
            pass

    frames = []
    for frame in iter_frame(gif):
        if to_numpy:
            frames.append(np.array(frame) / 127.5 - 1)
        else:
            frames.append(frame)

    return frames


""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--gif', dest='gif', default='./pics/horse.gif', help='the input gif')
parser.add_argument('--save_path', dest='save_path', default='./pics/horse2zebra.gif', help='path to save the output gif')
parser.add_argument('--duration', dest='duration', type=float, default=0.07, help='duration of the output gif')
parser.add_argument('--dataset', dest='dataset', default='horse2zebra', help='which dataset to use')
parser.add_argument('--direction', dest='direction', default='a2b', help='translation direction')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=256, help='then crop to this size')
args = parser.parse_args()

gif_path = args.gif
save_path = args.save_path
duration = args.duration
dataset = args.dataset
direction = args.direction
crop_size = args.crop_size

assert direction == 'a2b' or direction == 'b2a', 'Direction should be a2b or b2a!'


""" run """
frames = []
a_reals_ipt_ori = gif_frames(Image.open(gif_path))
size_ori = a_reals_ipt_ori[0].shape[0:2]
with tf.Session() as sess:
    a_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
    a2b = models.generator(a_real, direction)

    # retore
    saver = tf.train.Saver()
    ckpt_path = utils.load_checkpoint('./checkpoints/' + dataset, sess, saver)
    if ckpt_path is None:
        raise Exception('No checkpoint!')
    else:
        print('Copy variables from % s' % ckpt_path)

    for a_real_ipt_ori in a_reals_ipt_ori:
        a_real_ipt = im.imresize(a_real_ipt_ori, [crop_size, crop_size])
        a_real_ipt.shape = 1, crop_size, crop_size, 3
        a2b_opt = sess.run(a2b, feed_dict={a_real: a_real_ipt})

        a2b_opt_ori = im.imresize(a2b_opt.squeeze(), size_ori)
        img_opt_ori = np.array([a_real_ipt_ori, a2b_opt_ori])
        img_opt_ori = im.im2uint(im.immerge(img_opt_ori, 1, 2))
        frames.append(img_opt_ori)

writeGif(save_path, frames, duration)
print('save in %s' % save_path)
