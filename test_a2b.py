from __future__ import absolute_import, division, print_function

import argparse
from glob import glob
import os

import image_utils as im
import models
import numpy as np
import tensorflow as tf
import utils


""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', dest='dataset', default='summer2winter_yosemite', help='which dataset to use')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=256, help='then crop to this size')
args = parser.parse_args()

dataset = args.dataset
crop_size = args.crop_size


""" run """
with tf.Session() as sess:
    a_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])

    a2b = models.generator(a_real, 'a2b')

    # retore
    try:
        ckpt_path = utils.load_checkpoint('./outputs/checkpoints/' + dataset, sess)
    except:
        raise Exception('No checkpoint!')

    # test
    a_list = glob('./datasets/' + dataset + '/testA/*.jpg')

    a_save_dir = './outputs/test_predictions/' + dataset + '/testA'
    utils.mkdir([a_save_dir])

    for i in range(len(a_list)):
        a_real_ipt = im.imresize(im.imread(a_list[i]), [crop_size, crop_size])
        a_real_ipt.shape = 1, crop_size, crop_size, 3
        a2b_opt = sess.run([a2b], feed_dict={a_real: a_real_ipt})
        a_img_opt = np.concatenate(a2b_opt, axis=0)

        img_name = os.path.basename(a_list[i])
        im.imwrite(im.immerge(a_img_opt, 1, 1), a_save_dir + '/' + img_name)
        print('Save %s' % (a_save_dir + '/' + img_name))
