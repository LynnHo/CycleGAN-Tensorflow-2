from __future__ import absolute_import, division, print_function

import argparse
from glob import glob
import os

import image_utils as im
import models
import numpy as np
import tensorflow as tf
import utils
import time

""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--checkpoints', dest='checkpoints', default='./outputs/checkpoints/summer2winter_yosemite',
                    help='path of model checkpoint files')
parser.add_argument('--dataset', dest='dataset', default='./datasets/summer2winter_yosemite/testA',
                    help='path of images to process')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=256, help='then crop to this size')
args = parser.parse_args()

checkpoints = args.checkpoints
dataset = args.dataset
crop_size = args.crop_size

""" run """
with tf.Session() as sess:

    print("\nLoading Images...\n")
    print("Images will be cropped to:", (crop_size, crop_size))
    a_list = glob(dataset + '/*.jpg')

    print("Creating destination")
    a_save_dir = dataset + '/inference_results'
    utils.mkdir([a_save_dir])

    # Define the graph input
    a_input = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])

    # Define the graph
    a2b = models.generator(a_input, 'a2b')

    # Restore weights
    try:
        ckpt_path = utils.load_checkpoint(checkpoints, sess)
    except IOError as e:
        raise Exception('No checkpoint found!')

    start = time.time()
    # Inference
    for i in range(len(a_list)):
        # Define shapes for images fed to the graph
        a_feed = im.imresize(im.imread(a_list[i]), [crop_size, crop_size])
        a_feed.shape = 1, crop_size, crop_size, 3

        # Feed in images to the graph
        a2b_result = sess.run(a2b, feed_dict={a_input: a_feed})

        # Create and save the output image
        a_img_opt = np.concatenate((a_feed, a2b_result), axis=0)
        img_name = os.path.basename(a_list[i])
        im.imwrite(im.immerge(a_img_opt, 1, 2), a_save_dir + '/' + img_name)
        print('Save %s' % (a_save_dir + '/' + img_name))

        if i == 100:
            end = time.time()
    end2 = time.time()
    print("Time to process first 100 images:", end - start)
    print("Time to process all %d images: %f" % (i + 1, end2 - start))
