from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
from glob import glob

import data
import image_utils as im
import models
import numpy as np
import tensorflow as tf
import utils


""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', dest='dataset', default='horse2zebra', help='which dataset to use')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in a batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
args = parser.parse_args()

dataset = args.dataset
load_size = args.load_size
crop_size = args.crop_size
epoch = args.epoch
batch_size = args.batch_size
lr = args.lr


""" graph """
# models
generator_a2b = partial(models.generator, scope='a2b')
generator_b2a = partial(models.generator, scope='b2a')
discriminator_a = partial(models.discriminator, scope='a')
discriminator_b = partial(models.discriminator, scope='b')

# operations
a_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
b_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
a2b_sample = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
b2a_sample = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])

a2b = generator_a2b(a_real)
b2a = generator_b2a(b_real)
b2a2b = generator_a2b(b2a)
a2b2a = generator_b2a(a2b)

a_logit = discriminator_a(a_real)
b2a_logit = discriminator_a(b2a)
b2a_sample_logit = discriminator_a(b2a_sample)
b_logit = discriminator_b(b_real)
a2b_logit = discriminator_b(a2b)
a2b_sample_logit = discriminator_b(a2b_sample)

# losses
g_loss_a2b = tf.losses.mean_squared_error(a2b_logit, tf.ones_like(a2b_logit))
g_loss_b2a = tf.losses.mean_squared_error(b2a_logit, tf.ones_like(b2a_logit))
cyc_loss_a = tf.losses.absolute_difference(a_real, a2b2a)
cyc_loss_b = tf.losses.absolute_difference(b_real, b2a2b)
g_loss = g_loss_a2b + g_loss_b2a + cyc_loss_a * 10.0 + cyc_loss_b * 10.0

d_loss_a_real = tf.losses.mean_squared_error(a_logit, tf.ones_like(a_logit))
d_loss_b2a_sample = tf.losses.mean_squared_error(b2a_sample_logit, tf.zeros_like(b2a_sample_logit))
d_loss_a = d_loss_a_real + d_loss_b2a_sample

d_loss_b_real = tf.losses.mean_squared_error(b_logit, tf.ones_like(b_logit))
d_loss_a2b_sample = tf.losses.mean_squared_error(a2b_sample_logit, tf.zeros_like(a2b_sample_logit))
d_loss_b = d_loss_b_real + d_loss_a2b_sample

# summaries
g_summary = utils.summary({g_loss_a2b: 'g_loss_a2b',
                           g_loss_b2a: 'g_loss_b2a',
                           cyc_loss_a: 'cyc_loss_a',
                           cyc_loss_b: 'cyc_loss_b'})
d_summary_a = utils.summary({d_loss_a: 'd_loss_a'})
d_summary_b = utils.summary({d_loss_b: 'd_loss_b'})

# optim
t_var = tf.trainable_variables()
d_a_var = [var for var in t_var if 'a_discriminator' in var.name]
d_b_var = [var for var in t_var if 'b_discriminator' in var.name]
g_var = [var for var in t_var if 'a2b_generator' in var.name or 'b2a_generator' in var.name]

d_a_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss_a, var_list=d_a_var)
d_b_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss_b, var_list=d_b_var)
g_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_var)


""" train """
''' init '''
# session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# counter
it_cnt, update_cnt = utils.counter()

''' data '''
a_img_paths = glob('./datasets/' + dataset + '/trainA/*.jpg')
b_img_paths = glob('./datasets/' + dataset + '/trainB/*.jpg')
a_data_pool = data.ImageData(sess, a_img_paths, batch_size, load_size=load_size, crop_size=crop_size)
b_data_pool = data.ImageData(sess, b_img_paths, batch_size, load_size=load_size, crop_size=crop_size)

a_test_img_paths = glob('./datasets/' + dataset + '/testA/*.jpg')
b_test_img_paths = glob('./datasets/' + dataset + '/testB/*.jpg')
a_test_pool = data.ImageData(sess, a_test_img_paths, batch_size, load_size=load_size, crop_size=crop_size)
b_test_pool = data.ImageData(sess, b_test_img_paths, batch_size, load_size=load_size, crop_size=crop_size)

a2b_pool = utils.ItemPool()
b2a_pool = utils.ItemPool()

''' summary '''
summary_writer = tf.summary.FileWriter('./outputs/summaries/' + dataset, sess.graph)

''' saver '''
saver = tf.train.Saver(max_to_keep=5)

''' restore '''
ckpt_dir = './outputs/checkpoints/' + dataset
utils.mkdir(ckpt_dir)
try:
    utils.load_checkpoint(ckpt_dir, sess)
except:
    # Save a graph definition (once)
    tf.train.write_graph(sess.graph.as_graph_def(), "", './outputs/checkpoints/' + dataset + 'graph.pb')
    
    # Weights initialization
    sess.run(tf.global_variables_initializer())

'''train'''
try:
    batch_epoch = min(len(a_data_pool), len(b_data_pool)) // batch_size
    max_it = epoch * batch_epoch
    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        # prepare data
        a_real_ipt = a_data_pool.batch()
        b_real_ipt = b_data_pool.batch()
        a2b_opt, b2a_opt = sess.run([a2b, b2a], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
        a2b_sample_ipt = np.array(a2b_pool(list(a2b_opt)))
        b2a_sample_ipt = np.array(b2a_pool(list(b2a_opt)))

        # train G
        g_summary_opt, _ = sess.run([g_summary, g_train_op], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
        summary_writer.add_summary(g_summary_opt, it)
        # train D_b
        d_summary_b_opt, _ = sess.run([d_summary_b, d_b_train_op], feed_dict={b_real: b_real_ipt, a2b_sample: a2b_sample_ipt})
        summary_writer.add_summary(d_summary_b_opt, it)
        # train D_a
        d_summary_a_opt, _ = sess.run([d_summary_a, d_a_train_op], feed_dict={a_real: a_real_ipt, b2a_sample: b2a_sample_ipt})
        summary_writer.add_summary(d_summary_a_opt, it)

        # display
        if it % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        # save
        if (it + 1) % 1000 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        if (it + 1) % 100 == 0:
            a_real_ipt = a_test_pool.batch()
            b_real_ipt = b_test_pool.batch()
            [a2b_opt, a2b2a_opt, b2a_opt, b2a2b_opt] = sess.run([a2b, a2b2a, b2a, b2a2b], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
            sample_opt = np.concatenate((a_real_ipt, a2b_opt, a2b2a_opt, b_real_ipt, b2a_opt, b2a2b_opt), axis=0)

            save_dir = './outputs/sample_images_while_training/' + dataset
            utils.mkdir(save_dir)
            im.imwrite(im.immerge(sample_opt, 2, 3), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))
except:
    save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
    print('Model saved in file: % s' % save_path)
    sess.close()
