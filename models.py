from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import ops
import functools
import tensorflow as tf
import tensorflow.contrib.slim as slim


conv = functools.partial(slim.conv2d, activation_fn=None)
deconv = functools.partial(slim.conv2d_transpose, activation_fn=None)
relu = tf.nn.relu
lrelu = functools.partial(ops.leak_relu, leak=0.2)


def discriminator(img, scope, df_dim=64, reuse=False, train=True):

    bn = functools.partial(slim.batch_norm, scale=True, is_training=train,
                           decay=0.9, epsilon=1e-5, updates_collections=None)

    with tf.variable_scope(scope + '_discriminator', reuse=reuse):
        h0 = lrelu(conv(img, df_dim, 4, 2, scope='h0_conv'))    # h0 is (128 x 128 x df_dim)
        h1 = lrelu(bn(conv(h0, df_dim * 2, 4, 2, scope='h1_conv'), scope='h1_bn'))  # h1 is (64 x 64 x df_dim*2)
        h2 = lrelu(bn(conv(h1, df_dim * 4, 4, 2, scope='h2_conv'), scope='h2_bn'))  # h2 is (32x 32 x df_dim*4)
        h3 = lrelu(bn(conv(h2, df_dim * 8, 4, 1, scope='h3_conv'), scope='h3_bn'))  # h3 is (32 x 32 x df_dim*8)
        h4 = conv(h3, 1, 4, 1, scope='h4_conv')  # h4 is (32 x 32 x 1)

        return h4


def generator(img, scope, gf_dim=64, reuse=False, train=True):

    bn = functools.partial(slim.batch_norm, scale=True, is_training=train,
                           decay=0.9, epsilon=1e-5, updates_collections=None)

    def residule_block(x, dim, scope='res'):
        y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = relu(bn(conv(y, dim, 3, 1, padding='VALID', scope=scope + '_conv1'), scope=scope + '_bn1'))
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = bn(conv(y, dim, 3, 1, padding='VALID', scope=scope + '_conv2'), scope=scope + '_bn2')
        return y + x

    with tf.variable_scope(scope + '_generator', reuse=reuse):
        c0 = tf.pad(img, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = relu(bn(conv(c0, gf_dim, 7, 1, padding='VALID', scope='c1_conv'), scope='c1_bn'))
        c2 = relu(bn(conv(c1, gf_dim * 2, 3, 2, scope='c2_conv'), scope='c2_bn'))
        c3 = relu(bn(conv(c2, gf_dim * 4, 3, 2, scope='c3_conv'), scope='c3_bn'))

        r1 = residule_block(c3, gf_dim * 4, scope='r1')
        r2 = residule_block(r1, gf_dim * 4, scope='r2')
        r3 = residule_block(r2, gf_dim * 4, scope='r3')
        r4 = residule_block(r3, gf_dim * 4, scope='r4')
        r5 = residule_block(r4, gf_dim * 4, scope='r5')
        r6 = residule_block(r5, gf_dim * 4, scope='r6')
        r7 = residule_block(r6, gf_dim * 4, scope='r7')
        r8 = residule_block(r7, gf_dim * 4, scope='r8')
        r9 = residule_block(r8, gf_dim * 4, scope='r9')

        d1 = relu(bn(deconv(r9, gf_dim * 2, 3, 2, scope='d1_dconv'), scope='d1_bn'))
        d2 = relu(bn(deconv(d1, gf_dim, 3, 2, scope='d2_dconv'), scope='d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = conv(d2, 3, 7, 1, padding='VALID', scope='pred_conv')
        pred = tf.nn.tanh(pred)

        return pred
