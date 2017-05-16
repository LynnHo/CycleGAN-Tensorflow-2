from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import re
import tensorflow as tf


""" ops """


def leak_relu(x, leak, scope=None):
    with tf.name_scope(scope, 'leak_relu', [x, leak]):
        y = tf.maximum(x, leak * x)
        return y


""" loss """


def l2_loss(a, b, weights=1.0, scope=None):
    with tf.name_scope(scope, 'l2_loss', [a, b, weights]):
        loss = tf.reduce_mean((a - b) ** 2) * weights
        return loss


def l1_loss(a, b, weights=1.0, scope=None):
    with tf.name_scope(scope, 'l1_loss', [a, b, weights]):
        loss = tf.reduce_mean(tf.abs(a - b)) * weights
        return loss


""" summary """


def summary(tensor, summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram']):
    """ Attach a lot of summaries to a Tensor. """

    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', tensor.name)
    tensor_name = re.sub(':', '-', tensor_name)

    with tf.name_scope('summary_' + tensor_name):
        summaries = []
        if len(tensor._shape) == 0:
            summaries.append(tf.summary.scalar(tensor_name, tensor))
        else:
            if 'mean' in summary_type:
                mean = tf.reduce_mean(tensor)
                summaries.append(tf.summary.scalar(tensor_name + '/mean', mean))
            if 'stddev' in summary_type:
                mean = tf.reduce_mean(tensor)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
                summaries.append(tf.summary.scalar(tensor_name + '/stddev', stddev))
            if 'max' in summary_type:
                summaries.append(tf.summary.scalar(tensor_name + '/max', tf.reduce_max(tensor)))
            if 'min' in summary_type:
                summaries.append(tf.summary.scalar(tensor_name + '/min', tf.reduce_min(tensor)))
            if 'sparsity' in summary_type:
                summaries.append(tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(tensor)))
            if 'histogram' in summary_type:
                summaries.append(tf.summary.histogram(tensor_name, tensor))
        return tf.summary.merge(summaries)


def summary_tensors(tensors, summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram']):
    with tf.name_scope('summary_tensors'):
        summaries = []
        for tensor in tensors:
            summaries.append(summary(tensor, summary_type))
        return tf.summary.merge(summaries)


""" others """


def counter(scope='counter'):
    with tf.variable_scope(scope):
        counter = tf.Variable(0, dtype=tf.int32, name='counter')
        update_cnt = tf.assign(counter, tf.add(counter, 1))
        return counter, update_cnt
