import tensorflow as tf
import numpy as np
import os
import sys

from mnist_data_layers import load_mnist_data, get_variable, conv2d_stride_2_valid, avg_pool_3x3_same_size


def cryptonets_model(x, mode):
    if mode not in set(['train', 'test']):
        print('mode should be train or test')
        raise Exception()
	
    with tf.name_scope('reshape'):
        print('padding')
        paddings = [[0, 0], [0, 1], [0, 1], [0, 0]]
        x = tf.pad(x, paddings)
        print('padded')

    # conv layer 1, Input: N x 28 x 28 x 1, Output: N x 13 x 13 x 5
    with tf.name_scope('conv1'):
        W_conv1 = get_variable("W_conv1", [5, 5, 1, 5], mode)
        h_conv1 = tf.square(conv2d_stride_2_valid(x, W_conv1))

    # pooling layer 1, Input: N x 13 x 13 x 5, Output: N x 13 x 13 x 5
    with tf.name_scope('pool1'):
        h_pool1 = avg_pool_3x3_same_size(h_conv1)

    # conv layer 2, Input: N x 13 x 13 x 5, Output: N x 5 x 5 x 50
    with tf.name_scope('conv2'):
        W_conv2 = get_variable("W_conv2", [5, 5, 5, 50], mode)
        h_conv2 = conv2d_stride_2_valid(h_pool1, W_conv2)

    # pooling layer 2, Input: N x 5 x 5 x 50, Output: N x 5 x 5 x 50
    with tf.name_scope('pool2'):
        h_pool2 = avg_pool_3x3_same_size(h_conv2)

    # fc layer 1, Input: N x 5 x 5 x 50, Output: N x 100
    with tf.name_scope('fc1'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 50])
        W_fc1 = get_variable("W_fc1", [5 * 5 * 50, 100], mode)
        h_fc1 = tf.square(tf.matmul(h_pool2_flat, W_fc1))

    # fc layer 2, Input: N x 100, Output: N x 10
    with tf.name_scope('fc2'):
        W_fc2 = get_variable("W_fc2", [100, 10], mode)
        y_conv = tf.matmul(h_fc1, W_fc2)
    return y_conv
