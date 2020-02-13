import argparse
import time
import numpy as np
import tensorflow as tf
import ngraph_bridge
from tensorflow.core.protobuf import rewriter_config_pb2

from mnist_data_layers import load_mnist_data, get_variable, conv2d_stride_2_valid, str2bool, server_argument_parser, server_config_from_flags

def cryptonets_test_flattened(x):
    paddings = [[0, 0], [0, 1], [0, 1], [0, 0]]
    x = tf.pad(x, paddings)
    W_conv1 = get_variable('W_conv1', [5, 5, 1, 5], 'test')
    y = conv2d_stride_2_valid(x, W_conv1)
    y = tf.square(y)
    W_flatten = get_variable('W_flatten', [5 * 13 * 13, 100], 'test')
    y = tf.reshape(y, [-1, 5 * 13 * 13])
    y = tf.matmul(y, W_flatten)
    y = tf.square(y)
    W_fc2 = get_variable('W_fc2', [100, 10], 'test')
    y = tf.matmul(y, W_fc2)
    return y


def test_mnist_cnn(FLAGS):
    (x_train, y_train, x_test, y_test) = load_mnist_data()
    x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1], name='input')
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])
    y_conv = cryptonets_test_flattened(x)
    config = server_config_from_flags(FLAGS, x.name)
    print('config', config)
    with tf.compat.v1.Session(config=config) as sess:
        x_test = x_test[:FLAGS.batch_size]
        y_test = y_test[:FLAGS.batch_size]
        start_time = time.time()
        y_conv_val = y_conv.eval(feed_dict={x: x_test, y_: y_test})
        elasped_time = (time.time() - start_time)
        print("total time(s)", np.round(elasped_time, 3))
        print('y_conv_val', np.round(y_conv_val, 2))

    y_test_batch = y_test[:FLAGS.batch_size]
    y_label_batch = np.argmax(y_test_batch, 1)
    correct_prediction = np.equal(np.argmax(y_conv_val, 1), y_label_batch)
    error_count = np.size(correct_prediction) - np.sum(correct_prediction)
    test_accuracy = np.mean(correct_prediction)

    print('Misclassification:', error_count, 'of', FLAGS.batch_size, 'elements.')
    print('Accuracy: ', test_accuracy)


if __name__ == '__main__':
    parser = server_argument_parser()
    FLAGS, unparsed = parser.parse_known_args()

    if unparsed:
        print('Unparsed flags:', unparsed)
    if FLAGS.encrypt_server_data and FLAGS.enable_client:
        raise Exception("encrypt_server_data flag only valid when client is not enabled. Note: the client can specify whether or not to encrypt the data using 'encrypt' or 'plain' in the configuration map")

    test_mnist_cnn(FLAGS)