import tensorflow as tf
import numpy as np
import argparse
from tensorflow.core.protobuf import rewriter_config_pb2


def load_cifar_data():
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = tf.compat.v1.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.compat.v1.keras.utils.to_categorical(y_test, num_classes=10)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    return (x_train, y_train, x_test, y_test)


def get_train_batch(train_iter, batch_size, x_train, y_train):
    si = train_iter * batch_size
    ei = si + batch_size

    data_count = x_train.shape[0]

    if si > data_count and ei > data_count:
        si %= data_count
        ei %= data_count
        x_batch = x_train[si:ei]
        y_batch = y_train[si:ei]
    elif ei > data_count:
        ei %= data_count
        x_batch = np.concatenate((x_train[si:], x_train[0:ei]))
        y_batch = np.concatenate((y_train[si:], y_train[0:ei]))
    else:
        x_batch = x_train[si:ei]
        y_batch = y_train[si:ei]

    return x_batch, y_batch


def conv2d_stride_2_valid(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')


def avg_pool_3x3_same_size(x):
    return tf.nn.avg_pool2d(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')


def max_pool_3x3_same_size(x):
    return tf.nn.max_pool2d(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')


def get_variable(name, shape, mode):
    if mode not in set(['train', 'test']):
        print('mode should be train or test')
        raise Exception()

    if mode == 'train':
        return tf.compat.v1.get_variable(name, shape)
    else:
        return tf.constant(np.loadtxt(name + '.txt', dtype=np.float32).reshape(shape))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def server_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--enable_client', type=str2bool, default=False, help='Enable the client')
    parser.add_argument('--backend', type=str, default='HE_SEAL', help='Name of backend to use')
    parser.add_argument('--encryption_parameters', type=str, default='', help= 'Filename containing json description of encryption parameters, or json description itself')
    parser.add_argument('--encrypt_server_data', type=str2bool, default=False, help= 'Encrypt server data (should not be used when enable_client is used)')
    parser.add_argument('--pack_data', type=str2bool, default=True, help='Use plaintext packing on data')
    return parser


def server_config_from_flags(FLAGS, tensor_param_name):
    rewriter_options = rewriter_config_pb2.RewriterConfig()
    rewriter_options.meta_optimizer_iterations = (rewriter_config_pb2.RewriterConfig.ONE)
    rewriter_options.min_graph_nodes = -1
    server_config = rewriter_options.custom_optimizers.add()
    server_config.name = "ngraph-optimizer"
    server_config.parameter_map["ngraph_backend"].s = FLAGS.backend.encode()
    server_config.parameter_map["device_id"].s = b''
    server_config.parameter_map["encryption_parameters"].s = FLAGS.encryption_parameters.encode()
    server_config.parameter_map['enable_client'].s = str(FLAGS.enable_client).encode()

    if FLAGS.enable_client:
        server_config.parameter_map[tensor_param_name].s = b'client_input'
    elif FLAGS.encrypt_server_data:
        server_config.parameter_map[tensor_param_name].s = b'encrypt'
    else:
        server_config.parameter_map[tensor_param_name].s = b'plain'

    if FLAGS.pack_data:
        server_config.parameter_map[tensor_param_name].s += b',packed'

    config = tf.compat.v1.ConfigProto()
    config.MergeFrom(tf.compat.v1.ConfigProto(graph_options=tf.compat.v1.GraphOptions(rewrite_options=rewriter_options)))

    return config