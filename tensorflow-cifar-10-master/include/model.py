from tensorflow.compat import v1 as tf 


def model():
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10

    with tf.name_scope('main_params'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    with tf.variable_scope('conv1') as scope:
        paddings = [[0, 0], [0, 0], [0, 0], [0, 0]]
        conv = tf.layers.conv2d(
            inputs=x_image,
            filters=128,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding=paddings,
            activation=tf.nn.relu
        )
        pool = tf.nn.avg_pool(conv, ksize=[2, 2], strides=[2, 2], padding='SAME')
        conv = tf.layers.conv2d(
            inputs=pool,
            filters=83,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding='SAME',
            activation=tf.math.square
        )
        pool = tf.nn.avg_pool(conv, ksize=[2, 2], strides=[2, 2], padding='SAME')
        conv = tf.layers.conv2d(
            inputs=pool,
            filters=163,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding=paddings,
            activation=tf.math.square
        )
        pool = tf.nn.avg_pool(conv, ksize=[2, 2], strides=[2, 2], padding='SAME')
        drop = tf.layers.dropout(pool, rate=0.5)

    with tf.variable_scope('fully_connected') as scope:
        flat = tf.reshape(drop, [-1, 3 * 3 * 163])
        fc = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
        drop = tf.layers.dropout(fc, rate=0.5)
        softmax = tf.layers.dense(inputs=drop, units=_NUM_CLASSES, name=scope.name)

    y_pred_cls = tf.argmax(softmax, axis=1)

    return x, y, softmax, y_pred_cls, global_step, learning_rate


def lr(epoch):
    learning_rate = 1e-2
    return learning_rate
