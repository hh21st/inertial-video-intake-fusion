"""Small CNN for inertial data """

import tensorflow as tf

SCOPE = "small_cnn"

class Model(object):
    """Base class for 7 layer CNN model."""

    def __init__(self, params):
        self.num_filters = params.small_num_filters
        self.kernel_size = params.small_kernel_size
        self.pool_size = params.small_pool_size
        self.num_classes = params.num_classes

    def __call__(self, inputs, is_training, scope=SCOPE):
        with tf.variable_scope(scope):

            for i, num_filters in enumerate(self.num_filters):
                inputs = tf.keras.layers.Conv1D(
                    filters=num_filters,
                    kernel_size=self.kernel_size,
                    padding='same',
                    activation=tf.nn.relu)(inputs)
                inputs = tf.keras.layers.MaxPool1D(
                    pool_size=self.pool_size)(inputs)

            inputs = tf.keras.layers.Flatten()(inputs)
            inputs = tf.keras.layers.Dense(self.num_classes)(inputs)

            return inputs
