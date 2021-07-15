"""ResNet-style CNN for inertial data
Based on github.com/tensorflow/models/blob/master/official/resnet"""

import tensorflow as tf

SCOPE = "resnet_cnn"
BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5


def batch_norm(inputs):
    """Performs a batch normalization using a standard set of parameters."""
    return tf.keras.layers.BatchNormalization(
        axis=2, momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON, center=True, scale=True, fused=True,
        name='batch_normalization')(inputs)


def fixed_padding(inputs, kernel_size):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
        inputs: A tensor of size [batch, seq, channels].
        kernel_size: The kernel to be used in the conv1d or max_pool1d operation.
            Should be a positive integer.
    Returns:
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    paddings = [[0, 0], [pad_beg, pad_end], [0, 0]]

    padded_inputs = tf.pad(tensor=inputs, paddings=paddings)

    return padded_inputs


def conv1d_fixed_padding(inputs, filters, kernel_size, strides):
    """Strided 1-D convolution with explicit padding."""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)
    return tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())(inputs)


def conv1d_bottleneck_block_v2(inputs, filters, is_training, projection_shortcut, strides):
    """A single block for ResNet v2 with bottleneck - adapted for 1D

    Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    Adapted to the ordering conventions of: Batch normalization then ReLU
    then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

    Args:
        inputs: A tensor of size [batch, seq, channels].
        filters: The number of filters for convolutions.
        is_training: Boolean indicating training or inference mode.
        projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
        strides: The block's stride.

    Returns:
        The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    inputs = batch_norm(inputs)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv1d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1)

    inputs = batch_norm(inputs)
    inputs = tf.nn.relu(inputs)

    inputs = conv1d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=2, strides=strides)

    inputs = batch_norm(inputs)
    inputs = tf.nn.relu(inputs)

    inputs = conv1d_fixed_padding(
        inputs=inputs, filters=4*filters, kernel_size=1, strides=1)

    return inputs + shortcut


def conv1d_block_layer(inputs, filters, blocks, strides, is_training, name):
    """Create one layer of blocks for a 1D ResNet model.

    Args:
        inputs: A tensor of size [batch, seq, channels].
        filters: The number of filters for the first convolution of the layer.
        blocks: The number of blocks contained in the layer.
        strides: The stride to use for the first convolution of the layer. If
            greater than 1, this layer will ultimately downsample the input.
        is_training: Are we currently training the model?
        name: A string name for the tensor output of the block layer.

    Returns:
        The output tensor of the block layer.
    """
    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4

    def _projection_shortcut(inputs):
        return conv1d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = conv1d_bottleneck_block_v2(
        inputs=inputs, filters=filters, is_training=is_training,
        projection_shortcut=_projection_shortcut, strides=strides)

    for _ in range(1, blocks):
        inputs = conv1d_bottleneck_block_v2(
            inputs=inputs, filters=filters, is_training=is_training,
            projection_shortcut=None, strides=1)

    return tf.identity(inputs, name)


class Model(object):
    """Base class for ResNet 1D CNN model."""

    def __init__(self, params):
        self.block_sizes = params.resnet_block_sizes
        self.block_strides = params.resnet_block_strides
        self.conv_stride = params.resnet_conv_stride
        self.first_pool_size = params.resnet_first_pool_size
        self.first_pool_stride = params.resnet_first_pool_stride
        self.kernel_size = params.resnet_kernel_size
        self.num_classes = params.num_classes
        self.num_filters = params.resnet_num_filters
        self.num_lstm = params.num_lstm

    def __call__(self, inputs, is_training, scope=SCOPE):
        with tf.variable_scope(scope):

            # First conv layer
            inputs = conv1d_fixed_padding(
                inputs=inputs, filters=self.num_filters,
                kernel_size=self.kernel_size, strides=1)
            inputs = tf.identity(inputs, "initial_conv")

            # First pool layer
            inputs = tf.keras.layers.MaxPool1D(
                pool_size=self.first_pool_size, strides=self.first_pool_stride,
                padding='same')(inputs)

            for i, num_blocks in enumerate(self.block_sizes):
                num_filters = self.num_filters * (2**i)
                inputs = conv1d_block_layer(
                    inputs=inputs, filters=num_filters, blocks=num_blocks,
                    strides=self.block_strides[i], is_training=is_training,
                    name='block_layer{}'.format(i+1))

            inputs = batch_norm(inputs)
            inputs = tf.nn.relu(inputs)

            # Recurrent layer
            inputs = tf.keras.layers.LSTM(
                units=self.num_lstm, return_sequences=True)(inputs)
            inputs = tf.identity(inputs, 'lstm')

            # Dense
            logits = tf.keras.layers.Dense(units=self.num_classes)(inputs)
            logits = tf.identity(logits, 'final_dense')

            return inputs
