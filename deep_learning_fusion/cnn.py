"""CNN for inertial data"""

import tensorflow as tf

#def c4_old(self, inputs, is_training):
#    seq_pool=1
#    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
#    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
#    seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
#    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
#    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
#    seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
#    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
#    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
#    seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16
#    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
#    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
#    seq_pool=seq_pool*2 # seq_pool=16 => seq_length= 128 /16 = 8
#    return seq_pool, inputs

def c13(self, inputs, is_training):
    seq_pool=1
    inputs = tf.keras.layers.Conv1D(filters=512, kernel_size=1, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=1, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=3, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=7, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=7, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=7, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=7, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=9, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=9, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=9, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=9, padding='same', activation=tf.nn.relu)(inputs)
    return seq_pool, inputs

def c(inputs, depth, kernel_sizes, max_kernel_size, filters, padding, padding_size, dense_units):
    def cond1d(inputs, filters, kernel_size, padding):
        return tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, activation=tf.nn.relu)(inputs)
    seq_pool=1 # seq_pool=1 => seq_length= seq_length/seq_pool = 128 /1 = 128
    for i in range(depth):
        kernel_size = kernel_sizes[i]
        kernel_size = kernel_size if kernel_size<=max_kernel_size else max_kernel_size
        if padding=='valid':
            padding_size+=kernel_size-1
        inputs = cond1d(inputs, filters, kernel_size, padding)
    if dense_units>0:
        inputs = tf.keras.layers.Dense(dense_units)(inputs)
    return seq_pool, padding_size, inputs

class Model(object):
    """Base class for CNN model."""

    def __init__(self, params):
        self.sub_mode = params.sub_mode
    def __call__(self, inputs, var_scope_suffix, is_training, padding_size):
        var_scope = 'cnn' + var_scope_suffix
        with tf.variable_scope(var_scope):
            sub_mode = self.sub_mode.split('|')[0]
            sub_mode_dict = dict(item.split(':') for item in sub_mode.split(';'))
            depth = int(sub_mode_dict['d']) 
            kernel_sizes = [int(item) for item in str(sub_mode_dict['ks'])] if 'ks' in sub_mode_dict else list(range(1,int(depth)*2,2)) 
            filters = int(sub_mode_dict['fs']) if 'fs' in sub_mode_dict else 128 
            padding = sub_mode_dict['pad'] if 'pad' in sub_mode_dict else 'same' 
            max_kernel_size = int(sub_mode_dict['mks']) if 'mks' in sub_mode_dict else 9 
            dense_units = int(sub_mode_dict['du']) if 'du' in sub_mode_dict else 0 
            return c(inputs, depth, kernel_sizes, max_kernel_size, filters, padding, padding_size, dense_units)



