"""CNN GRU for inertial data"""

import tensorflow as tf

def cg1(self, inputs, is_training):
    seq_pool=1
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=16 => seq_length= 128 /16 = 8
    #inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    #inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    #seq_pool=seq_pool*2 # seq_pool=32 => seq_length= 128 /32 = 4
    #inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    #inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    #seq_pool=seq_pool*2 # seq_pool=64 => seq_length= 128 /64 = 2
    #inputs = tf.keras.layers.Conv1D(filters=512, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    #inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    #seq_pool=seq_pool*2 # seq_pool=128 => seq_length= 128 /128 = 1

    inputs = tf.keras.layers.Dense(8)(inputs)

    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs


def cg2(self, inputs, is_training):
    seq_pool=1
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=16 => seq_length= 128 /16 = 8

    inputs = tf.keras.layers.Dense(8)(inputs)

    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cg3(self, inputs, is_training):
    seq_pool=1
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=16 => seq_length= 128 /16 = 8

    inputs = tf.keras.layers.Dense(8)(inputs)

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cg3_1(self, inputs, is_training):
    seq_pool=1
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=16 => seq_length= 128 /16 = 8

    inputs = tf.keras.layers.Dense(4)(inputs)

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cg3_1_1(self, inputs, is_training):
    seq_pool=1
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=16 => seq_length= 128 /16 = 8
    inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=32 => seq_length= 128 /32 = 4

    inputs = tf.keras.layers.Dense(4)(inputs)

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cg3_2(self, inputs, is_training):
    seq_pool=1
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=16 => seq_length= 128 /16 = 8

    inputs = tf.keras.layers.Dense(2)(inputs)

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cg3_2_1(self, inputs, is_training):
    seq_pool=1
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=16 => seq_length= 128 /16 = 8
    inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=32 => seq_length= 128 /32 = 4
    inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=64 => seq_length= 128 /64 = 2

    inputs = tf.keras.layers.Dense(2)(inputs)

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cg4(self, inputs, is_training):
    seq_pool=1
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=16 => seq_length= 128 /16 = 8

    inputs = tf.keras.layers.Dense(16)(inputs)

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cg4_1(self, inputs, is_training):
    seq_pool=1
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16

    inputs = tf.keras.layers.Dense(16)(inputs)

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(64, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cg5(self, inputs, is_training):
    seq_pool=1
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
    inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=16 => seq_length= 128 /16 = 8

    inputs = tf.keras.layers.Dense(8)(inputs)

    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    inputs = tf.keras.layers.GRU(128, return_sequences=True)(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs


class Model(object):
    """Base class for CNN GRU model."""

    def __init__(self, params):
        self.params = params
        self.num_classes = params.num_classes
        self.sub_mode = params.sub_mode
    def __call__(self, inputs, is_training):
        var_scope = 'gru'
        with tf.variable_scope(var_scope):
            if self.sub_mode == 'cg1':
                return cg1(self, inputs, is_training)
            elif self.sub_mode == 'cg2':
                return cg2(self, inputs, is_training)
            elif self.sub_mode == 'cg3':
                return cg3(self, inputs, is_training)
            elif self.sub_mode == 'cg3_1':
                return cg3_1(self, inputs, is_training)
            elif self.sub_mode == 'cg3_1_1':
                return cg3_1_1(self, inputs, is_training)
            elif self.sub_mode == 'cg3_2':
                return cg3_2(self, inputs, is_training)
            elif self.sub_mode == 'cg3_2_1':
                return cg3_2_1(self, inputs, is_training)
            elif self.sub_mode == 'cg4':
                return cg4(self, inputs, is_training)
            elif self.sub_mode == 'cg4_1':
                return cg4_1(self, inputs, is_training)
            elif self.sub_mode == 'cg5':
                return cg5(self, inputs, is_training)
            else:
                raise RuntimeError('sub mode {0} is not implemented'.format(self.sub_mode))

