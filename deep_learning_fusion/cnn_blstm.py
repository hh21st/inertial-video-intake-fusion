"""CNN Bidirectional LSTM for inertial data"""

import tensorflow as tf

def cl1(self, inputs, is_training):
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

    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cl2(self, inputs, is_training):
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

    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cl3(self, inputs, is_training):
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

    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cl3_0(self, inputs, is_training):
    seq_pool=1
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16
    inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=16 => seq_length= 128 /16 = 8

    inputs = tf.keras.layers.Dense(8)(inputs)

    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(inputs)
    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cl3_0_1(self, inputs, is_training):
    seq_pool=1
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16
    inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=16 => seq_length= 128 /16 = 8

    inputs = tf.keras.layers.Dense(8)(inputs)

    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True))(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cl3_0_2(self, inputs, is_training):
    seq_pool=1
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16
    inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=16 => seq_length= 128 /16 = 8

    inputs = tf.keras.layers.Dense(16)(inputs)

    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True))(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cl3_1(self, inputs, is_training):
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

    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cl3_1_0(self, inputs, is_training):
    seq_pool=1
    inputs = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=2 => seq_length= seq_length/seq_pool = 128 /2 = 64
    inputs = tf.keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=4 => seq_length= 128 /4 = 32
    inputs = tf.keras.layers.Conv1D(filters=248, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.Conv1D(filters=256, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=8 => seq_length= 128 /8 = 16
    inputs = tf.keras.layers.Conv1D(filters=512, kernel_size=10, padding='same', activation=tf.nn.relu)(inputs)
    inputs = tf.keras.layers.MaxPool1D(pool_size=2)(inputs)
    seq_pool=seq_pool*2 # seq_pool=16 => seq_length= 128 /16 = 8

    inputs = tf.keras.layers.Dense(4)(inputs)

    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cl3_1_1(self, inputs, is_training):
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

    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cl3_1_1_0(self, inputs, is_training):
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

    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True))(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cl3_2(self, inputs, is_training):
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

    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cl3_2_1(self, inputs, is_training):
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

    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cl4(self, inputs, is_training):
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

    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cl4_1(self, inputs, is_training):
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

    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs

def cl5(self, inputs, is_training):
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

    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
    inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)

    inputs = tf.keras.layers.Dense(self.num_classes)(inputs)
    return seq_pool, inputs


class Model(object):
    """Base class for CNN BLSTM model."""

    def __init__(self, params):
        self.params = params
        self.num_classes = params.num_classes
        self.sub_mode = params.sub_mode
    def __call__(self, inputs, is_training):
        var_scope = 'blstm'
        with tf.variable_scope(var_scope):
            if self.sub_mode == 'cl1':
                return cl1(self, inputs, is_training)
            elif self.sub_mode == 'cl2':
                return cl2(self, inputs, is_training)
            elif self.sub_mode == 'cl3':
                return cl3(self, inputs, is_training)
            elif self.sub_mode == 'cl3_0':
                return cl3_0(self, inputs, is_training)
            elif self.sub_mode == 'cl3_0_1':
                return cl3_0_1(self, inputs, is_training)
            elif self.sub_mode == 'cl3_0_2':
                return cl3_0_2(self, inputs, is_training)
            elif self.sub_mode == 'cl3_1':
                return cl3_1(self, inputs, is_training)
            elif self.sub_mode == 'cl3_1_0':
                return cl3_1_0(self, inputs, is_training)
            elif self.sub_mode == 'cl3_1_1':
                return cl3_1_1(self, inputs, is_training)
            elif self.sub_mode == 'cl3_1_1_0':
                return cl3_1_1_0(self, inputs, is_training)
            elif self.sub_mode == 'cl3_2':
                return cl3_2(self, inputs, is_training)
            elif self.sub_mode == 'cl3_2_1':
                return cl3_2_1(self, inputs, is_training)
            elif self.sub_mode == 'cl4':
                return cl4(self, inputs, is_training)
            elif self.sub_mode == 'cl4_1':
                return cl4_1(self, inputs, is_training)
            elif self.sub_mode == 'cl5':
                return cl5(self, inputs, is_training)
            else:
                raise RuntimeError('sub mode {0} is not implemented'.format(self.sub_mode))

