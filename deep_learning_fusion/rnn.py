"""RNN for inertial data"""

import tensorflow as tf

def r(inputs, type, depth, units, num_classes, has_dense):
    def lstm(inputs, units):
        return tf.keras.layers.LSTM(units=units, return_sequences=True)(inputs)
    def blstm(inputs, units):
        return tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=units, return_sequences=True))(inputs)
    def gru(inputs, units):
        return tf.keras.layers.GRU(units=units, return_sequences=True)(inputs)
    def get_rnn_layer(type, inputs, units):
        if type == 'l':
            return lstm(inputs,units)
        elif type == 'b':
            return blstm(inputs,units)
        elif type == 'g':
            return gru(inputs,units)
        else:
            raise RuntimeError('type {0} is not defined'.format(type))
    for i in range(depth):
        inputs = get_rnn_layer(type, inputs, units)
    if depth == 0:
        inputs = tf.keras.layers.Flatten()(inputs)
    if has_dense:
        inputs = tf.keras.layers.Dense(num_classes)(inputs)
    return inputs


class Model(object):
    """Base class for LSTM model."""

    def __init__(self, params):
        self.params = params
        self.num_classes = params.num_classes
        self.sub_mode = params.sub_mode
    def __call__(self, inputs, var_scope_suffix, is_training, has_dense = True):
        var_scope = 'rnn' + var_scope_suffix
        with tf.variable_scope(var_scope):
            sub_mode = self.sub_mode.split('|')[1]
            sub_mode_dict = dict(item.split(':') for item in sub_mode.split(';'))
            type = sub_mode_dict['t'] if 't' in sub_mode_dict else 'l' # 'l' or 'b' or 'g'
            depth = int(sub_mode_dict['d']) if 'd' in sub_mode_dict else 2
            units = int(sub_mode_dict['u']) if 'u' in sub_mode_dict else 64 
            return r(inputs, type, depth, units, self.num_classes, has_dense)

