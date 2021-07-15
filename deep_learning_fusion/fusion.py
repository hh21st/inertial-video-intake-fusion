"""Fusion model for inertial data"""

import numpy as np
import tensorflow as tf
import cnn
import rnn

def get_cnn_model(params, inputs, var_scope_suffix, is_training, padding_size = 0):
    cnn_model = cnn.Model(params) 
    return cnn_model(inputs, '_inputs1', is_training, padding_size)

def get_rnn_model(params, inputs, var_scope_suffix, is_training, has_dense = True):
    if params.model == 'cnn_rnn':
        rnn_model = rnn.Model(params)
    else:
        raise RuntimeError('model {0} is not applicable'.format(params.model))
    return rnn_model(inputs, var_scope_suffix, is_training, has_dense)

def fuse_late(params, inputs1, inputs2, is_training):
    seq_pool1, padding_size1, inputs1 = get_cnn_model(params, inputs1, '_inputs1', is_training)
    seq_pool2, padding_size2, inputs2 = get_cnn_model(params, inputs2, '_inputs2', is_training)
    assert seq_pool1 == seq_pool2, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs 2 is {}'.format(seq_pool1, seq_pool2)
    inputs1 = get_rnn_model(params, inputs1, '_inputs1', is_training)
    inputs1 = tf.nn.softmax(inputs1, name='softmax_tensor')
    inputs2 = get_rnn_model(params, inputs2, '_inputs2', is_training)
    inputs2 = tf.nn.softmax(inputs2, name='softmax_tensor')
    return seq_pool1, padding_size1, inputs1, inputs2

def fuse_late_fork(params, inputs1, inputs2, inputs3, inputs4, is_training):
    seq_pool1, padding_size1, inputs1 = get_cnn_model(params, inputs1, '_inputs1', is_training)
    seq_pool2, padding_size2, inputs2 = get_cnn_model(params, inputs2, '_inputs2', is_training)
    seq_pool3, padding_size3, inputs3 = get_cnn_model(params, inputs3, '_inputs3', is_training)
    seq_pool4, padding_size4, inputs4 = get_cnn_model(params, inputs4, '_inputs4', is_training)
    assert seq_pool1 == seq_pool2, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs 2 is {}'.format(seq_pool1, seq_pool2)
    assert seq_pool3 == seq_pool4, 'seq_pool for inputs 3 is {} whereas seq_pool for inputs 4 is {}'.format(seq_pool3, seq_pool4)
    assert seq_pool1 == seq_pool3, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs 3 is {}'.format(seq_pool1, seq_pool3)
    inputs1 = get_rnn_model(params, inputs1, '_inputs1', is_training)
    inputs1 = tf.nn.softmax(inputs1, name='softmax_tensor')
    inputs2 = get_rnn_model(params, inputs2, '_inputs2', is_training)
    inputs2 = tf.nn.softmax(inputs2, name='softmax_tensor')
    inputs3 = get_rnn_model(params, inputs3, '_inputs3', is_training)
    inputs3 = tf.nn.softmax(inputs3, name='softmax_tensor')
    inputs4 = get_rnn_model(params, inputs4, '_inputs4', is_training)
    inputs4 = tf.nn.softmax(inputs4, name='softmax_tensor')
    return seq_pool1, padding_size1, inputs1, inputs2, inputs4, inputs4

def fuse_early(params, inputs1, inputs2, is_training):
    seq_pool1, padding_size1, inputs1 = get_cnn_model(params, inputs1, '_inputs1', is_training)
    seq_pool2, padding_size2, inputs2 = get_cnn_model(params, inputs2, '_inputs2', is_training)
    assert seq_pool1 == seq_pool2, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs 2 is {}'.format(seq_pool1, seq_pool2)
    inputs = tf.concat([inputs1, inputs2], axis = 2)
    inputs = get_rnn_model(params , inputs, '_inputs', is_training)
    return seq_pool1, padding_size1, inputs

def fuse_early_merge_cnn(params, inputs1, inputs2, is_training):
    seq_pool1, padding_size1, inputs1 = get_cnn_model(params, inputs1, '_inputs1', is_training)
    seq_pool2, padding_size2, inputs2 = get_cnn_model(params, inputs2, '_inputs2', is_training)
    assert seq_pool1 == seq_pool2, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs 2 is {}'.format(seq_pool1, seq_pool2)
    inputs = tf.concat([inputs1, inputs2], axis = 2)
    seq_pool, padding_size, inputs = get_cnn_model(params, inputs, '_inputs', is_training, padding_size)
    assert seq_pool1 == seq_pool, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs (merge) is {}'.format(seq_pool1, seq_pool)
    inputs = get_rnn_model(params ,inputs, '_inputs', is_training)
    return seq_pool1, padding_size1, inputs

def fuse_early_merge_rnn(params, inputs1, inputs2, is_training):
    seq_pool1, padding_size1, inputs1 = get_cnn_model(params, inputs1, '_inputs1', is_training)
    seq_pool2, padding_size2, inputs2 = get_cnn_model(params, inputs2, '_inputs2', is_training)
    assert seq_pool1 == seq_pool2, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs 2 is {}'.format(seq_pool1, seq_pool2)
    inputs1 = get_rnn_model(params, inputs1, '_inputs1', is_training, False)
    inputs2 = get_rnn_model(params, inputs2, '_inputs2', is_training, False)
    inputs = tf.concat([inputs1, inputs2], axis = 2)
    inputs = get_rnn_model(params ,inputs, '_inputs', is_training)
    return seq_pool1, padding_size1, inputs

def fuse_early_fork(params, inputs1, inputs2, inputs3, inputs4, is_training):
    seq_pool1, padding_size1, inputs1 = get_cnn_model(params, inputs1, '_inputs1', is_training)
    seq_pool2, padding_size2, inputs2 = get_cnn_model(params, inputs2, '_inputs2', is_training)
    seq_pool3, padding_size3, inputs3 = get_cnn_model(params, inputs3, '_inputs3', is_training)
    seq_pool4, padding_size4, inputs4 = get_cnn_model(params, inputs4, '_inputs4', is_training)
    assert seq_pool1 == seq_pool2, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs 2 is {}'.format(seq_pool1, seq_pool2)
    assert seq_pool3 == seq_pool4, 'seq_pool for inputs 3 is {} whereas seq_pool for inputs 4 is {}'.format(seq_pool3, seq_pool4)
    assert seq_pool1 == seq_pool3, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs 3 is {}'.format(seq_pool1, seq_pool3)
    inputs = tf.concat([inputs1, inputs2, inputs3, inputs4], axis = 2)
    inputs = get_rnn_model(params ,inputs, '_inputs', is_training)
    return seq_pool1, padding_size1, inputs

def fuse_early_fork_merge_cnn(params, inputs1, inputs2, inputs3, inputs4, is_training):
    seq_pool1, padding_size1, inputs1 = get_cnn_model(params, inputs1, '_inputs1', is_training)
    seq_pool2, padding_size2, inputs2 = get_cnn_model(params, inputs2, '_inputs2', is_training)
    seq_pool3, padding_size3, inputs3 = get_cnn_model(params, inputs3, '_inputs3', is_training)
    seq_pool4, padding_size4, inputs4 = get_cnn_model(params, inputs4, '_inputs4', is_training)
    assert seq_pool1 == seq_pool2, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs 2 is {}'.format(seq_pool1, seq_pool2)
    assert seq_pool3 == seq_pool4, 'seq_pool for inputs 3 is {} whereas seq_pool for inputs 4 is {}'.format(seq_pool3, seq_pool4)
    assert seq_pool1 == seq_pool3, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs 3 is {}'.format(seq_pool1, seq_pool3)
    inputs = tf.concat([inputs1, inputs2, inputs3, inputs4], axis = 2)
    seq_pool, padding_size, inputs = get_cnn_model(params, inputs, '_inputs', is_training, padding_size)
    assert seq_pool1 == seq_pool, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs (merge) is {}'.format(seq_pool1, seq_pool)
    inputs = get_rnn_model(params ,inputs, '_inputs', is_training)
    return seq_pool1, padding_size1, inputs

def fuse_early_fork_merge_rnn(params, inputs1, inputs2, inputs3, inputs4, is_training):
    seq_pool1, padding_size1, inputs1 = get_cnn_model(params, inputs1, '_inputs1', is_training)
    seq_pool2, padding_size2, inputs2 = get_cnn_model(params, inputs2, '_inputs2', is_training)
    seq_pool3, padding_size3, inputs3 = get_cnn_model(params, inputs3, '_inputs3', is_training)
    seq_pool4, padding_size4, inputs4 = get_cnn_model(params, inputs4, '_inputs4', is_training)
    assert seq_pool1 == seq_pool2, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs 2 is {}'.format(seq_pool1, seq_pool2)
    assert seq_pool3 == seq_pool4, 'seq_pool for inputs 3 is {} whereas seq_pool for inputs 4 is {}'.format(seq_pool3, seq_pool4)
    assert seq_pool1 == seq_pool3, 'seq_pool for inputs 1 is {} whereas seq_pool for inputs 3 is {}'.format(seq_pool1, seq_pool3)
    inputs1 = get_rnn_model(params, inputs1, '_inputs1', is_training, False)
    inputs2 = get_rnn_model(params, inputs2, '_inputs2', is_training, False)
    inputs3 = get_rnn_model(params, inputs3, '_inputs3', is_training, False)
    inputs4 = get_rnn_model(params, inputs4, '_inputs4', is_training, False)
    inputs = tf.concat([inputs1, inputs2, inputs3, inputs4], axis = 2)
    inputs = get_rnn_model(params ,inputs, '_inputs', is_training)
    return seq_pool1, padding_size1, inputs

def fuse_earliest(params, inputs, is_training):
    seq_pool, padding_size, inputs = get_cnn_model(params, inputs, '_inputs1', is_training)
    inputs = get_rnn_model(params ,inputs, '_inputs', is_training)
    return seq_pool, padding_size, inputs

######################################################################################################################

def dnd(params, inputs, is_training):
    
    inputs_dom = inputs[:,:,0:6]
    inputs_ndom = inputs[:,:,6:12]

    if params.f_strategy == 'late':
        return fuse_late(params, inputs_dom, inputs_ndom, is_training)
    elif params.f_strategy == 'early':
        return fuse_early(params, inputs_dom, inputs_ndom, is_training)
    elif params.f_strategy == 'early_merge_cnn':
        return fuse_early_merge_cnn(params, inputs_dom, inputs_ndom, is_training)
    elif params.f_strategy == 'early_merge_rnn':
        return fuse_early_merge_rnn(params, inputs_dom, inputs_ndom, is_training)
    else:
        raise RuntimeError('{0} is not supported in fusion.py'.format(params.f_strategy))

def dnd1(params, inputs, is_training):
    seq_pool, padding_size, inputs_left, inputs_right = dnd(params, inputs, is_training)
    inputs = inputs_left + inputs_right
    return seq_pool, padding_size, inputs

def ag(params, inputs, is_training):
    
    inputs_accel_dom = inputs[:,:,0:3] # 0,1,2
    inputs_gyro_dom = inputs[:,:,3:6] # 3,4,5
    inputs_accel_ndom = inputs[:,:,6:9] # 6,7,8 
    inputs_gyro_ndom = inputs[:,:,9:12] # 9,10,11

    inputs_accel = tf.concat([inputs_accel_dom, inputs_accel_ndom], axis = 2)
    inputs_gyro = tf.concat([inputs_gyro_dom, inputs_gyro_ndom], axis = 2)
    if params.f_strategy == 'late':
        return fuse_late(params, inputs_accel, inputs_gyro, is_training)
    elif params.f_strategy == 'early':
        return fuse_early(params, inputs_accel, inputs_gyro, is_training)
    elif params.f_strategy == 'early_merge_cnn':
        return fuse_early_merge_cnn(params, inputs_accel, inputs_gyro, is_training)
    elif params.f_strategy == 'early_merge_rnn':
        return fuse_early_merge_rnn(params, inputs_accel, inputs_gyro, is_training)
    else:
        raise RuntimeError('{0} is not supported in fusion.py'.format(params.f_strategy))

def ag1(params, inputs, is_training):
    if params.f_strategy == 'late':
        seq_pool, padding_size, inputs_accel, inputs_gyro = ag(params, inputs, is_training)
        return seq_pool, padding_size, inputs_accel + inputs_gyro
    else:
        return ag(params, inputs, is_training)

def ag2(params, inputs, is_training):
    if params.f_strategy == 'late':
        seq_pool, padding_size, inputs_accel, inputs_gyro = ag(params, inputs, is_training)
        return seq_pool, padding_size, inputs_accel + inputs_gyro + inputs_gyro
    else:
        raise RuntimeError('{0} is not supported in {1} mode of fusion.py'.format(params.f_strategy,params.f_mode))

def agdnd(params, inputs, is_training):
    inputs_accel_dom = inputs[:,:,0:3] # 0,1,2
    inputs_gyro_dom = inputs[:,:,3:6] # 3,4,5
    inputs_accel_ndom = inputs[:,:,6:9] # 6,7,8 
    inputs_gyro_ndom = inputs[:,:,9:12] # 9,10,11
    if params.f_strategy == 'late':
        return fuse_late_fork(params, inputs_accel_dom, inputs_accel_ndom, inputs_gyro_dom, inputs_gyro_ndom, is_training)
    elif params.f_strategy == 'early':
        return fuse_early_fork(params, inputs_accel_dom, inputs_accel_ndom, inputs_gyro_dom, inputs_gyro_ndom, is_training)
    elif params.f_strategy == 'early_merge_cnn':
        return fuse_early_fork_merge_cnn(params, inputs_accel_dom, inputs_accel_ndom, inputs_gyro_dom, inputs_gyro_ndom, is_training)
    elif params.f_strategy == 'early_merge_rnn':
        return fuse_early_fork_merge_rnn(params, inputs_accel_dom, inputs_accel_ndom, inputs_gyro_dom, inputs_gyro_ndom, is_training)
    else:
        raise RuntimeError('Strategy {0} is not supported in fusion.py'.format(params.f_strategy))

def agdnd1(params, inputs, is_training):
    if params.f_strategy == 'late':
        seq_pool, padding_size, inputs_accel_dom, inputs_accel_ndom, inputs_gyro_dom, inputs_gyro_ndom = agdnd(params, inputs, is_training)
        return seq_pool, padding_size, inputs_accel_dom + inputs_accel_ndom + inputs_gyro_dom + inputs_gyro_ndom
    else:
        return agdnd(params, inputs, is_training)

def ad3and1gd4gnd2(params, inputs, is_training):
    if params.f_strategy == 'late':
        seq_pool, padding_size, inputs_accel_dom, inputs_accel_ndom, inputs_gyro_dom, inputs_gyro_ndom = agdnd(params, inputs, is_training)
        return seq_pool, (inputs_accel_dom*3) + inputs_accel_ndom + (inputs_gyro_dom*4) + (inputs_gyro_ndom*2)
    else:
        raise RuntimeError('{0} is not supported in {1} mode of fusion.py'.format(params.f_strategy,params.f_mode))
           
class Model(object):
    """Base class for Modality CNN BLSTM model."""

    def __init__(self, params):
        self.params = params
    def __call__(self, inputs, is_training):
        if self.params.fusion == 'earliest':
            return fuse_earliest(self.params, inputs, is_training)
        elif self.params.fusion == 'accel_gyro':
            if self.params.f_mode == 'ag1':
                return ag1(self.params, inputs, is_training)
            if self.params.f_mode == 'ag2':
                return ag2(self.params, inputs, is_training)
        elif self.params.fusion == 'dom_ndom':
            if self.params.f_mode == 'dnd1': 
                return dnd1(self.params, inputs, is_training)
        elif self.params.fusion == 'accel_gyro_dom_ndom':
            if self.params.f_mode == 'agdnd1': 
                return agdnd1(self.params, inputs, is_training)
            elif self.params.f_mode == 'ad3and1gd4gnd2': 
                return ad3and1gd4gnd2(self.params, inputs, is_training)
        else:
            raise RuntimeError('f mode {0} is not implemented'.format(self.params.f_mode))
