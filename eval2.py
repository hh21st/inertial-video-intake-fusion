"""Evaluate exported frame-level probabilities."""

from __future__ import division
import argparse
import csv
import glob
import numpy as np
import os
import tensorflow as tf
import absl
import utils

FLAGS = absl.app.flags.FLAGS
if 'prob_dir' not in FLAGS.__flags.keys():
    absl.app.flags.DEFINE_string(
        name='prob_dir', default='', help='Directory for probability data.')
if 'min_dist' not in FLAGS.__flags.keys():
    absl.app.flags.DEFINE_integer(
        name='min_dist', default=128, help='Minimum frames between detections.')
if 'threshold' not in FLAGS.__flags.keys():
    absl.app.flags.DEFINE_float(
        name='threshold', default=0.9, help='Detection threshold probability')
if 'eval_mode' not in FLAGS.__flags.keys():
    absl.app.flags.DEFINE_enum(
        name='eval_mode', default='predict', enum_values=['estimate', 'predict'], 
        help='estimation using validation set or predicting using test set')
if 'min_threshold' not in FLAGS.__flags.keys():
    absl.app.flags.DEFINE_float(
        name='min_threshold', default=0.5, help='Minimum detection threshold probability')
if 'max_threshold' not in FLAGS.__flags.keys():
    absl.app.flags.DEFINE_float(
        name='max_threshold', default=1, help='Maximum detection threshold probability')
if 'inc_threshold' not in FLAGS.__flags.keys():
    absl.app.flags.DEFINE_float(
        name='inc_threshold', default=0.001, help='Increment for detection threshold search')
if 'col_label' not in FLAGS.__flags.keys():
    absl.app.flags.DEFINE_integer(
        name='col_label', default=1, help='Col number of label in csv')
if 'col_prob' not in FLAGS.__flags.keys():
    absl.app.flags.DEFINE_integer(
        name='col_prob', default=2, help='Col number of probability in csv')

absl.app.flags.DEFINE_string(
    name='src_dir', default=r'\\uncle.newcastle.edu.au\entities\research\oreba\OREBA\Phase 1\Synchronised', help='Directory to search for data.')
absl.app.flags.DEFINE_string(
    name='dom_hand_info_file_name', default='dominant_hand.csv', help='the name of the file that contains the dominant hand info')


CSV_SUFFIX = '*.csv'

def read_dominant_from_file(src_dir, subject_id, dom_hand_info_file_name):
    file_full_name = os.path.join(src_dir, dom_hand_info_file_name)
    dom_hand_info = csv.reader(open(file_full_name, 'r'), delimiter=',')
    next(dom_hand_info, None)
    for row in dom_hand_info:
        if subject_id == row[0]:
            return row[1].strip().lower()
    return 'not found'

def read_dominant(subject_id):
    if len(subject_id) == 4:
        subject_id += '_1'
    return read_dominant_from_file(FLAGS.src_dir, subject_id, FLAGS.dom_hand_info_file_name)

def import_probs_and_labels(filepath, col_label, col_prob):
    """Import probabilities and labels from csv"""
    filenames = glob.glob(os.path.join(filepath, CSV_SUFFIX))
    assert filenames, "No prob files were found"
    labels = []
    probs = []
    labels_2 = []
    labels_3 = []
    labels_4 = []
    frame_ids = []
    p_ids = []
    for filename in filenames:
        with open(filename) as dest_f:
            for row in csv.reader(dest_f, delimiter=','):
                labels.append(int(row[col_label]))
                probs.append(float(row[col_prob]))
                labels_2.append(row[3])
                labels_3.append(row[4])
                labels_4.append(row[5])
                frame_ids.append(row[0])
                p_ids.append(row[6])
    labels = np.array(labels)
    probs = np.array(probs)

    return probs, labels, labels_2, labels_3, labels_4, frame_ids, p_ids

def max_search(probs, threshold, mindist):
    """Perform a max search"""
    # Threshold probs
    probabilities = np.copy(probs)
    probabilities[probabilities <= threshold] = 0
    # Potential detections
    idx_p = np.where(probabilities > 0)[0]
    if (idx_p.size == 0):
        return np.zeros(probs.shape)
    # Identify start and end of detections
    p_d = np.diff(idx_p) - 1
    p = np.where(p_d > 0)[0]
    p_start = np.concatenate(([0], p+1))
    p_end = np.concatenate((p, [idx_p.shape[0]-1]))
    # Infer start and end indices of detections
    idx_start = idx_p[p_start]
    idx_end = idx_p[p_end]
    idx_max = [start+np.argmax(probabilities[start:end+1])
        for start, end in zip(idx_start, idx_end)]
    # Remove detections within mindist
    max_diff = np.diff(idx_max)
    carry = 0; rem_i = []
    for i, diff in enumerate(np.concatenate(([mindist], max_diff))):
        if (diff + carry < mindist):
            rem_i.append(i)
            carry += diff
        else:
            carry = 0
    rem_i = np.array(rem_i)
    idx_max_mindist = np.delete(idx_max, rem_i)
    # Return detections
    detections = np.zeros(probabilities.shape, dtype=np.int32)
    detections[idx_max_mindist] = 1
    return detections

def eval_stage_1(probs, labels):
    """Stage 1 evaluation based on frame-level probabilitites"""
    frame_tp_1 = np.intersect1d(np.where(probs >= 0.5), np.where(labels == 1)).shape[0]
    frame_fn_1 = np.intersect1d(np.where(probs < 0.5), np.where(labels == 1)).shape[0]
    frame_rec_1 = frame_tp_1 / (frame_tp_1 + frame_fn_1)
    frame_tp_0 = np.intersect1d(np.where(probs < 0.5), np.where(labels == 0)).shape[0]
    frame_fn_0 = np.intersect1d(np.where(probs >= 0.5), np.where(labels == 0)).shape[0]
    frame_rec_0 = frame_tp_0 / (frame_tp_0 + frame_fn_0)
    uar = (frame_rec_1 + frame_rec_0) / 2.0
    return uar

def eval_stage_2(dets, labels, labels_2, labels_3, labels_4, frame_ids, p_ids):
    """Stage 2 evaluation based on gesture-level metric proposed by Kyritsis et al. (2019)"""
    def _split_idx(labels):
        idx_t = np.where(labels == 1)[0]
        t_d = np.diff(idx_t) - 1
        t = np.where(t_d > 0)[0]
        t_start = np.concatenate(([0], t+1))
        t_end = np.concatenate((t, [idx_t.shape[0]-1]))
        idx_start = idx_t[t_start]
        idx_end = idx_t[t_end]
        return [np.arange(start, end+1) for start, end in zip(idx_start, idx_end)]
    idxs_t = _split_idx(labels)
    idxs_f = np.where(labels == 0)
    splits_t = [dets[split_idx] for split_idx in idxs_t]
    splits_f = dets[idxs_f]
    tp = np.sum([1 if np.sum(split) > 0 else 0 for split in splits_t])
    fn = np.sum([0 if np.sum(split) > 0 else 1 for split in splits_t])
    fp_1 = np.sum([np.sum(split)-1 if np.sum(split)>1 else 0 for split in splits_t])
    fp_2 = np.sum(splits_f)

    labels_2 = np.asarray(labels_2)
    labels_3 = np.asarray(labels_3)
    labels_4 = np.asarray(labels_4)
    frame_ids = np.asarray(frame_ids)
    p_ids = np.asarray(p_ids)
    
    idxs_fp_1__2 = [split_idx for split_idx in idxs_t if np.sum(dets[split_idx])==2]
    idxs_fp_1__3 = [split_idx for split_idx in idxs_t if np.sum(dets[split_idx])==3]
    idxs_fp_1__4 = [split_idx for split_idx in idxs_t if np.sum(dets[split_idx])==4]
    idxs_fp_1__5 = [split_idx for split_idx in idxs_t if np.sum(dets[split_idx])==5]
    idxs_fp_1__6 = [split_idx for split_idx in idxs_t if np.sum(dets[split_idx])==6]
    assert len([split_idx for split_idx in idxs_t if np.sum(dets[split_idx])>6])==0, '[split_idx for split_idx in idxs_t if np.sum(dets[split_idx])>6],{}'.format([split_idx for split_idx in idxs_t if np.sum(dets[split_idx])>6])
    len_idxs_fp_1 = len(idxs_fp_1__2)
    len_idxs_fp_1 += len(idxs_fp_1__3) * 2
    len_idxs_fp_1 += len(idxs_fp_1__4) * 3
    len_idxs_fp_1 += len(idxs_fp_1__5) * 4
    len_idxs_fp_1 += len(idxs_fp_1__6) * 5
    idxs_fp_2 = [idx_f for idx_f in idxs_f[0] if dets[idx_f] == 1]

    assert fp_1 == len_idxs_fp_1, 'fp_1 == len_idxs_fp_1,{0},{1}'.format(fp_1, len_idxs_fp_1)
    assert fp_2 == len(idxs_fp_2), 'fp_2 == len(idxs_fp_2),{0},{1}'.format(fp_2, len(idxs_fp_2))
    
####
    idxs_t_short = [split_idx for split_idx in idxs_t if len(split_idx)<32]
    
    assert len(idxs_t_short)==0
    
    idxs_tp = [split_idx for split_idx in idxs_t if np.sum(dets[split_idx])>0]
    
    idxs_tp_labels2_Eat = [idx_tp for idx_tp in idxs_tp if labels_2[idx_tp][0]=='Intake-Eat']
    idxs_tp_labels2_Drink = [idx_tp for idx_tp in idxs_tp if labels_2[idx_tp][0]=='Intake-Drink']
    idxs_tp_labels2_Lick = [idx_tp for idx_tp in idxs_tp if labels_2[idx_tp][0]=='Intake-Lick']
    idxs_tp_labels2_Idle = [idx_tp for idx_tp in idxs_tp if labels_2[idx_tp][0]=='Idle']
    
    idxs_tp_labels2_Eat_len = len(idxs_tp_labels2_Eat)
    idxs_tp_labels2_Drink_len = len(idxs_tp_labels2_Drink)
    idxs_tp_labels2_Lick_len = len(idxs_tp_labels2_Lick)
    idxs_tp_labels2_Idle_len = len(idxs_tp_labels2_Idle)
    assert idxs_tp_labels2_Idle_len == 0, 'idxs_tp_labels2_Idle_len'
    assert idxs_tp_labels2_Eat_len + idxs_tp_labels2_Drink_len + idxs_tp_labels2_Lick_len == len(idxs_tp), 'idxs_tp_labels2_Eat_len + idxs_tp_labels2_Drink_len + idxs_tp_labels2_Lick_len,{0},{1}'.format(idxs_tp_labels2_Eat_len + idxs_tp_labels2_Drink_len + idxs_tp_labels2_Lick_len,len(idxs_tp))

    idxs_tp_labels3_Right = [idx_tp for idx_tp in idxs_tp if labels_3[idx_tp][0]=='Right']
    idxs_tp_labels3_Left = [idx_tp for idx_tp in idxs_tp if labels_3[idx_tp][0]=='Left']
    idxs_tp_labels3_Both = [idx_tp for idx_tp in idxs_tp if labels_3[idx_tp][0]=='Both']
    idxs_tp_labels3_Idle = [idx_tp for idx_tp in idxs_tp if labels_3[idx_tp][0]=='Idle']

    idxs_tp_labels3_Right_len = len(idxs_tp_labels3_Right)
    idxs_tp_labels3_Left_len = len(idxs_tp_labels3_Left)
    idxs_tp_labels3_Both_len = len(idxs_tp_labels3_Both)
    idxs_tp_labels3_Idle_len = len(idxs_tp_labels3_Idle)
    assert idxs_tp_labels3_Idle_len == 0, 'idxs_tp_labels3_Idle_len'
    assert idxs_tp_labels3_Right_len + idxs_tp_labels3_Left_len + idxs_tp_labels3_Both_len == len(idxs_tp), 'idxs_tp_labels3_Right_len + idxs_tp_labels3_Left_len + idxs_tp_labels3_Both_len,{0},{1}'.format(idxs_tp_labels3_Right_len + idxs_tp_labels3_Left_len + idxs_tp_labels3_Both_len,len(idxs_tp))

    idxs_tp_labels3_dominant = [idx_tp for idx_tp in idxs_tp if (labels_3[idx_tp][0]=='Right' and read_dominant(p_ids[idx_tp][0])=='right') or (labels_3[idx_tp][0]=='Left' and read_dominant(p_ids[idx_tp][0])=='left')]
    idxs_tp_labels3_nondominant = [idx_tp for idx_tp in idxs_tp if (labels_3[idx_tp][0]=='Right' and read_dominant(p_ids[idx_tp][0])=='left') or (labels_3[idx_tp][0]=='Left' and read_dominant(p_ids[idx_tp][0])=='right')]

    idxs_tp_labels3_dominant_len = len(idxs_tp_labels3_dominant)
    idxs_tp_labels3_nondominant_len = len(idxs_tp_labels3_nondominant)

    assert idxs_tp_labels3_dominant_len + idxs_tp_labels3_nondominant_len == idxs_tp_labels3_Right_len + idxs_tp_labels3_Left_len , 'dominant and nondominantlen {0} not the same as right and left {1}'.format(idxs_tp_labels3_dominant_len + idxs_tp_labels3_nondominant_len, idxs_tp_labels3_Right_len + idxs_tp_labels3_Left_len)
    
    idxs_tp_labels4_Spoon = [idx_tp for idx_tp in idxs_tp if labels_4[idx_tp][0]=='Spoon']
    idxs_tp_labels4_Fork = [idx_tp for idx_tp in idxs_tp if labels_4[idx_tp][0]=='Fork']
    idxs_tp_labels4_Cup = [idx_tp for idx_tp in idxs_tp if labels_4[idx_tp][0]=='Cup']
    idxs_tp_labels4_Hand = [idx_tp for idx_tp in idxs_tp if labels_4[idx_tp][0]=='Hand']
    idxs_tp_labels4_Knife = [idx_tp for idx_tp in idxs_tp if labels_4[idx_tp][0]=='Knife']
    idxs_tp_labels4_Finger = [idx_tp for idx_tp in idxs_tp if labels_4[idx_tp][0]=='Finger']
    idxs_tp_labels4_Idle = [idx_tp for idx_tp in idxs_tp if labels_4[idx_tp][0]=='Idle']

    idxs_tp_labels4_Spoon_len = len(idxs_tp_labels4_Spoon)
    idxs_tp_labels4_Fork_len = len(idxs_tp_labels4_Fork)
    idxs_tp_labels4_Cup_len = len(idxs_tp_labels4_Cup)
    idxs_tp_labels4_Hand_len = len(idxs_tp_labels4_Hand)
    idxs_tp_labels4_Knife_len = len(idxs_tp_labels4_Knife)
    idxs_tp_labels4_Finger_len = len(idxs_tp_labels4_Finger)
    idxs_tp_labels4_Idle_len = len(idxs_tp_labels4_Idle)
    assert idxs_tp_labels4_Idle_len == 0, 'idxs_tp_labels4_Idle_len'
    assert idxs_tp_labels4_Spoon_len + idxs_tp_labels4_Fork_len + idxs_tp_labels4_Cup_len + idxs_tp_labels4_Hand_len + idxs_tp_labels4_Knife_len + idxs_tp_labels4_Finger_len == len(idxs_tp), 'idxs_tp_labels4_Spoon_len + idxs_tp_labels4_Fork_len + idxs_tp_labels4_Cup_len + idxs_tp_labels4_Hand_len + idxs_tp_labels4_Knife_len + idxs_tp_labels4_Finger_len,{0},{1}'.format(idxs_tp_labels4_Spoon_len + idxs_tp_labels4_Fork_len + idxs_tp_labels4_Cup_len + idxs_tp_labels4_Hand_len + idxs_tp_labels4_Finger_len,len(idxs_tp))
####

####
    idxs_fn = [split_idx for split_idx in idxs_t if np.sum(dets[split_idx])<=0]
    
    idxs_fn_labels2_Eat = [idx_fn for idx_fn in idxs_fn if labels_2[idx_fn][0]=='Intake-Eat']
    idxs_fn_labels2_Drink = [idx_fn for idx_fn in idxs_fn if labels_2[idx_fn][0]=='Intake-Drink']
    idxs_fn_labels2_Lick = [idx_fn for idx_fn in idxs_fn if labels_2[idx_fn][0]=='Intake-Lick']
    idxs_fn_labels2_Idle = [idx_fn for idx_fn in idxs_fn if labels_2[idx_fn][0]=='Idle']
    
    idxs_fn_labels2_Eat_len = len(idxs_fn_labels2_Eat)
    idxs_fn_labels2_Drink_len = len(idxs_fn_labels2_Drink)
    idxs_fn_labels2_Lick_len = len(idxs_fn_labels2_Lick)
    idxs_fn_labels2_Idle_len = len(idxs_fn_labels2_Idle)
    assert idxs_fn_labels2_Idle_len == 0, 'idxs_fn_labels2_Idle_len'
    assert idxs_fn_labels2_Eat_len + idxs_fn_labels2_Drink_len + idxs_fn_labels2_Lick_len == len(idxs_fn), 'idxs_fn_labels2_Eat_len + idxs_fn_labels2_Drink_len + idxs_fn_labels2_Lick_len,{0},{1}'.format(idxs_fn_labels2_Eat_len + idxs_fn_labels2_Drink_len + idxs_fn_labels2_Lick_len,len(idxs_fn))

    idxs_fn_labels3_Right = [idx_fn for idx_fn in idxs_fn if labels_3[idx_fn][0]=='Right']
    idxs_fn_labels3_Left = [idx_fn for idx_fn in idxs_fn if labels_3[idx_fn][0]=='Left']
    idxs_fn_labels3_Both = [idx_fn for idx_fn in idxs_fn if labels_3[idx_fn][0]=='Both']
    idxs_fn_labels3_Idle = [idx_fn for idx_fn in idxs_fn if labels_3[idx_fn][0]=='Idle']

    idxs_fn_labels3_Right_len = len(idxs_fn_labels3_Right)
    idxs_fn_labels3_Left_len = len(idxs_fn_labels3_Left)
    idxs_fn_labels3_Both_len = len(idxs_fn_labels3_Both)
    idxs_fn_labels3_Idle_len = len(idxs_fn_labels3_Idle)
    assert idxs_fn_labels3_Idle_len == 0, 'idxs_fn_labels3_Idle_len'
    assert idxs_fn_labels3_Right_len + idxs_fn_labels3_Left_len + idxs_fn_labels3_Both_len == len(idxs_fn), 'idxs_fn_labels3_Right_len + idxs_fn_labels3_Left_len + idxs_fn_labels3_Both_len,{0},{1}'.format(idxs_fn_labels3_Right_len + idxs_fn_labels3_Left_len + idxs_fn_labels3_Both_len,len(idxs_fn))

    idxs_fn_labels3_dominant = [idx_fn for idx_fn in idxs_fn if (labels_3[idx_fn][0]=='Right' and read_dominant(p_ids[idx_fn][0])=='right') or (labels_3[idx_fn][0]=='Left' and read_dominant(p_ids[idx_fn][0])=='left')]
    idxs_fn_labels3_nondominant = [idx_fn for idx_fn in idxs_fn if (labels_3[idx_fn][0]=='Right' and read_dominant(p_ids[idx_fn][0])=='left') or (labels_3[idx_fn][0]=='Left' and read_dominant(p_ids[idx_fn][0])=='right')]

    idxs_fn_labels3_dominant_len = len(idxs_fn_labels3_dominant)
    idxs_fn_labels3_nondominant_len = len(idxs_fn_labels3_nondominant)

    assert idxs_fn_labels3_dominant_len + idxs_fn_labels3_nondominant_len == idxs_fn_labels3_Right_len + idxs_fn_labels3_Left_len , 'dominant and nondominantlen {0} not the same as right and left {1}'.format(idxs_fn_labels3_dominant_len + idxs_fn_labels3_nondominant_len, idxs_fn_labels3_Right_len + idxs_fn_labels3_Left_len)
    
    idxs_fn_labels4_Spoon = [idx_fn for idx_fn in idxs_fn if labels_4[idx_fn][0]=='Spoon']
    idxs_fn_labels4_Fork = [idx_fn for idx_fn in idxs_fn if labels_4[idx_fn][0]=='Fork']
    idxs_fn_labels4_Cup = [idx_fn for idx_fn in idxs_fn if labels_4[idx_fn][0]=='Cup']
    idxs_fn_labels4_Hand = [idx_fn for idx_fn in idxs_fn if labels_4[idx_fn][0]=='Hand']
    idxs_fn_labels4_Knife = [idx_fn for idx_fn in idxs_fn if labels_4[idx_fn][0]=='Knife']
    idxs_fn_labels4_Finger = [idx_fn for idx_fn in idxs_fn if labels_4[idx_fn][0]=='Finger']
    idxs_fn_labels4_Idle = [idx_fn for idx_fn in idxs_fn if labels_4[idx_fn][0]=='Idle']

    idxs_fn_labels4_Spoon_len = len(idxs_fn_labels4_Spoon)
    idxs_fn_labels4_Fork_len = len(idxs_fn_labels4_Fork)
    idxs_fn_labels4_Cup_len = len(idxs_fn_labels4_Cup)
    idxs_fn_labels4_Hand_len = len(idxs_fn_labels4_Hand)
    idxs_fn_labels4_Knife_len = len(idxs_fn_labels4_Knife)
    idxs_fn_labels4_Finger_len = len(idxs_fn_labels4_Finger)
    idxs_fn_labels4_Idle_len = len(idxs_fn_labels4_Idle)
    assert idxs_fn_labels4_Idle_len == 0, 'idxs_fn_labels4_Idle_len'
    assert idxs_fn_labels4_Spoon_len + idxs_fn_labels4_Fork_len + idxs_fn_labels4_Cup_len + idxs_fn_labels4_Hand_len + idxs_fn_labels4_Knife_len + idxs_fn_labels4_Finger_len == len(idxs_fn), 'idxs_fn_labels4_Spoon_len + idxs_fn_labels4_Fork_len + idxs_fn_labels4_Cup_len + idxs_fn_labels4_Hand_len + idxs_fn_labels4_Knife_len + idxs_fn_labels4_Finger_len,{0},{1}'.format(idxs_fn_labels4_Spoon_len + idxs_fn_labels4_Fork_len + idxs_fn_labels4_Cup_len + idxs_fn_labels4_Hand_len + idxs_fn_labels4_Finger_len,len(idxs_fn))
####

    if tp > 0:
        prec = tp / (tp + fp_1 + fp_2)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
    else:
        prec = 0
        rec = 0
        f1 = 0
    return tp, fn, fp_1, fp_2, prec, rec, f1,len(idxs_tp),\
       idxs_tp_labels2_Eat_len, idxs_tp_labels2_Drink_len, idxs_tp_labels2_Lick_len,\
       idxs_tp_labels3_Right_len, idxs_tp_labels3_Left_len, idxs_tp_labels3_Both_len, idxs_tp_labels3_dominant_len, idxs_tp_labels3_nondominant_len,\
       idxs_tp_labels4_Spoon_len, idxs_tp_labels4_Fork_len, idxs_tp_labels4_Cup_len, idxs_tp_labels4_Hand_len, idxs_tp_labels4_Knife_len, idxs_tp_labels4_Finger_len,\
       idxs_fn_labels2_Eat_len, idxs_fn_labels2_Drink_len, idxs_fn_labels2_Lick_len,\
       idxs_fn_labels3_Right_len, idxs_fn_labels3_Left_len, idxs_fn_labels3_Both_len, idxs_fn_labels3_dominant_len, idxs_fn_labels3_nondominant_len,\
       idxs_fn_labels4_Spoon_len, idxs_fn_labels4_Fork_len, idxs_fn_labels4_Cup_len, idxs_fn_labels4_Hand_len, idxs_fn_labels4_Knife_len, idxs_fn_labels4_Finger_len

def write_logfile(result_file_name, uar, best_threshold, tp, fn, fp_1, fp_2, prec, rec, f1, len_idxs_tp,
        idxs_tp_labels2_Eat_len, idxs_tp_labels2_Drink_len, idxs_tp_labels2_Lick_len,
        idxs_tp_labels3_Right_len, idxs_tp_labels3_Left_len, idxs_tp_labels3_Both_len, idxs_tp_labels3_dominant_len, idxs_tp_labels3_nondominant_len,
        idxs_tp_labels4_Spoon_len, idxs_tp_labels4_Fork_len, idxs_tp_labels4_Cup_len, idxs_tp_labels4_Hand_len, idxs_tp_labels4_Knife_len, idxs_tp_labels4_Finger_len,
        idxs_fn_labels2_Eat_len, idxs_fn_labels2_Drink_len, idxs_fn_labels2_Lick_len,
        idxs_fn_labels3_Right_len, idxs_fn_labels3_Left_len, idxs_fn_labels3_Both_len, idxs_fn_labels3_dominant_len, idxs_fn_labels3_nondominant_len,
        idxs_fn_labels4_Spoon_len, idxs_fn_labels4_Fork_len, idxs_fn_labels4_Cup_len, idxs_fn_labels4_Hand_len, idxs_fn_labels4_Knife_len, idxs_fn_labels4_Finger_len):
    result_file = open(result_file_name, 'w')
    result_file.write('UAR: {}'.format(uar))
    result_file.write('\n')
    result_file.write('Best threshold: {}'.format(best_threshold))
    result_file.write('\n')
    result_file.write('F1: {}'.format(f1))
    result_file.write('\n')
    result_file.write('Precision: {}'.format(prec))
    result_file.write('\n')
    result_file.write('Recall: {}'.format(rec))
    result_file.write('\n')
    result_file.write('TP: {}'.format(tp))
    result_file.write('\n')
    result_file.write('FN: {}'.format(fn))
    result_file.write('\n')
    result_file.write('FP_1: {}'.format(fp_1))
    result_file.write('\n')
    result_file.write('FP_2: {}'.format(fp_2))
    result_file.write('\n')
    result_file.write('len_idxs_tp               :{}'.format(len_idxs_tp))
    result_file.write('\n')
    result_file.write('idxs_tp_labels2_Eat_len   :{}'.format(idxs_tp_labels2_Eat_len))
    result_file.write('\n')
    result_file.write('idxs_tp_labels2_Drink_len :{}'.format(idxs_tp_labels2_Drink_len))
    result_file.write('\n')
    result_file.write('idxs_tp_labels2_Lick_len  :{}'.format(idxs_tp_labels2_Lick_len))
    result_file.write('\n')
    result_file.write('idxs_tp_labels3_Right_len :{}'.format(idxs_tp_labels3_Right_len))
    result_file.write('\n')
    result_file.write('idxs_tp_labels3_Left_len  :{}'.format(idxs_tp_labels3_Left_len))
    result_file.write('\n')
    result_file.write('idxs_tp_labels3_Both_len  :{}'.format(idxs_tp_labels3_Both_len))
    result_file.write('\n')
    result_file.write('idxs_tp_labels3_dominant_len  :{}'.format(idxs_tp_labels3_dominant_len))
    result_file.write('\n')
    result_file.write('idxs_tp_labels3_nondominant_len  :{}'.format(idxs_tp_labels3_nondominant_len))
    result_file.write('\n')
    result_file.write('idxs_tp_labels4_Spoon_len :{}'.format(idxs_tp_labels4_Spoon_len))
    result_file.write('\n')
    result_file.write('idxs_tp_labels4_Fork_len  :{}'.format(idxs_tp_labels4_Fork_len))
    result_file.write('\n')
    result_file.write('idxs_tp_labels4_Cup_len   :{}'.format(idxs_tp_labels4_Cup_len))
    result_file.write('\n')
    result_file.write('idxs_tp_labels4_Hand_len  :{}'.format(idxs_tp_labels4_Hand_len))
    result_file.write('\n')
    result_file.write('idxs_tp_labels4_Knife_len :{}'.format(idxs_tp_labels4_Knife_len))
    result_file.write('\n')
    result_file.write('idxs_tp_labels4_Finger_len:{}'.format(idxs_tp_labels4_Finger_len))
    result_file.write('\n')
    result_file.write('idxs_fn_labels2_Eat_len   :{}'.format(idxs_fn_labels2_Eat_len))
    result_file.write('\n')
    result_file.write('idxs_fn_labels2_Drink_len :{}'.format(idxs_fn_labels2_Drink_len))
    result_file.write('\n')
    result_file.write('idxs_fn_labels2_Lick_len  :{}'.format(idxs_fn_labels2_Lick_len))
    result_file.write('\n')
    result_file.write('idxs_fn_labels3_Right_len :{}'.format(idxs_fn_labels3_Right_len))
    result_file.write('\n')
    result_file.write('idxs_fn_labels3_Left_len  :{}'.format(idxs_fn_labels3_Left_len))
    result_file.write('\n')
    result_file.write('idxs_fn_labels3_Both_len  :{}'.format(idxs_fn_labels3_Both_len))
    result_file.write('\n')
    result_file.write('idxs_fn_labels3_dominant_len  :{}'.format(idxs_fn_labels3_dominant_len))
    result_file.write('\n')
    result_file.write('idxs_fn_labels3_nondominant_len  :{}'.format(idxs_fn_labels3_nondominant_len))
    result_file.write('\n')
    result_file.write('idxs_fn_labels4_Spoon_len :{}'.format(idxs_fn_labels4_Spoon_len))
    result_file.write('\n')
    result_file.write('idxs_fn_labels4_Fork_len  :{}'.format(idxs_fn_labels4_Fork_len))
    result_file.write('\n')
    result_file.write('idxs_fn_labels4_Cup_len   :{}'.format(idxs_fn_labels4_Cup_len))
    result_file.write('\n')
    result_file.write('idxs_fn_labels4_Hand_len  :{}'.format(idxs_fn_labels4_Hand_len))
    result_file.write('\n')
    result_file.write('idxs_fn_labels4_Knife_len :{}'.format(idxs_fn_labels4_Knife_len))
    result_file.write('\n')
    result_file.write('idxs_fn_labels4_Finger_len:{}'.format(idxs_fn_labels4_Finger_len))
    result_file.write('\n')
    
    result_file.close()
    

def main(args=None):
    # Import the probs and labels from csv
    probs, labels, labels_2, labels_3, labels_4, frame_ids, p_ids = import_probs_and_labels(FLAGS.prob_dir, FLAGS.col_label, FLAGS.col_prob)
    # Calculate UAR for Stage I
    uar = eval_stage_1(probs, labels)
    parent_dir = utils.get_parent_dir(FLAGS.prob_dir)
    parent_dir_name = utils.get_current_dir_name(parent_dir)
    result_file_name_suffix = '_f1score.txt' if FLAGS.eval_mode == 'estimate' else '_f1score_test.txt' 
    result_file_name = os.path.join(parent_dir, parent_dir_name + result_file_name_suffix)
    if not FLAGS.overwrite and os.path.isfile(result_file_name):
        return -1,-1,-1,-1,-1,-1,-1,-1,'-1'
    # Perform grid search
    if FLAGS.eval_mode == 'estimate':
        # All evaluated threshold values
        threshold_vals = np.arange(FLAGS.min_threshold, FLAGS.max_threshold, FLAGS.inc_threshold)
        f1_results = []
        for threshold in threshold_vals:
            # Perform max search
            dets = max_search(probs, threshold, FLAGS.min_dist)
            # Calculate Stage II
            _, _, _, _, _, _, f1, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = eval_stage_2(dets, labels, labels_2, labels_3, labels_4, frame_ids, p_ids)
            f1_results.append(f1)
        # Find best threshold
        best_threshold = threshold_vals[np.argmax(f1_results)]
        final_dets = max_search(probs, best_threshold, FLAGS.min_dist)
        dets = final_dets
    else:
        best_threshold = FLAGS.threshold
        result_file_name_validation = os.path.join(parent_dir, parent_dir_name + '_f1score.txt')
        with open(result_file_name_validation) as result_file_val:
            for i, line in enumerate(result_file_val):
                if line.startswith('Best'):
                    best_threshold = float(line.split(':')[1].strip())
                    break
        # Perform max search
        dets = max_search(probs, best_threshold, FLAGS.min_dist)
        # Calculate Stage II

    tp, fn, fp_1, fp_2, prec, rec, f1, len_idxs_tp,\
    idxs_tp_labels2_Eat_len, idxs_tp_labels2_Drink_len, idxs_tp_labels2_Lick_len,\
    idxs_tp_labels3_Right_len, idxs_tp_labels3_Left_len, idxs_tp_labels3_Both_len, idxs_tp_labels3_dominant_len, idxs_tp_labels3_nondominant_len,\
    idxs_tp_labels4_Spoon_len, idxs_tp_labels4_Fork_len, idxs_tp_labels4_Cup_len, idxs_tp_labels4_Hand_len, idxs_tp_labels4_Knife_len, idxs_tp_labels4_Finger_len,\
    idxs_fn_labels2_Eat_len, idxs_fn_labels2_Drink_len, idxs_fn_labels2_Lick_len,\
    idxs_fn_labels3_Right_len, idxs_fn_labels3_Left_len, idxs_fn_labels3_Both_len, idxs_fn_labels3_dominant_len, idxs_fn_labels3_nondominant_len,\
    idxs_fn_labels4_Spoon_len, idxs_fn_labels4_Fork_len, idxs_fn_labels4_Cup_len, idxs_fn_labels4_Hand_len, idxs_fn_labels4_Knife_len, idxs_fn_labels4_Finger_len = eval_stage_2(dets, labels, labels_2, labels_3, labels_4, frame_ids, p_ids)
        
    write_logfile(result_file_name, uar, best_threshold, tp, fn, fp_1, fp_2, prec, rec, f1, len_idxs_tp,
    idxs_tp_labels2_Eat_len, idxs_tp_labels2_Drink_len, idxs_tp_labels2_Lick_len,
    idxs_tp_labels3_Right_len, idxs_tp_labels3_Left_len, idxs_tp_labels3_Both_len, idxs_tp_labels3_dominant_len, idxs_tp_labels3_nondominant_len,
    idxs_tp_labels4_Spoon_len, idxs_tp_labels4_Fork_len, idxs_tp_labels4_Cup_len, idxs_tp_labels4_Hand_len, idxs_tp_labels4_Knife_len, idxs_tp_labels4_Finger_len,
    idxs_fn_labels2_Eat_len, idxs_fn_labels2_Drink_len, idxs_fn_labels2_Lick_len,
    idxs_fn_labels3_Right_len, idxs_fn_labels3_Left_len, idxs_fn_labels3_Both_len, idxs_fn_labels3_dominant_len, idxs_fn_labels3_nondominant_len,
    idxs_fn_labels4_Spoon_len, idxs_fn_labels4_Fork_len, idxs_fn_labels4_Cup_len, idxs_fn_labels4_Hand_len, idxs_fn_labels4_Knife_len, idxs_fn_labels4_Finger_len)

    return uar, tp, fn, fp_1, fp_2, prec, rec, f1, best_threshold
# Run
if __name__ == '__main__':
    absl.app.run(main=main)
