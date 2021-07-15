"""Add other labels to prob files."""

import argparse
import csv
import glob
import numpy as np
import os
#import tensorflow as tf
import operator
import eval
import utils

import score_level_fusion4_labelunion

CSV_SUFFIX = '*.csv'
MIN_DIST_SECOND = 2
VIDEO_SAMPLE_RATE = 8
IMU_SAMPLE_RATE = 8

LABEL_IDLE = 'Idle'
LABEL1_INTAKE = 'Intake'
LABEL2_EAT = 'Intake-Eat'
LABEL2_DRINK = 'Intake-Drink'

def round_percent(number, rounding_digits = 4):
    return round(number * 100, rounding_digits)

def get_label(split_idxs_true, sub_labels, label, labels):
    #the following two lines should return the same results
    return [split_idx for split_idx in split_idxs_true if next(sub_labels[idx] for idx in split_idx if sub_labels[idx] != LABEL_IDLE and labels[idx] == 1) == label]

def get_label2_count(split_idxs_true, imu_labels2, labels):
    if len(split_idxs_true) == 0:
        return 0, 0, 0, 0
    eat = get_label(split_idxs_true, imu_labels2, LABEL2_EAT, labels)
    drink = get_label(split_idxs_true, imu_labels2, LABEL2_DRINK, labels)
    return \
        round_percent(len(eat) / len(split_idxs_true)), \
        round_percent(len(drink) / len(split_idxs_true)), \
        len(eat), \
        len(drink), \

def get_label4_count(split_idxs_true, imu_labels4, labels):
    if len(split_idxs_true) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    spoon = get_label(split_idxs_true, imu_labels4, 'Spoon', labels)
    fork = get_label(split_idxs_true, imu_labels4, 'Fork', labels)
    cup = get_label(split_idxs_true, imu_labels4, 'Cup', labels)
    bottle = get_label(split_idxs_true, imu_labels4, 'Bottle', labels)
    hand = get_label(split_idxs_true, imu_labels4, 'Hand', labels)
    knife = get_label(split_idxs_true, imu_labels4, 'Knife', labels)
    finger = get_label(split_idxs_true, imu_labels4, 'Finger', labels)
    return \
        round_percent(len(spoon) / len(split_idxs_true)), \
        round_percent(len(fork) / len(split_idxs_true)), \
        round_percent(len(cup)+len(bottle) / len(split_idxs_true)), \
        round_percent(len(hand) / len(split_idxs_true)), \
        round_percent(len(knife) / len(split_idxs_true)), \
        round_percent(len(finger) / len(split_idxs_true)), \
        len(spoon), \
        len(fork), \
        len(cup)+len(bottle), \
        len(hand), \
        len(knife), \
        len(finger)

def read_vid_imu_prob_file(filenamepath):
    vid_pIds = []
    vid_frames = []
    vid_labels = []
    vid_probs = []
    imu_pIds = []
    imu_frames = []
    imu_labels = []
    imu_probs = []
    imu_labels1 = []
    imu_labels2 = []
    imu_labels3 = []
    imu_labels4 = []
    imu_threshold = 0
    vid_threshold = 0
    with open(filenamepath) as prob_file:
        for row in csv.reader(prob_file, delimiter=','):
            vid_pIds.append(row[0])
            vid_frames.append(row[1])
            vid_labels.append(row[2])
            vid_probs.append(row[3])
            if vid_threshold == 0:
                vid_threshold = float(row[4])
            imu_pIds.append(row[5])
            imu_frames.append(row[6])
            imu_labels.append(row[7])
            imu_probs.append(row[8])
            if imu_threshold == 0:
                imu_threshold = float(row[9])
            imu_labels1.append(row[10])
            imu_labels2.append(row[11])
            imu_labels3.append(row[12])
            imu_labels4.append(row[13])
    return vid_pIds, vid_frames, vid_labels, vid_probs, imu_pIds, imu_frames, imu_labels, imu_probs, imu_labels1, imu_labels2, imu_labels3, imu_labels4, vid_threshold, imu_threshold

def get_all_idxs_within_mindist(idxs, min_dist, return_all = False):
    idxs = np.array(idxs)
    idxs_diff = np.diff(idxs)
    carry = 0; rem_i = []
    for i, diff in enumerate(np.concatenate(([min_dist], idxs_diff))):
        if (diff + carry < min_dist):
            if return_all and (len(rem_i) == 0 or rem_i[-1] != i-1):
                rem_i.append(i-1)
            rem_i.append(i)
            carry += diff
        else:
            carry = 0
    return np.array(rem_i)

def labels_union(vid_labels, imu_labels):
    labels_all = [1 if i==1 or v==1 else 0 for v, i in zip(vid_labels, imu_labels)]
    return np.array(labels_all)

def concat_distinct(x,y):
    if x == []:
        return np.array(y)
    if y == []:
        return np.array(x)
    return np.concatenate(x, np.setdiff1d(x,y), axis=0)

def analyse_vid_imu_probs(vid_pIds, vid_frames, vid_labels, vid_probs, imu_pIds, imu_frames, imu_labels, imu_probs, imu_labels1, imu_labels2, imu_labels3, imu_labels4, vid_threshold, imu_threshold, min_dist, detections_fus):
    def _split_idx(labels):
        idx_t = np.where(labels == 1)[0]
        t_d = np.diff(idx_t) - 1
        t = np.where(t_d > 0)[0]
        t_start = np.concatenate(([0], t + 1))
        t_end = np.concatenate((t, [idx_t.shape[0] - 1]))
        idx_start = idx_t[t_start]
        idx_end = idx_t[t_end]
        return [np.arange(start, end + 1) for start, end in zip(idx_start, idx_end)]
    
    vid_probs = np.array([float(vp) for vp in vid_probs])
    vid_labels = np.array([int(vl) for vl in vid_labels])
    imu_probs = np.array([float(ip) for ip in imu_probs])
    imu_labels = np.array([int(il) for il in imu_labels])
    vid_labels = labels_union(vid_labels, imu_labels)
    imu_labels = vid_labels

    detections_vid = eval.max_search(vid_probs, vid_threshold, min_dist)
    split_idxs_true_vid = _split_idx(vid_labels)
    split_idxs_tp_vid = [split_idx for split_idx in split_idxs_true_vid if np.sum(detections_vid[split_idx]) > 0]
    detections_imu = eval.max_search(imu_probs, imu_threshold, min_dist)
    split_idxs_true_imu = _split_idx(imu_labels)
    assert len(split_idxs_true_vid) == len(split_idxs_true_imu), 'split_idxs_true_vid_len:{0}, split_idxs_true_imu_len:{1}'.format(split_idxs_true_vid,split_idxs_true_imu)
 
    idxs_false_vid = np.where(vid_labels == 0)[0]
    idxs_false_imu = np.where(imu_labels == 0)[0]

    split_idxs_tp_imu = [split_idx for split_idx in split_idxs_true_imu if np.sum(detections_imu[split_idx]) > 0]
  
    split_idxs_tp_fus = [split_idx for split_idx in split_idxs_true_vid if np.sum(detections_fus[split_idx]) > 0]
    split_idxs_fn_fus = [split_idx for split_idx in split_idxs_true_vid if np.sum(detections_fus[split_idx]) == 0]
    split_idxs_fp_1_fus_2 = [split_idx for split_idx in split_idxs_true_vid if np.sum(detections_fus[split_idx]) == 2]
    split_idxs_fp_1_fus_3 = [split_idx for split_idx in split_idxs_true_vid if np.sum(detections_fus[split_idx]) == 3]
    split_idxs_fp_1_fus = concat_distinct(split_idxs_fp_1_fus_2, split_idxs_fp_1_fus_3)
    idxs_fp_2_fus = [idx for idx in idxs_false_vid if detections_fus[idx] == 1]

    split_idxs_tp_vid_only = [split_idx_vid for split_idx_vid in split_idxs_tp_vid if np.sum([len(np.intersect1d(split_idx_vid, split_idx_imu)) for split_idx_imu in split_idxs_tp_imu]) == 0]
    #5
    split_idxs_tp_vid_only_fus_tp = [split_idx_vid_only for split_idx_vid_only in split_idxs_tp_vid_only if np.sum([len(np.intersect1d(split_idx_vid_only, split_idx_fus)) for split_idx_fus in split_idxs_tp_fus]) > 0]
    #6
    split_idxs_tp_vid_only_fus_fn = [split_idx_vid_only for split_idx_vid_only in split_idxs_tp_vid_only if np.sum([len(np.intersect1d(split_idx_vid_only, split_idx_fus)) for split_idx_fus in split_idxs_fn_fus]) > 0]
    split_idxs_fn_overlap_vid = [split_idx_true for split_idx_true in split_idxs_true_vid if np.sum([len(np.intersect1d(split_idx_true, split_idx_vid)) for split_idx_vid in split_idxs_tp_vid]) == 0 and np.sum([len(np.intersect1d(split_idx_true, split_idx_imu)) for split_idx_imu in split_idxs_tp_imu]) == 0]
    #7
    split_idxs_fn_overlap_vid_fus_tp = [split_idx_fn_overlap_vid for split_idx_fn_overlap_vid in split_idxs_fn_overlap_vid if np.sum([len(np.intersect1d(split_idx_fn_overlap_vid, split_idx_tp_fus)) for split_idx_tp_fus in split_idxs_tp_fus]) > 0]
    #8
    split_idxs_fn_overlap_vid_fus_fn = [split_idx_fn_overlap_vid for split_idx_fn_overlap_vid in split_idxs_fn_overlap_vid if np.sum([len(np.intersect1d(split_idx_fn_overlap_vid, split_idx_fn_fus)) for split_idx_fn_fus in split_idxs_fn_fus]) > 0]
    split_idxs_fn_vid_only = [split_idx_true for split_idx_true in split_idxs_true_vid if np.sum([len(np.intersect1d(split_idx_true, split_idx_vid)) for split_idx_vid in split_idxs_tp_vid]) == 0 and np.sum([len(np.intersect1d(split_idx_true, split_idx_imu)) for split_idx_imu in split_idxs_tp_imu]) > 0]
    tp_vid_only_len = len(split_idxs_tp_vid_only)
    split_idxs_tp_imu_only = [split_idx_imu for split_idx_imu in split_idxs_tp_imu if np.sum([len(np.intersect1d(split_idx_imu, split_idx_vid)) for split_idx_vid in split_idxs_tp_vid]) == 0]
    #1
    split_idxs_tp_imu_only_fus_tp = [split_idx_imu_only for split_idx_imu_only in split_idxs_tp_imu_only if np.sum([len(np.intersect1d(split_idx_imu_only, split_idx_fus)) for split_idx_fus in split_idxs_tp_fus]) > 0]
    #2
    split_idxs_tp_imu_only_fus_fn = [split_idx_imu_only for split_idx_imu_only in split_idxs_tp_imu_only if np.sum([len(np.intersect1d(split_idx_imu_only, split_idx_fus)) for split_idx_fus in split_idxs_fn_fus]) > 0]
    split_idxs_fn_overlap_imu = [split_idx_true for split_idx_true in split_idxs_true_imu if np.sum([len(np.intersect1d(split_idx_true, split_idx_vid)) for split_idx_vid in split_idxs_tp_vid]) == 0 and np.sum([len(np.intersect1d(split_idx_true, split_idx_imu)) for split_idx_imu in split_idxs_tp_imu]) == 0]
    split_idxs_fn_imu_only = [split_idx_true for split_idx_true in split_idxs_true_imu if np.sum([len(np.intersect1d(split_idx_true, split_idx_vid)) for split_idx_vid in split_idxs_tp_vid]) > 0 and np.sum([len(np.intersect1d(split_idx_true, split_idx_imu)) for split_idx_imu in split_idxs_tp_imu]) == 0]
    tp_imu_only_len = len(split_idxs_tp_imu_only)
    assert len(split_idxs_fn_overlap_vid) == len(split_idxs_fn_overlap_imu), 'tp_vid_overlap_len:{0}, tp_imu_overlap_len:{1}'.format(split_idxs_fn_overlap_vid,split_idxs_fn_overlap_imu)
    fn_overlap_len = len(split_idxs_fn_overlap_vid)
    fn_vid_only_len = len(split_idxs_fn_vid_only)
    fn_imu_only_len = len(split_idxs_fn_imu_only)
    assert tp_vid_only_len == fn_imu_only_len, 'tp_vid_only_len {0} <> fn_imu_only_len {1}'.format(tp_vid_only_len,fn_imu_only_len)
    assert tp_imu_only_len == fn_vid_only_len, 'tp_imu_only_len {0} <> fn_vid_only_len {1}'.format(tp_imu_only_len,fn_vid_only_len)
    split_idxs_tp_vid_overlap = [split_idx_vid for split_idx_vid in split_idxs_tp_vid if np.sum([len(np.intersect1d(split_idx_vid, split_idx_imu)) for split_idx_imu in split_idxs_tp_imu]) > 0]
    #3
    split_idxs_tp_vid_overlap_fus_tp = [split_idx_tp_vid_overlap for split_idx_tp_vid_overlap in split_idxs_tp_vid_overlap if np.sum([len(np.intersect1d(split_idx_tp_vid_overlap, split_idx_fus)) for split_idx_fus in split_idxs_tp_fus]) > 0]
    #4
    split_idxs_tp_vid_overlap_fus_fn = [split_idx_tp_vid_overlap for split_idx_tp_vid_overlap in split_idxs_tp_vid_overlap if np.sum([len(np.intersect1d(split_idx_tp_vid_overlap, split_idx_fus)) for split_idx_fus in split_idxs_fn_fus]) > 0]
    split_idxs_tp_imu_overlap = [split_idx_imu for split_idx_imu in split_idxs_tp_imu if np.sum([len(np.intersect1d(split_idx_imu, split_idx_vid)) for split_idx_vid in split_idxs_tp_vid]) > 0]
    assert len(split_idxs_tp_vid_overlap) == len(split_idxs_tp_imu_overlap), 'tp_vid_overlap_len:{0}, tp_imu_overlap_len:{1}'.format(split_idxs_tp_vid_overlap,split_idxs_tp_imu_overlap)
    tp_overlap_len = len(split_idxs_tp_vid_overlap)
    true_len = tp_vid_only_len + tp_imu_only_len + tp_overlap_len + fn_overlap_len
    assert len(split_idxs_true_vid) == true_len, 'split_idxs_true_vid_len:{0}, tp_all_len:{1}'.format(split_idxs_true_vid,true_len)

    split_idxs_fp_1_vid_2 = [split_idx for split_idx in split_idxs_true_vid if np.sum(detections_vid[split_idx]) == 2]
    split_idxs_fp_1_vid_3 = [split_idx for split_idx in split_idxs_true_vid if np.sum(detections_vid[split_idx]) == 3]
    assert len([split_idx for split_idx in split_idxs_true_vid if np.sum(detections_vid[split_idx]) > 3]) == 0, 'number of {0} splits for vid contain more than two fp1'.format(len([split_idx for split_idx in split_idxs_true_vid if np.sum(detections_vid[split_idx]) > 2]))
    split_idxs_fp_1_imu_2 = [split_idx for split_idx in split_idxs_true_imu if np.sum(detections_imu[split_idx]) == 2]
    split_idxs_fp_1_imu_3 = [split_idx for split_idx in split_idxs_true_imu if np.sum(detections_imu[split_idx]) == 3]
    assert len([split_idx for split_idx in split_idxs_true_imu if np.sum(detections_imu[split_idx]) > 3]) == 0, 'number of {0} splits for imu contain more than two fp1'.format(len([split_idx for split_idx in split_idxs_true_imu if np.sum(detections_imu[split_idx]) > 2]))
    split_idxs_fp_1_vid_2_only = [split_idx_vid for split_idx_vid in split_idxs_fp_1_vid_2 if np.sum([len(np.intersect1d(split_idx_vid, split_idx_imu)) for split_idx_imu in split_idxs_fp_1_imu_2]) == 0]
    split_idxs_fp_1_imu_2_only = [split_idx_imu for split_idx_imu in split_idxs_fp_1_imu_2 if np.sum([len(np.intersect1d(split_idx_imu, split_idx_vid)) for split_idx_vid in split_idxs_fp_1_vid_2]) == 0]
    split_idxs_fp_1_vid_3_only = [split_idx_vid for split_idx_vid in split_idxs_fp_1_vid_3 if np.sum([len(np.intersect1d(split_idx_vid, split_idx_imu)) for split_idx_imu in split_idxs_fp_1_imu_3]) == 0]
    split_idxs_fp_1_imu_3_only = [split_idx_imu for split_idx_imu in split_idxs_fp_1_imu_3 if np.sum([len(np.intersect1d(split_idx_imu, split_idx_vid)) for split_idx_vid in split_idxs_fp_1_vid_3]) == 0]
    split_idxs_fp_1_vid_overlap_2 = [split_idx_vid for split_idx_vid in split_idxs_fp_1_vid_2 if np.sum([len(np.intersect1d(split_idx_vid, split_idx_imu)) for split_idx_imu in split_idxs_fp_1_imu_2]) > 0]
    split_idxs_fp_1_imu_overlap_2 = [split_idx_imu for split_idx_imu in split_idxs_fp_1_imu_2 if np.sum([len(np.intersect1d(split_idx_imu, split_idx_vid)) for split_idx_vid in split_idxs_fp_1_vid_2]) > 0]
    split_idxs_fp_1_vid_overlap_3 = [split_idx_vid for split_idx_vid in split_idxs_fp_1_vid_3 if np.sum([len(np.intersect1d(split_idx_vid, split_idx_imu)) for split_idx_imu in split_idxs_fp_1_imu_3]) > 0]
    split_idxs_fp_1_imu_overlap_3 = [split_idx_imu for split_idx_imu in split_idxs_fp_1_imu_3 if np.sum([len(np.intersect1d(split_idx_imu, split_idx_vid)) for split_idx_vid in split_idxs_fp_1_vid_3]) > 0]
    assert len(split_idxs_fp_1_vid_overlap_2) == len(split_idxs_fp_1_imu_overlap_2), 'fp_1_vid_overlap_2_len:{0}, fp_1_imu_overlap_2_len:{1}'.format(split_idxs_fp_1_vid_overlap_2,split_idxs_fp_1_imu_overlap_2)
    assert len(split_idxs_fp_1_vid_overlap_3) == len(split_idxs_fp_1_imu_overlap_3), 'fp_1_vid_overlap_3_len:{0}, fp_1_imu_overlap_3_len:{1}'.format(split_idxs_fp_1_vid_overlap_3,split_idxs_fp_1_imu_overlap_3)
    #13
    split_idxs_fp_1_vid_only_fus_fp1_2 = [split_idx_fp_1_vid_only for split_idx_fp_1_vid_only in split_idxs_fp_1_vid_2_only if np.sum([len(np.intersect1d(split_idx_fp_1_vid_only, split_idx_fus)) for split_idx_fus in split_idxs_fp_1_fus]) > 0]
    split_idxs_fp_1_vid_only_fus_fp1_3 = [split_idx_fp_1_vid_only for split_idx_fp_1_vid_only in split_idxs_fp_1_vid_3_only if np.sum([len(np.intersect1d(split_idx_fp_1_vid_only, split_idx_fus)) for split_idx_fus in split_idxs_fp_1_fus]) > 0]
    #11
    split_idxs_fp_1_imu_only_fus_fp1_2 = [split_idx_fp_1_imu_only for split_idx_fp_1_imu_only in split_idxs_fp_1_imu_2_only if np.sum([len(np.intersect1d(split_idx_fp_1_imu_only, split_idx_fus)) for split_idx_fus in split_idxs_fp_1_fus]) > 0]
    split_idxs_fp_1_imu_only_fus_fp1_3 = [split_idx_fp_1_imu_only for split_idx_fp_1_imu_only in split_idxs_fp_1_imu_3_only if np.sum([len(np.intersect1d(split_idx_fp_1_imu_only, split_idx_fus)) for split_idx_fus in split_idxs_fp_1_fus]) > 0]
    #12
    split_idxs_fp_1_vid_overlap_fus_fp1_2 = [split_idx_fp_1_vid_overlap for split_idx_fp_1_vid_overlap in split_idxs_fp_1_vid_overlap_2 if np.sum([len(np.intersect1d(split_idx_fp_1_vid_overlap, split_idx_fus)) for split_idx_fus in split_idxs_fp_1_fus]) > 0]
    split_idxs_fp_1_vid_overlap_fus_fp1_3 = [split_idx_fp_1_vid_overlap for split_idx_fp_1_vid_overlap in split_idxs_fp_1_vid_overlap_3 if np.sum([len(np.intersect1d(split_idx_fp_1_vid_overlap, split_idx_fus)) for split_idx_fus in split_idxs_fp_1_fus]) > 0]

    idxs_fp_2_vid = [idx for idx in idxs_false_vid if detections_vid[idx] == 1]
    idxs_fp_2_vid_approx_detections_len = len(get_all_idxs_within_mindist(idxs_fp_2_vid, min_dist, True))
    assert idxs_fp_2_vid_approx_detections_len == 0 , 'idxs_fp_2_vid_approx_detections_len = {0}'.format(idxs_fp_2_vid_approx_detections_len)
    idxs_fp_2_vid_union = [idx for idx in idxs_false_vid if detections_vid[idx] == 1 or detections_imu[idx] == 1]
    idxs_fp_2_vid_union = np.array(idxs_fp_2_vid_union)
    idxs_fp_2_imu = [idx for idx in idxs_false_imu if detections_imu[idx] == 1]
    idxs_fp_2_imu_approx_detections_len = len(get_all_idxs_within_mindist(idxs_fp_2_imu, min_dist, True))
    assert idxs_fp_2_imu_approx_detections_len == 0 , 'idxs_fp_2_imu_approx_detections_len = {0}'.format(idxs_fp_2_imu_approx_detections_len)
    idxs_fp_2_imu_union = [idx for idx in idxs_false_imu if detections_imu[idx] == 1 or detections_vid[idx] == 1]
    idxs_fp_2_imu_union = np.array(idxs_fp_2_imu_union)
    idxs_fp_2_vid_imu_union_not_the_same = np.where((idxs_fp_2_vid_union ==idxs_fp_2_imu_union) == False)[0]
    if len(idxs_fp_2_vid_imu_union_not_the_same) > 0: 
        print('len of idxs_fp_2_vid_imu_union_not_the_same = {0}'.format(len(idxs_fp_2_vid_imu_union_not_the_same)))
    idxs_idxs_fp_2_vid_overlap_union = get_all_idxs_within_mindist(idxs_fp_2_vid_union, min_dist, True)
    if len(idxs_idxs_fp_2_vid_overlap_union)>0:
        idxs_fp_2_vid_overlap_union = idxs_fp_2_vid_union[idxs_idxs_fp_2_vid_overlap_union]
        idxs_fp_2_vid_only = np.setdiff1d(idxs_fp_2_vid, idxs_fp_2_vid_overlap_union) 
    else:
        idxs_fp_2_vid_overlap_union = np.array([])
        idxs_fp_2_vid_only = np.array(idxs_fp_2_vid)
    idxs_idxs_fp_2_imu_overlap_union = get_all_idxs_within_mindist(idxs_fp_2_imu_union, min_dist, True)
    if len(idxs_idxs_fp_2_imu_overlap_union)>0:
        idxs_fp_2_imu_overlap_union = idxs_fp_2_imu_union[idxs_idxs_fp_2_imu_overlap_union]
        idxs_fp_2_imu_only = np.setdiff1d(idxs_fp_2_imu, idxs_fp_2_imu_overlap_union) 
    else:
        idxs_fp_2_imu_overlap_union = np.array([])
        idxs_fp_2_imu_only = np.array(idxs_fp_2_imu)
    
    idxs_fp_2_overlap = np.unique(np.sort(np.concatenate((idxs_fp_2_vid_overlap_union,idxs_fp_2_imu_overlap_union))))
    if len(idxs_fp_2_overlap) > 0:
        idxs_fp_2_overlap = np.delete(idxs_fp_2_overlap, get_all_idxs_within_mindist(idxs_fp_2_overlap, min_dist, False))
    else:
        idxs_fp_2_overlap = []

    #23
    idxs_fp_2_vid_only_fus_fp2_union = np.sort(np.concatenate((idxs_fp_2_vid_only,idxs_fp_2_fus)))
    idxs_idxs_fp_2_vid_only_fus_fp2 = get_all_idxs_within_mindist(idxs_fp_2_vid_only_fus_fp2_union, min_dist, False)
    if len(idxs_idxs_fp_2_vid_only_fus_fp2)>0:
        idxs_fp_2_vid_only_fus_fp2 = idxs_fp_2_vid_only_fus_fp2_union[idxs_idxs_fp_2_vid_only_fus_fp2]
    else:
        idxs_fp_2_vid_only_fus_fp2 = np.array([])
    #21
    idxs_fp_2_imu_only_fus_fp2_union = np.sort(np.concatenate((idxs_fp_2_imu_only,idxs_fp_2_fus)))
    idxs_idxs_fp_2_imu_only_fus_fp2 = get_all_idxs_within_mindist(idxs_fp_2_imu_only_fus_fp2_union, min_dist, False)
    if len(idxs_idxs_fp_2_imu_only_fus_fp2)>0:
        idxs_fp_2_imu_only_fus_fp2 = idxs_fp_2_imu_only_fus_fp2_union[idxs_idxs_fp_2_imu_only_fus_fp2]
    else:
        idxs_fp_2_imu_only_fus_fp2 = np.array([])
    #22
    idxs_fp_2_vid_overlap_fus_fp2_union = np.sort(np.concatenate((idxs_fp_2_overlap,idxs_fp_2_fus)))
    idxs_idxs_fp_2_vid_overlap_fus_fp2 = get_all_idxs_within_mindist(idxs_fp_2_vid_overlap_fus_fp2_union, min_dist, False)
    if len(idxs_idxs_fp_2_vid_overlap_fus_fp2)>0:
        idxs_fp_2_vid_overlap_fus_fp2 = idxs_fp_2_vid_overlap_fus_fp2_union[idxs_idxs_fp_2_vid_overlap_fus_fp2]
    else:
        idxs_fp_2_vid_overlap_fus_fp2 = np.array([])

    split_idxs_fp_1_vid_len = len(split_idxs_fp_1_vid_2)+2*len(split_idxs_fp_1_vid_3)
    split_idxs_fp_1_imu_len = len(split_idxs_fp_1_imu_2)+2*len(split_idxs_fp_1_imu_3)
    split_idxs_fp_1_vid_only_len = len(split_idxs_fp_1_vid_2_only)+2*len(split_idxs_fp_1_vid_3_only)
    split_idxs_fp_1_imu_only_len = len(split_idxs_fp_1_imu_2_only)+2*len(split_idxs_fp_1_imu_3_only)
    split_idxs_fp_1_overlap_len = len(split_idxs_fp_1_vid_overlap_2)+2*len(split_idxs_fp_1_vid_overlap_3)
    split_idxs_fp_1_len = split_idxs_fp_1_vid_only_len + split_idxs_fp_1_imu_only_len + split_idxs_fp_1_overlap_len
    idxs_fp_2_vid_len = len(idxs_fp_2_vid) 
    idxs_fp_2_imu_len = len(idxs_fp_2_imu)
    idxs_fp_2_vid_only_len = len(idxs_fp_2_vid_only) 
    idxs_fp_2_imu_only_len = len(idxs_fp_2_imu_only)
    idxs_fp_2_overlap_len = len(idxs_fp_2_overlap) 
    idxs_fp_2_len = idxs_fp_2_vid_only_len + idxs_fp_2_imu_only_len + idxs_fp_2_overlap_len

    tp_vid_only_percent = round_percent(tp_vid_only_len / true_len)
    tp_imu_only_percent = round_percent(tp_imu_only_len / true_len)
    tp_overlap_percent = round_percent(tp_overlap_len / true_len)
    fn_overlap_percent = round_percent(fn_overlap_len / true_len)
    
    tp_vid_only_label2_Eat_percent, tp_vid_only_label2_Drink_percent, tp_vid_only_label2_Eat_len, tp_vid_only_label2_Drink_len = get_label2_count(split_idxs_tp_vid_only, imu_labels2, imu_labels)
    tp_imu_only_label2_Eat_percent, tp_imu_only_label2_Drink_percent, tp_imu_only_label2_Eat_len, tp_imu_only_label2_Drink_len = get_label2_count(split_idxs_tp_imu_only, imu_labels2, imu_labels)
    tp_overlap_label2_Eat_percent, tp_overlap_label2_Drink_percent, tp_overlap_label2_Eat_len, tp_overlap_label2_Drink_len = get_label2_count(split_idxs_tp_imu_overlap, imu_labels2, imu_labels)
    fn_overlap_label2_Eat_percent, fn_overlap_label2_Drink_percent, fn_overlap_label2_Eat_len, fn_overlap_label2_Drink_len = get_label2_count(split_idxs_fn_overlap_vid, imu_labels2, imu_labels)
    
    tp_vid_only_label4_Spoon_percent, tp_vid_only_label4_Fork_percent, tp_vid_only_label4_Cup_percent, tp_vid_only_label4_Hand_percent, tp_vid_only_label4_Knife_percent, tp_vid_only_label4_Finger_percent, tp_vid_only_label4_Spoon_len, tp_vid_only_label4_Fork_len, tp_vid_only_label4_Cup_len, tp_vid_only_label4_Hand_len, tp_vid_only_label4_Knife_len, tp_vid_only_label4_Finger_len = get_label4_count(split_idxs_tp_vid_only, imu_labels4, imu_labels)
    tp_imu_only_label4_Spoon_percent, tp_imu_only_label4_Fork_percent, tp_imu_only_label4_Cup_percent, tp_imu_only_label4_Hand_percent, tp_imu_only_label4_Knife_percent, tp_imu_only_label4_Finger_percent, tp_imu_only_label4_Spoon_len, tp_imu_only_label4_Fork_len, tp_imu_only_label4_Cup_len, tp_imu_only_label4_Hand_len, tp_imu_only_label4_Knife_len, tp_imu_only_label4_Finger_len = get_label4_count(split_idxs_tp_imu_only, imu_labels4, imu_labels)
    tp_overlap_label4_Spoon_percent, tp_overlap_label4_Fork_percent, tp_overlap_label4_Cup_percent, tp_overlap_label4_Hand_percent, tp_overlap_label4_Knife_percent, tp_overlap_label4_Finger_percent, tp_overlap_label4_Spoon_len, tp_overlap_label4_Fork_len, tp_overlap_label4_Cup_len, tp_overlap_label4_Hand_len, tp_overlap_label4_Knife_len, tp_overlap_label4_Finger_len = get_label4_count(split_idxs_tp_imu_overlap, imu_labels4, imu_labels)
    fn_overlap_label4_Spoon_percent, fn_overlap_label4_Fork_percent, fn_overlap_label4_Cup_percent, fn_overlap_label4_Hand_percent, fn_overlap_label4_Knife_percent, fn_overlap_label4_Finger_percent, fn_overlap_label4_Spoon_len, fn_overlap_label4_Fork_len, fn_overlap_label4_Cup_len, fn_overlap_label4_Hand_len, fn_overlap_label4_Knife_len, fn_overlap_label4_Finger_len = get_label4_count(split_idxs_fn_overlap_vid, imu_labels4, imu_labels)

    split_idxs_fp_1_vid_percent =         round_percent(split_idxs_fp_1_vid_len / split_idxs_fp_1_len) if split_idxs_fp_1_len != 0 else 0
    split_idxs_fp_1_imu_percent =         round_percent(split_idxs_fp_1_imu_len / split_idxs_fp_1_len) if split_idxs_fp_1_len != 0 else 0
    split_idxs_fp_1_vid_only_percent =    round_percent(split_idxs_fp_1_vid_only_len / split_idxs_fp_1_len) if split_idxs_fp_1_len != 0 else 0
    split_idxs_fp_1_imu_only_percent =    round_percent(split_idxs_fp_1_imu_only_len / split_idxs_fp_1_len) if split_idxs_fp_1_len != 0 else 0
    split_idxs_fp_1_overlap_percent = round_percent(split_idxs_fp_1_overlap_len / split_idxs_fp_1_len) if split_idxs_fp_1_len != 0 else 0
    
    idxs_fp_2_vid_percent =      round_percent(idxs_fp_2_vid_len / idxs_fp_2_len) if idxs_fp_2_len != 0 else 0
    idxs_fp_2_imu_percent =      round_percent(idxs_fp_2_imu_len / idxs_fp_2_len) if idxs_fp_2_len != 0 else 0
    idxs_fp_2_vid_only_percent = round_percent(idxs_fp_2_vid_only_len / idxs_fp_2_len) if idxs_fp_2_len != 0 else 0
    idxs_fp_2_imu_only_percent = round_percent(idxs_fp_2_imu_only_len / idxs_fp_2_len) if idxs_fp_2_len != 0 else 0
    idxs_fp_2_overlap_percent =  round_percent(idxs_fp_2_overlap_len / idxs_fp_2_len) if idxs_fp_2_len != 0 else 0

    tp_vid_only_fus_tp_len = len(split_idxs_tp_vid_only_fus_tp)
    tp_vid_only_fus_fn_len = len(split_idxs_tp_vid_only_fus_fn)
    tp_imu_only_fus_tp_len = len(split_idxs_tp_imu_only_fus_tp)
    tp_imu_only_fus_fn_len = len(split_idxs_tp_imu_only_fus_fn)
    tp_vid_overlap_fus_tp_len = len(split_idxs_tp_vid_overlap_fus_tp)
    tp_vid_overlap_fus_fn_len = len(split_idxs_tp_vid_overlap_fus_fn)
    fn_overlap_vid_fus_tp_len = len(split_idxs_fn_overlap_vid_fus_tp)
    fn_overlap_vid_fus_fn_len = len(split_idxs_fn_overlap_vid_fus_fn)
    fp_1_vid_only_fus_fp1_len = len(split_idxs_fp_1_vid_only_fus_fp1_2)+2*len(split_idxs_fp_1_vid_only_fus_fp1_3)
    fp_1_imu_only_fus_fp1_len = len(split_idxs_fp_1_imu_only_fus_fp1_2)+2*len(split_idxs_fp_1_imu_only_fus_fp1_3)
    fp_1_vid_overlap_fus_fp1_len = len(split_idxs_fp_1_vid_overlap_fus_fp1_2)+2*len(split_idxs_fp_1_vid_overlap_fus_fp1_3)
    fp_2_vid_only_fus_fp2_len = len(idxs_fp_2_vid_only_fus_fp2)
    fp_2_imu_only_fus_fp2_len = len(idxs_fp_2_imu_only_fus_fp2)
    fp_2_vid_overlap_fus_fp2_len = len(idxs_fp_2_vid_overlap_fus_fp2)

    return vid_pIds[0], tp_vid_only_percent, tp_imu_only_percent, tp_overlap_percent, fn_overlap_percent, tp_vid_only_len, tp_imu_only_len, tp_overlap_len, fn_overlap_len, \
        tp_vid_only_label2_Eat_percent, tp_vid_only_label2_Drink_percent, tp_vid_only_label2_Eat_len, tp_vid_only_label2_Drink_len, \
        tp_imu_only_label2_Eat_percent, tp_imu_only_label2_Drink_percent, tp_imu_only_label2_Eat_len, tp_imu_only_label2_Drink_len, \
        tp_overlap_label2_Eat_percent, tp_overlap_label2_Drink_percent, tp_overlap_label2_Eat_len, tp_overlap_label2_Drink_len, \
        fn_overlap_label2_Eat_percent, fn_overlap_label2_Drink_percent, fn_overlap_label2_Eat_len, fn_overlap_label2_Drink_len, \
        tp_vid_only_label4_Spoon_percent, tp_vid_only_label4_Fork_percent, tp_vid_only_label4_Cup_percent, tp_vid_only_label4_Hand_percent, tp_vid_only_label4_Knife_percent, tp_vid_only_label4_Finger_percent, tp_vid_only_label4_Spoon_len, tp_vid_only_label4_Fork_len, tp_vid_only_label4_Cup_len, tp_vid_only_label4_Hand_len, tp_vid_only_label4_Knife_len, tp_vid_only_label4_Finger_len ,\
        tp_imu_only_label4_Spoon_percent, tp_imu_only_label4_Fork_percent, tp_imu_only_label4_Cup_percent, tp_imu_only_label4_Hand_percent, tp_imu_only_label4_Knife_percent, tp_imu_only_label4_Finger_percent, tp_imu_only_label4_Spoon_len, tp_imu_only_label4_Fork_len, tp_imu_only_label4_Cup_len, tp_imu_only_label4_Hand_len, tp_imu_only_label4_Knife_len, tp_imu_only_label4_Finger_len ,\
        tp_overlap_label4_Spoon_percent, tp_overlap_label4_Fork_percent, tp_overlap_label4_Cup_percent, tp_overlap_label4_Hand_percent, tp_overlap_label4_Knife_percent, tp_overlap_label4_Finger_percent, tp_overlap_label4_Spoon_len, tp_overlap_label4_Fork_len, tp_overlap_label4_Cup_len, tp_overlap_label4_Hand_len, tp_overlap_label4_Knife_len, tp_overlap_label4_Finger_len ,\
        fn_overlap_label4_Spoon_percent, fn_overlap_label4_Fork_percent, fn_overlap_label4_Cup_percent, fn_overlap_label4_Hand_percent, fn_overlap_label4_Knife_percent, fn_overlap_label4_Finger_percent, fn_overlap_label4_Spoon_len, fn_overlap_label4_Fork_len, fn_overlap_label4_Cup_len, fn_overlap_label4_Hand_len, fn_overlap_label4_Knife_len, fn_overlap_label4_Finger_len ,\
        split_idxs_fp_1_vid_percent, split_idxs_fp_1_imu_percent, split_idxs_fp_1_vid_only_percent, split_idxs_fp_1_imu_only_percent, split_idxs_fp_1_overlap_percent, \
        split_idxs_fp_1_vid_len, split_idxs_fp_1_imu_len, split_idxs_fp_1_vid_only_len, split_idxs_fp_1_imu_only_len, split_idxs_fp_1_overlap_len, split_idxs_fp_1_len, \
        idxs_fp_2_vid_percent, idxs_fp_2_imu_percent, idxs_fp_2_vid_only_percent, idxs_fp_2_imu_only_percent, idxs_fp_2_overlap_percent, \
        idxs_fp_2_vid_len, idxs_fp_2_imu_len, idxs_fp_2_vid_only_len, idxs_fp_2_imu_only_len, idxs_fp_2_overlap_len, idxs_fp_2_len, \
        tp_vid_only_fus_tp_len, tp_vid_only_fus_fn_len, tp_imu_only_fus_tp_len, tp_imu_only_fus_fn_len, tp_vid_overlap_fus_tp_len, tp_vid_overlap_fus_fn_len, \
        fn_overlap_vid_fus_tp_len, fn_overlap_vid_fus_fn_len, \
        fp_1_vid_only_fus_fp1_len, fp_1_imu_only_fus_fp1_len, fp_1_vid_overlap_fus_fp1_len, \
        fp_2_vid_only_fus_fp2_len, fp_2_imu_only_fus_fp2_len, fp_2_vid_overlap_fus_fp2_len

def analyse_vid_imu_prob_merged_files(vid_imu_prob_merge_dir, vid_imu_analysis_results_filename, min_dist, vid_threshold, imu_threshold, all_threshold, fusion_method):
    if utils.is_file(vid_imu_analysis_results_filename):
        RuntimeError("file {} already exists!".format(vid_imu_analysis_results_filename))
        
    total_tp_vid_only_len = 0
    total_tp_imu_only_len = 0
    total_tp_overlap_len = 0
    total_fn_overlap_len = 0

    total_tp_vid_only_label2_Eat_len = 0 
    total_tp_vid_only_label2_Drink_len = 0 
    total_tp_imu_only_label2_Eat_len = 0 
    total_tp_imu_only_label2_Drink_len = 0 
    total_tp_overlap_label2_Eat_len = 0 
    total_tp_overlap_label2_Drink_len = 0 
    total_fn_overlap_label2_Eat_len = 0 
    total_fn_overlap_label2_Drink_len = 0 

    total_tp_vid_only_label4_Spoon_len = 0 
    total_tp_vid_only_label4_Fork_len = 0 
    total_tp_vid_only_label4_Cup_len = 0 
    total_tp_vid_only_label4_Hand_len = 0 
    total_tp_vid_only_label4_Knife_len = 0 
    total_tp_vid_only_label4_Finger_len = 0
    total_tp_imu_only_label4_Spoon_len = 0 
    total_tp_imu_only_label4_Fork_len = 0 
    total_tp_imu_only_label4_Cup_len = 0 
    total_tp_imu_only_label4_Hand_len = 0 
    total_tp_imu_only_label4_Knife_len = 0 
    total_tp_imu_only_label4_Finger_len = 0
    total_tp_overlap_label4_Spoon_len = 0 
    total_tp_overlap_label4_Fork_len = 0 
    total_tp_overlap_label4_Cup_len = 0 
    total_tp_overlap_label4_Hand_len = 0 
    total_tp_overlap_label4_Knife_len = 0 
    total_tp_overlap_label4_Finger_len = 0
    total_fn_overlap_label4_Spoon_len = 0 
    total_fn_overlap_label4_Fork_len = 0 
    total_fn_overlap_label4_Cup_len = 0 
    total_fn_overlap_label4_Hand_len = 0 
    total_fn_overlap_label4_Knife_len = 0 
    total_fn_overlap_label4_Finger_len = 0
    total_split_idxs_fp_1_vid_len = 0
    total_split_idxs_fp_1_imu_len = 0 
    total_split_idxs_fp_1_vid_only_len = 0 
    total_split_idxs_fp_1_imu_only_len = 0 
    total_split_idxs_fp_1_overlap_len = 0
    total_split_idxs_fp_1_len = 0
    total_idxs_fp_2_vid_len = 0 
    total_idxs_fp_2_imu_len = 0
    total_idxs_fp_2_vid_only_len = 0
    total_idxs_fp_2_imu_only_len = 0
    total_idxs_fp_2_overlap_len = 0
    total_idxs_fp_2_len = 0 

    total_tp_vid_only_fus_tp_len = 0
    total_tp_vid_only_fus_fn_len = 0
    total_tp_imu_only_fus_tp_len = 0
    total_tp_imu_only_fus_fn_len = 0
    total_tp_vid_overlap_fus_tp_len = 0
    total_tp_vid_overlap_fus_fn_len = 0
    total_fn_overlap_vid_fus_tp_len = 0
    total_fn_overlap_vid_fus_fn_len = 0
    total_fp_1_vid_only_fus_fp1_len = 0
    total_fp_1_imu_only_fus_fp1_len = 0
    total_fp_1_vid_overlap_fus_fp1_len = 0
    total_fp_2_vid_only_fus_fp2_len = 0
    total_fp_2_imu_only_fus_fp2_len = 0
    total_fp_2_vid_overlap_fus_fp2_len = 0

    vid_imu_prob_filenames = glob.glob(os.path.join(vid_imu_prob_merge_dir, CSV_SUFFIX))
    with open(vid_imu_analysis_results_filename, 'w') as results_file:
        percentages = ''
        results_file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},{25},{26},{27},{28},{29},{30},{31},{32},{33},{34},{35},{36},{37},{38},{39},{40},{41},{42},{43},{44},{45},{46},{47},{48},{49},{50},{51},{52},{53},{54},{55},{56},{57},{58},{59},{60}'\
            .format('Participant Id', 'TP vid only', 'TP imu only', 'TP overlap', 'FN overlap', \
            'tp vid only Eat', 'tp vid only Drink', \
            'tp imu only Eat', 'tp imu only Drink', \
            'tp overlap Eat', 'tp overlap Drink', \
            'fn Eat', 'fn Drink', \
            'tp vid only Spoon', 'tp vid only Fork', 'tp vid only Cup', 'tp vid only Hand', 'tp vid only Knife', 'tp vid only Finger ', \
            'tp imu only Spoon', 'tp imu only Fork', 'tp imu only Cup', 'tp imu only Hand', 'tp imu only Knife', 'tp imu only Finger ', \
            'tp overlap Spoon', 'tp overlap Fork', 'tp overlap Cup', 'tp overlap Hand', 'tp overlap Knife', 'tp overlap Finger ', \
            'fn overlap Spoon', 'fn overlap Fork', 'fn overlap Cup', 'fn overlap Hand', 'fn overlap Knife', 'fn overlap Finger', \
            'vid fp1', 'imu fp1', 'vid only fp1', 'imu only fp1', 'vid imu fp1 overlap', 'vid fp2', 'imu fp2', 'vid only fp2', 'imu only fp2', 'vid imu fp2 overlap', \
            'tp vid only fus tp', 'tp vid only fus fn', 'tp imu only fus tp', 'tp imu only fus fn', 'tp vid overlap fus tp', \
            'tp vid overlap fus fn', 'fn overlap vid fus tp', 'fn overlap vid fus fn', \
            'fp 1 vid only fus fp1', 'fp 1 imu only fus fp1', 'fp 1 vid overlap fus fp1', \
            'fp 2 vid only fus fp2', 'fp 2 imu only fus fp2', 'fp 2 vid overlap fus fp2'))
        results_file.write('\n')
        for vid_imu_prob_filename in vid_imu_prob_filenames:
            vid_pIds, vid_frames, vid_labels, vid_probs, imu_pIds, imu_frames, imu_labels, imu_probs, imu_labels1, imu_labels2, imu_labels3, imu_labels4, _, _ = \
                read_vid_imu_prob_file(vid_imu_prob_filename)

            tp, fn, fp_1, fp_2, prec, rec, f1, vid_tp, vid_fn, vid_fp_1, vid_fp_2, vid_prec, vid_rec, vid_f1, imu_tp, imu_fn, imu_fp_1, imu_fp_2, imu_prec, imu_rec, imu_f1, all_probs, detections_all = \
                score_level_fusion4_labelunion.detection_fusion(vid_labels, vid_probs, imu_labels, imu_probs, min_dist, vid_threshold, imu_threshold, all_threshold, fusion_method)

            vid_pId, tp_vid_only_percent, tp_imu_only_percent, tp_overlap_percent, fn_overlap_percent, tp_vid_only_len, tp_imu_only_len, tp_overlap_len, fn_overlap_len, \
                tp_vid_only_label2_Eat_percent, tp_vid_only_label2_Drink_percent, tp_vid_only_label2_Eat_len, tp_vid_only_label2_Drink_len, \
                tp_imu_only_label2_Eat_percent, tp_imu_only_label2_Drink_percent, tp_imu_only_label2_Eat_len, tp_imu_only_label2_Drink_len, \
                tp_overlap_label2_Eat_percent, tp_overlap_label2_Drink_percent, tp_overlap_label2_Eat_len, tp_overlap_label2_Drink_len, \
                fn_overlap_label2_Eat_percent, fn_overlap_label2_Drink_percent, fn_overlap_label2_Eat_len, fn_overlap_label2_Drink_len, \
                tp_vid_only_label4_Spoon_percent, tp_vid_only_label4_Fork_percent, tp_vid_only_label4_Cup_percent, tp_vid_only_label4_Hand_percent, tp_vid_only_label4_Knife_percent, tp_vid_only_label4_Finger_percent, tp_vid_only_label4_Spoon_len, tp_vid_only_label4_Fork_len, tp_vid_only_label4_Cup_len, tp_vid_only_label4_Hand_len, tp_vid_only_label4_Knife_len, tp_vid_only_label4_Finger_len ,\
                tp_imu_only_label4_Spoon_percent, tp_imu_only_label4_Fork_percent, tp_imu_only_label4_Cup_percent, tp_imu_only_label4_Hand_percent, tp_imu_only_label4_Knife_percent, tp_imu_only_label4_Finger_percent, tp_imu_only_label4_Spoon_len, tp_imu_only_label4_Fork_len, tp_imu_only_label4_Cup_len, tp_imu_only_label4_Hand_len, tp_imu_only_label4_Knife_len, tp_imu_only_label4_Finger_len ,\
                tp_overlap_label4_Spoon_percent, tp_overlap_label4_Fork_percent, tp_overlap_label4_Cup_percent, tp_overlap_label4_Hand_percent, tp_overlap_label4_Knife_percent, tp_overlap_label4_Finger_percent, tp_overlap_label4_Spoon_len, tp_overlap_label4_Fork_len, tp_overlap_label4_Cup_len, tp_overlap_label4_Hand_len, tp_overlap_label4_Knife_len, tp_overlap_label4_Finger_len ,\
                fn_overlap_label4_Spoon_percent, fn_overlap_label4_Fork_percent, fn_overlap_label4_Cup_percent, fn_overlap_label4_Hand_percent, fn_overlap_label4_Knife_percent, fn_overlap_label4_Finger_percent, fn_overlap_label4_Spoon_len, fn_overlap_label4_Fork_len, fn_overlap_label4_Cup_len, fn_overlap_label4_Hand_len, fn_overlap_label4_Knife_len, fn_overlap_label4_Finger_len, \
                split_idxs_fp_1_vid_percent, split_idxs_fp_1_imu_percent, split_idxs_fp_1_vid_only_percent, split_idxs_fp_1_imu_only_percent, split_idxs_fp_1_overlap_percent, \
                split_idxs_fp_1_vid_len, split_idxs_fp_1_imu_len, split_idxs_fp_1_vid_only_len, split_idxs_fp_1_imu_only_len, split_idxs_fp_1_overlap_len, split_idxs_fp_1_len, \
                idxs_fp_2_vid_percent, idxs_fp_2_imu_percent, idxs_fp_2_vid_only_percent, idxs_fp_2_imu_only_percent, idxs_fp_2_overlap_percent, \
                idxs_fp_2_vid_len, idxs_fp_2_imu_len, idxs_fp_2_vid_only_len, idxs_fp_2_imu_only_len, idxs_fp_2_overlap_len, idxs_fp_2_len, \
                tp_vid_only_fus_tp_len, tp_vid_only_fus_fn_len, tp_imu_only_fus_tp_len, tp_imu_only_fus_fn_len, tp_vid_overlap_fus_tp_len, tp_vid_overlap_fus_fn_len, \
                fn_overlap_vid_fus_tp_len, fn_overlap_vid_fus_fn_len, \
                fp_1_vid_only_fus_fp1_len, fp_1_imu_only_fus_fp1_len, fp_1_vid_overlap_fus_fp1_len, \
                fp_2_vid_only_fus_fp2_len, fp_2_imu_only_fus_fp2_len, fp_2_vid_overlap_fus_fp2_len = \
                analyse_vid_imu_probs(vid_pIds, vid_frames, vid_labels, vid_probs, imu_pIds, imu_frames, imu_labels, imu_probs, imu_labels1, imu_labels2, imu_labels3, imu_labels4, vid_threshold, imu_threshold, min_dist, detections_all)

            percentages += '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},{25},{26},{27},{28},{29},{30},{31},{32},{33},{34},{35},{36},{37},{38},{39},{40},{41},{42},{43},{44},{45},{46}'\
                .format(vid_pId, tp_vid_only_percent, tp_imu_only_percent, tp_overlap_percent, fn_overlap_percent,\
                tp_vid_only_label2_Eat_percent, tp_vid_only_label2_Drink_percent, \
                tp_imu_only_label2_Eat_percent, tp_imu_only_label2_Drink_percent, \
                tp_overlap_label2_Eat_percent, tp_overlap_label2_Drink_percent, \
                fn_overlap_label2_Eat_percent, fn_overlap_label2_Drink_percent, \
                tp_vid_only_label4_Spoon_percent, tp_vid_only_label4_Fork_percent, tp_vid_only_label4_Cup_percent, tp_vid_only_label4_Hand_percent, tp_vid_only_label4_Knife_percent, tp_vid_only_label4_Finger_percent, \
                tp_imu_only_label4_Spoon_percent, tp_imu_only_label4_Fork_percent, tp_imu_only_label4_Cup_percent, tp_imu_only_label4_Hand_percent, tp_imu_only_label4_Knife_percent, tp_imu_only_label4_Finger_percent, \
                tp_overlap_label4_Spoon_percent, tp_overlap_label4_Fork_percent, tp_overlap_label4_Cup_percent, tp_overlap_label4_Hand_percent, tp_overlap_label4_Knife_percent, tp_overlap_label4_Finger_percent, \
                fn_overlap_label4_Spoon_percent, fn_overlap_label4_Fork_percent, fn_overlap_label4_Cup_percent, fn_overlap_label4_Hand_percent, fn_overlap_label4_Knife_percent, fn_overlap_label4_Finger_percent, \
                split_idxs_fp_1_vid_percent, split_idxs_fp_1_imu_percent, split_idxs_fp_1_vid_only_percent, split_idxs_fp_1_imu_only_percent, split_idxs_fp_1_overlap_percent, \
                idxs_fp_2_vid_percent, idxs_fp_2_imu_percent, idxs_fp_2_vid_only_percent, idxs_fp_2_imu_only_percent, idxs_fp_2_overlap_percent)
            results_file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},{25},{26},{27},{28},{29},{30},{31},{32},{33},{34},{35},{36},{37},{38},{39},{40},{41},{42},{43},{44},{45},{46},{47},{48},{49},{50},{51},{52},{53},{54},{55},{56},{57},{58},{59},{60}'\
                .format(vid_pId, tp_vid_only_len, tp_imu_only_len, tp_overlap_len, fn_overlap_len,\
                tp_vid_only_label2_Eat_len, tp_vid_only_label2_Drink_len, \
                tp_imu_only_label2_Eat_len, tp_imu_only_label2_Drink_len, \
                tp_overlap_label2_Eat_len, tp_overlap_label2_Drink_len, \
                fn_overlap_label2_Eat_len, fn_overlap_label2_Drink_len, \
                tp_vid_only_label4_Spoon_len, tp_vid_only_label4_Fork_len, tp_vid_only_label4_Cup_len, tp_vid_only_label4_Hand_len, tp_vid_only_label4_Knife_len, tp_vid_only_label4_Finger_len, \
                tp_imu_only_label4_Spoon_len, tp_imu_only_label4_Fork_len, tp_imu_only_label4_Cup_len, tp_imu_only_label4_Hand_len, tp_imu_only_label4_Knife_len, tp_imu_only_label4_Finger_len, \
                tp_overlap_label4_Spoon_len, tp_overlap_label4_Fork_len, tp_overlap_label4_Cup_len, tp_overlap_label4_Hand_len, tp_overlap_label4_Knife_len, tp_overlap_label4_Finger_len, \
                fn_overlap_label4_Spoon_len, fn_overlap_label4_Fork_len, fn_overlap_label4_Cup_len, fn_overlap_label4_Hand_len, fn_overlap_label4_Knife_len, fn_overlap_label4_Finger_len, \
                split_idxs_fp_1_vid_len, split_idxs_fp_1_imu_len, split_idxs_fp_1_vid_only_len, split_idxs_fp_1_imu_only_len, split_idxs_fp_1_overlap_len, \
                idxs_fp_2_vid_len, idxs_fp_2_imu_len, idxs_fp_2_vid_only_len, idxs_fp_2_imu_only_len, idxs_fp_2_overlap_len, \
                tp_vid_only_fus_tp_len, tp_vid_only_fus_fn_len, tp_imu_only_fus_tp_len, tp_imu_only_fus_fn_len, tp_vid_overlap_fus_tp_len, tp_vid_overlap_fus_fn_len, \
                fn_overlap_vid_fus_tp_len, fn_overlap_vid_fus_fn_len, \
                fp_1_vid_only_fus_fp1_len, fp_1_imu_only_fus_fp1_len, fp_1_vid_overlap_fus_fp1_len, \
                fp_2_vid_only_fus_fp2_len, fp_2_imu_only_fus_fp2_len, fp_2_vid_overlap_fus_fp2_len))
            percentages += '\n'
            results_file.write('\n')

            total_tp_vid_only_len += tp_vid_only_len
            total_tp_imu_only_len += tp_imu_only_len
            total_tp_overlap_len += tp_overlap_len
            total_fn_overlap_len += fn_overlap_len

            total_tp_vid_only_label2_Eat_len   += tp_vid_only_label2_Eat_len 
            total_tp_vid_only_label2_Drink_len += tp_vid_only_label2_Drink_len
            total_tp_imu_only_label2_Eat_len   += tp_imu_only_label2_Eat_len 
            total_tp_imu_only_label2_Drink_len += tp_imu_only_label2_Drink_len 
            total_tp_overlap_label2_Eat_len    += tp_overlap_label2_Eat_len  
            total_tp_overlap_label2_Drink_len  += tp_overlap_label2_Drink_len 
            total_fn_overlap_label2_Eat_len    += fn_overlap_label2_Eat_len 
            total_fn_overlap_label2_Drink_len  += fn_overlap_label2_Drink_len 
            
            total_tp_vid_only_label4_Spoon_len += tp_vid_only_label4_Spoon_len 
            total_tp_vid_only_label4_Fork_len  += tp_vid_only_label4_Fork_len 
            total_tp_vid_only_label4_Cup_len   += tp_vid_only_label4_Cup_len 
            total_tp_vid_only_label4_Hand_len  += tp_vid_only_label4_Hand_len 
            total_tp_vid_only_label4_Knife_len += tp_vid_only_label4_Knife_len 
            total_tp_vid_only_label4_Finger_len+= tp_vid_only_label4_Finger_len
            total_tp_imu_only_label4_Spoon_len += tp_imu_only_label4_Spoon_len 
            total_tp_imu_only_label4_Fork_len  += tp_imu_only_label4_Fork_len 
            total_tp_imu_only_label4_Cup_len   += tp_imu_only_label4_Cup_len 
            total_tp_imu_only_label4_Hand_len  += tp_imu_only_label4_Hand_len 
            total_tp_imu_only_label4_Knife_len += tp_imu_only_label4_Knife_len 
            total_tp_imu_only_label4_Finger_len+= tp_imu_only_label4_Finger_len
            total_tp_overlap_label4_Spoon_len  += tp_overlap_label4_Spoon_len 
            total_tp_overlap_label4_Fork_len   += tp_overlap_label4_Fork_len 
            total_tp_overlap_label4_Cup_len    += tp_overlap_label4_Cup_len 
            total_tp_overlap_label4_Hand_len   += tp_overlap_label4_Hand_len 
            total_tp_overlap_label4_Knife_len  += tp_overlap_label4_Knife_len 
            total_tp_overlap_label4_Finger_len += tp_overlap_label4_Finger_len
            total_fn_overlap_label4_Spoon_len          += fn_overlap_label4_Spoon_len 
            total_fn_overlap_label4_Fork_len           += fn_overlap_label4_Fork_len 
            total_fn_overlap_label4_Cup_len            += fn_overlap_label4_Cup_len 
            total_fn_overlap_label4_Hand_len           += fn_overlap_label4_Hand_len 
            total_fn_overlap_label4_Knife_len          += fn_overlap_label4_Knife_len 
            total_fn_overlap_label4_Finger_len         += fn_overlap_label4_Finger_len
            total_split_idxs_fp_1_vid_len           += split_idxs_fp_1_vid_len
            total_split_idxs_fp_1_imu_len           += split_idxs_fp_1_imu_len 
            total_split_idxs_fp_1_vid_only_len      += split_idxs_fp_1_vid_only_len
            total_split_idxs_fp_1_imu_only_len      += split_idxs_fp_1_imu_only_len
            total_split_idxs_fp_1_overlap_len       += split_idxs_fp_1_overlap_len
            total_split_idxs_fp_1_len               += split_idxs_fp_1_len
            total_idxs_fp_2_vid_len                 += idxs_fp_2_vid_len
            total_idxs_fp_2_imu_len                 += idxs_fp_2_imu_len
            total_idxs_fp_2_vid_only_len            += idxs_fp_2_vid_only_len
            total_idxs_fp_2_imu_only_len            += idxs_fp_2_imu_only_len
            total_idxs_fp_2_overlap_len             += idxs_fp_2_overlap_len
            total_idxs_fp_2_len                     += idxs_fp_2_len

            total_tp_vid_only_fus_tp_len            +=tp_vid_only_fus_tp_len
            total_tp_vid_only_fus_fn_len            +=tp_vid_only_fus_fn_len
            total_tp_imu_only_fus_tp_len            +=tp_imu_only_fus_tp_len
            total_tp_imu_only_fus_fn_len            +=tp_imu_only_fus_fn_len
            total_tp_vid_overlap_fus_tp_len         +=tp_vid_overlap_fus_tp_len
            total_tp_vid_overlap_fus_fn_len         +=tp_vid_overlap_fus_fn_len
            total_fn_overlap_vid_fus_tp_len         +=fn_overlap_vid_fus_tp_len
            total_fn_overlap_vid_fus_fn_len         +=fn_overlap_vid_fus_fn_len
            total_fp_1_vid_only_fus_fp1_len         +=fp_1_vid_only_fus_fp1_len
            total_fp_1_imu_only_fus_fp1_len         +=fp_1_imu_only_fus_fp1_len
            total_fp_1_vid_overlap_fus_fp1_len      +=fp_1_vid_overlap_fus_fp1_len
            total_fp_2_vid_only_fus_fp2_len         +=fp_2_vid_only_fus_fp2_len
            total_fp_2_imu_only_fus_fp2_len         +=fp_2_imu_only_fus_fp2_len
            total_fp_2_vid_overlap_fus_fp2_len      +=fp_2_vid_overlap_fus_fp2_len

            print('{} is done!'.format(vid_pId))

        num_of_participants = len(vid_imu_prob_filenames)
        total_true = total_tp_vid_only_len + total_tp_imu_only_len + total_tp_overlap_len + total_fn_overlap_len
        total_tp_vid_only_percent = round_percent(total_tp_vid_only_len / total_true)
        total_tp_imu_only_percent = round_percent(total_tp_imu_only_len / total_true)
        total_tp_overlap_percent = round_percent(total_tp_overlap_len / total_true)
        total_fn_overlap_percent = round_percent(total_fn_overlap_len / total_true)

        total_tp_vid_only_label2_Eat_percent = round_percent(total_tp_vid_only_label2_Eat_len / total_tp_vid_only_len)
        total_tp_vid_only_label2_Drink_percent = round_percent(total_tp_vid_only_label2_Drink_len / total_tp_vid_only_len)
        total_tp_imu_only_label2_Eat_percent = round_percent(total_tp_imu_only_label2_Eat_len / total_tp_imu_only_len)
        total_tp_imu_only_label2_Drink_percent = round_percent(total_tp_imu_only_label2_Drink_len / total_tp_imu_only_len)
        total_tp_overlap_label2_Eat_percent = round_percent(total_tp_overlap_label2_Eat_len / total_tp_overlap_len)
        total_tp_overlap_label2_Drink_percent = round_percent(total_tp_overlap_label2_Drink_len / total_tp_overlap_len)
        total_fn_overlap_label2_Eat_percent = round_percent(total_fn_overlap_label2_Eat_len / total_fn_overlap_len)
        total_fn_overlap_label2_Drink_percent = round_percent(total_fn_overlap_label2_Drink_len / total_fn_overlap_len)

        total_tp_vid_only_label4_Spoon_percent = round_percent(total_tp_vid_only_label4_Spoon_len / total_tp_vid_only_len)
        total_tp_vid_only_label4_Fork_percent = round_percent(total_tp_vid_only_label4_Fork_len / total_tp_vid_only_len)
        total_tp_vid_only_label4_Cup_percent = round_percent(total_tp_vid_only_label4_Cup_len / total_tp_vid_only_len)
        total_tp_vid_only_label4_Hand_percent = round_percent(total_tp_vid_only_label4_Hand_len / total_tp_vid_only_len)
        total_tp_vid_only_label4_Knife_percent = round_percent(total_tp_vid_only_label4_Knife_len / total_tp_vid_only_len)
        total_tp_vid_only_label4_Finger_percent = round_percent(total_tp_vid_only_label4_Finger_len / total_tp_vid_only_len)
        total_tp_imu_only_label4_Spoon_percent = round_percent(total_tp_imu_only_label4_Spoon_len / total_tp_imu_only_len)
        total_tp_imu_only_label4_Fork_percent = round_percent(total_tp_imu_only_label4_Fork_len / total_tp_imu_only_len)
        total_tp_imu_only_label4_Cup_percent = round_percent(total_tp_imu_only_label4_Cup_len / total_tp_imu_only_len)
        total_tp_imu_only_label4_Hand_percent = round_percent(total_tp_imu_only_label4_Hand_len / total_tp_imu_only_len)
        total_tp_imu_only_label4_Knife_percent = round_percent(total_tp_imu_only_label4_Knife_len / total_tp_imu_only_len)
        total_tp_imu_only_label4_Finger_percent = round_percent(total_tp_imu_only_label4_Finger_len / total_tp_imu_only_len)
        total_tp_overlap_label4_Spoon_percent = round_percent(total_tp_overlap_label4_Spoon_len / total_tp_overlap_len)
        total_tp_overlap_label4_Fork_percent = round_percent(total_tp_overlap_label4_Fork_len / total_tp_overlap_len)
        total_tp_overlap_label4_Cup_percent = round_percent(total_tp_overlap_label4_Cup_len / total_tp_overlap_len)
        total_tp_overlap_label4_Hand_percent = round_percent(total_tp_overlap_label4_Hand_len / total_tp_overlap_len)
        total_tp_overlap_label4_Knife_percent = round_percent(total_tp_overlap_label4_Knife_len / total_tp_overlap_len)
        total_tp_overlap_label4_Finger_percent = round_percent(total_tp_overlap_label4_Finger_len / total_tp_overlap_len)
        total_fn_overlap_label4_Spoon_percent = round_percent(total_fn_overlap_label4_Spoon_len / total_fn_overlap_len)
        total_fn_overlap_label4_Fork_percent = round_percent(total_fn_overlap_label4_Fork_len / total_fn_overlap_len)
        total_fn_overlap_label4_Cup_percent = round_percent(total_fn_overlap_label4_Cup_len / total_fn_overlap_len)
        total_fn_overlap_label4_Hand_percent = round_percent(total_fn_overlap_label4_Hand_len / total_fn_overlap_len)
        total_fn_overlap_label4_Knife_percent = round_percent(total_fn_overlap_label4_Knife_len / total_fn_overlap_len)
        total_fn_overlap_label4_Finger_percent = round_percent(total_fn_overlap_label4_Finger_len / total_fn_overlap_len)
        
        total_split_idxs_fp_1_vid_percent = round_percent(total_split_idxs_fp_1_vid_len / total_split_idxs_fp_1_len) if total_split_idxs_fp_1_len != 0 else 0
        total_split_idxs_fp_1_imu_percent = round_percent(total_split_idxs_fp_1_imu_len / total_split_idxs_fp_1_len) if total_split_idxs_fp_1_len != 0 else 0
        total_split_idxs_fp_1_vid_only_percent = round_percent(total_split_idxs_fp_1_vid_only_len / total_split_idxs_fp_1_len) if total_split_idxs_fp_1_len != 0 else 0
        total_split_idxs_fp_1_imu_only_percent = round_percent(total_split_idxs_fp_1_imu_only_len / total_split_idxs_fp_1_len) if total_split_idxs_fp_1_len != 0 else 0
        total_split_idxs_fp_1_overlap_percent = round_percent(total_split_idxs_fp_1_overlap_len / total_split_idxs_fp_1_len) if total_split_idxs_fp_1_len != 0 else 0

        total_idxs_fp_2_vid_percent = round_percent(total_idxs_fp_2_vid_len / total_idxs_fp_2_len) if total_idxs_fp_2_len != 0 else 0
        total_idxs_fp_2_imu_percent = round_percent(total_idxs_fp_2_imu_len / total_idxs_fp_2_len) if total_idxs_fp_2_len != 0 else 0
        total_idxs_fp_2_vid_only_percent = round_percent(total_idxs_fp_2_vid_only_len / total_idxs_fp_2_len) if total_idxs_fp_2_len != 0 else 0
        total_idxs_fp_2_imu_only_percent = round_percent(total_idxs_fp_2_imu_only_len / total_idxs_fp_2_len) if total_idxs_fp_2_len != 0 else 0
        total_idxs_fp_2_overlap_percent = round_percent(total_idxs_fp_2_overlap_len / total_idxs_fp_2_len) if total_idxs_fp_2_len != 0 else 0

        percentages += '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},{25},{26},{27},{28},{29},{30},{31},{32},{33},{34},{35},{36},{37},{38},{39},{40},{41},{42},{43},{44},{45},{46}'\
            .format('All', total_tp_vid_only_percent, total_tp_imu_only_percent, total_tp_overlap_percent, total_fn_overlap_percent, \
            total_tp_vid_only_label2_Eat_percent, total_tp_vid_only_label2_Drink_percent, \
            total_tp_imu_only_label2_Eat_percent, total_tp_imu_only_label2_Drink_percent, \
            total_tp_overlap_label2_Eat_percent, total_tp_overlap_label2_Drink_percent, \
            total_fn_overlap_label2_Eat_percent, total_fn_overlap_label2_Drink_percent, \
            total_tp_vid_only_label4_Spoon_percent, total_tp_vid_only_label4_Fork_percent, total_tp_vid_only_label4_Cup_percent, total_tp_vid_only_label4_Hand_percent, total_tp_vid_only_label4_Knife_percent, total_tp_vid_only_label4_Finger_percent, \
            total_tp_imu_only_label4_Spoon_percent, total_tp_imu_only_label4_Fork_percent, total_tp_imu_only_label4_Cup_percent, total_tp_imu_only_label4_Hand_percent, total_tp_imu_only_label4_Knife_percent, total_tp_imu_only_label4_Finger_percent, \
            total_tp_overlap_label4_Spoon_percent, total_tp_overlap_label4_Fork_percent, total_tp_overlap_label4_Cup_percent, total_tp_overlap_label4_Hand_percent, total_tp_overlap_label4_Knife_percent, total_tp_overlap_label4_Finger_percent, \
            total_fn_overlap_label4_Spoon_percent, total_fn_overlap_label4_Fork_percent, total_fn_overlap_label4_Cup_percent, total_fn_overlap_label4_Hand_percent, total_fn_overlap_label4_Knife_percent, total_fn_overlap_label4_Finger_percent, \
            total_split_idxs_fp_1_vid_percent, total_split_idxs_fp_1_imu_percent, total_split_idxs_fp_1_vid_only_percent, total_split_idxs_fp_1_imu_only_percent, total_split_idxs_fp_1_overlap_percent, \
            total_idxs_fp_2_vid_percent, total_idxs_fp_2_imu_percent, total_idxs_fp_2_vid_only_percent, total_idxs_fp_2_imu_only_percent, total_idxs_fp_2_overlap_percent)
        
        total_precision = utils.calc_precision(total_tp_imu_only_len+total_tp_vid_only_len+total_tp_overlap_len,total_idxs_fp_2_len+total_split_idxs_fp_1_len)
        total_recall = utils.calc_recall(total_tp_imu_only_len+total_tp_vid_only_len+total_tp_overlap_len,total_tp_imu_only_len+total_tp_vid_only_len+total_fn_overlap_len)
        total_F1 = utils.calc_f1(total_precision,total_recall,3)
        percentages += '\n'
        percentages += 'F1,{0},prec,{1},rec,{2}'.format(total_F1,total_precision,total_recall)
        total_precision = utils.calc_precision(total_tp_imu_only_len+total_tp_vid_only_len+total_tp_overlap_len,total_idxs_fp_2_overlap_len+total_split_idxs_fp_1_overlap_len)
        total_recall = utils.calc_recall(total_tp_imu_only_len+total_tp_vid_only_len+total_tp_overlap_len,0+0+total_fn_overlap_len)
        total_F1 = utils.calc_f1(total_precision,total_recall,3)
        percentages += '\n'
        percentages += 'F1,{0},prec,{1},rec,{2}'.format(total_F1,total_precision,total_recall)
        
        results_file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},{25},{26},{27},{28},{29},{30},{31},{32},{33},{34},{35},{36},{37},{38},{39},{40},{41},{42},{43},{44},{45},{46},{47},{48},{49},{50},{51},{52},{53},{54},{55},{56},{57},{58},{59},{60}'\
            .format('All', total_tp_vid_only_len, total_tp_imu_only_len, total_tp_overlap_len, total_fn_overlap_len, \
            total_tp_vid_only_label2_Eat_len, total_tp_vid_only_label2_Drink_len, \
            total_tp_imu_only_label2_Eat_len, total_tp_imu_only_label2_Drink_len, \
            total_tp_overlap_label2_Eat_len, total_tp_overlap_label2_Drink_len, \
            total_fn_overlap_label2_Eat_len, total_fn_overlap_label2_Drink_len, \
            total_tp_vid_only_label4_Spoon_len, total_tp_vid_only_label4_Fork_len, total_tp_vid_only_label4_Cup_len, total_tp_vid_only_label4_Hand_len, total_tp_vid_only_label4_Knife_len, total_tp_vid_only_label4_Finger_len, \
            total_tp_imu_only_label4_Spoon_len, total_tp_imu_only_label4_Fork_len, total_tp_imu_only_label4_Cup_len, total_tp_imu_only_label4_Hand_len, total_tp_imu_only_label4_Knife_len, total_tp_imu_only_label4_Finger_len, \
            total_tp_overlap_label4_Spoon_len, total_tp_overlap_label4_Fork_len, total_tp_overlap_label4_Cup_len, total_tp_overlap_label4_Hand_len, total_tp_overlap_label4_Knife_len, total_tp_overlap_label4_Finger_len, \
            total_fn_overlap_label4_Spoon_len, total_fn_overlap_label4_Fork_len, total_fn_overlap_label4_Cup_len, total_fn_overlap_label4_Hand_len, total_fn_overlap_label4_Knife_len, total_fn_overlap_label4_Finger_len, \
            total_split_idxs_fp_1_vid_len, total_split_idxs_fp_1_imu_len, total_split_idxs_fp_1_vid_only_len, total_split_idxs_fp_1_imu_only_len, total_split_idxs_fp_1_overlap_len, \
            total_idxs_fp_2_vid_len, total_idxs_fp_2_imu_len, total_idxs_fp_2_vid_only_len, total_idxs_fp_2_imu_only_len, total_idxs_fp_2_overlap_len, \
            total_tp_vid_only_fus_tp_len, total_tp_vid_only_fus_fn_len, total_tp_imu_only_fus_tp_len, total_tp_imu_only_fus_fn_len, total_tp_vid_overlap_fus_tp_len, total_tp_vid_overlap_fus_fn_len, \
            total_fn_overlap_vid_fus_tp_len, total_fn_overlap_vid_fus_fn_len, \
            total_fp_1_vid_only_fus_fp1_len, total_fp_1_imu_only_fus_fp1_len, total_fp_1_vid_overlap_fus_fp1_len, \
            total_fp_2_vid_only_fus_fp2_len, total_fp_2_imu_only_fus_fp2_len, total_fp_2_vid_overlap_fus_fp2_len))

        #results_file.write('\n')
        #results_file.write('\n')
        #results_file.write('\n')
        #results_file.write(percentages)
        results_file.write('\n')


def main(args=None):
    min_dist = MIN_DIST_SECOND * VIDEO_SAMPLE_RATE
    vid_threshold, imu_threshold, all_threshold = score_level_fusion4_labelunion.calc_threshold(args.vid_imu_prob_merge_dir_eval_org, args.vid_col_label, args.vid_col_prob, args.imu_col_label, args.imu_col_prob, min_dist, args.fusion_method)
    analyse_vid_imu_prob_merged_files(args.vid_imu_prob_merge_dir, args.vid_imu_analysis_results_filename, min_dist, vid_threshold, imu_threshold, all_threshold, args.fusion_method)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add other labels to prob files')
    parser.add_argument('--vid_col_label', type=int, default=2, nargs='?', help='Col number of label in csv for video')
    parser.add_argument('--vid_col_prob', type=int, default=3, nargs='?', help='Col number of probability in csv for video')
    parser.add_argument('--imu_col_label', type=int, default=7, nargs='?', help='Col number of label in csv for imu')
    parser.add_argument('--imu_col_prob', type=int, default=8, nargs='?', help='Col number of probability in csv for imu')
    parser.add_argument('--fusion_method', type=str, default='max', nargs='?', help='fusion method') #max,min,sum,wsum
    #phase 1 parameters
    parser.add_argument('--vid_imu_prob_merge_dir_eval_org', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\resnet_slowfast_117000_inertial8_4000\eval_org', nargs='?', help='')
    parser.add_argument('--vid_imu_prob_merge_dir', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\resnet_slowfast_117000_inertial8_4000\alllabels\test', nargs='?', help='Directory to create merged video and imu prob files.')
    parser.add_argument('--vid_imu_analysis_results_filename', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\resnet_slowfast_117000_imu8_4000_fusion_test_gesture_analysis_fusion.csv', nargs='?', help='Directory to create merged video and imu prob files.')

    #phase 2 parameters
    #parser.add_argument('--vid_imu_prob_merge_dir_eval_org', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\resnet_slowfast_162000_inertial8_4000\eval_org', nargs='?', help='')
    #parser.add_argument('--vid_imu_prob_merge_dir', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\resnet_slowfast_162000_inertial8_4000\alllabels\test', nargs='?', help='Directory to create merged video and imu prob files.')
    #parser.add_argument('--vid_imu_analysis_results_filename', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\resnet_slowfast_162000_imu8_4000_fusion_test_gesture_analysis_fusion.csv', nargs='?', help='Directory to create merged video and imu prob files.')
    args = parser.parse_args()
    main(args)
