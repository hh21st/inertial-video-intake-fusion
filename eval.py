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
absl.app.flags.DEFINE_integer(
    name='min_dist', default=128, help='Minimum frames between detections.')
absl.app.flags.DEFINE_float(
    name='threshold', default=0.9, help='Detection threshold probability')
absl.app.flags.DEFINE_enum(
    name='eval_mode', default='predict', enum_values=['estimate', 'predict'], 
    help='estimation using validation set or predicting using test set')
absl.app.flags.DEFINE_float(
    name='min_threshold', default=0.5, help='Minimum detection threshold probability')
absl.app.flags.DEFINE_float(
    name='max_threshold', default=1, help='Maximum detection threshold probability')
absl.app.flags.DEFINE_float(
    name='inc_threshold', default=0.001, help='Increment for detection threshold search')
absl.app.flags.DEFINE_integer(
    name='col_label', default=1, help='Col number of label in csv')
absl.app.flags.DEFINE_integer(
    name='col_prob', default=2, help='Col number of probability in csv')

CSV_SUFFIX = '*.csv'

def import_probs_and_labels(filepath, col_label, col_prob):
    """Import probabilities and labels from csv"""
    filenames = glob.glob(os.path.join(filepath, CSV_SUFFIX))
    assert filenames, "No prob files were found"
    labels = []
    probs = []
    for filename in filenames:
        with open(filename) as dest_f:
            for row in csv.reader(dest_f, delimiter=','):
                labels.append(int(row[col_label]))
                probs.append(float(row[col_prob]))
    labels = np.array(labels)
    probs = np.array(probs)

    return probs, labels

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

def eval_stage_2(dets, labels):
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
    if tp > 0:
        prec = tp / (tp + fp_1 + fp_2)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
    else:
        prec = 0
        rec = 0
        f1 = 0
    return tp, fn, fp_1, fp_2, prec, rec, f1

def main(args=None):
    # Import the probs and labels from csv
    probs, labels = import_probs_and_labels(FLAGS.prob_dir, FLAGS.col_label, FLAGS.col_prob)
    # Calculate UAR for Stage I
    uar = eval_stage_1(probs, labels)
    parent_dir = utils.get_parent_dir(FLAGS.prob_dir)
    parent_dir_name = utils.get_current_dir_name(parent_dir)
    result_file_name_suffix = '_f1score.txt' if FLAGS.eval_mode == 'estimate' else '_f1score_test.txt' 
    result_file_name = os.path.join(parent_dir, parent_dir_name + result_file_name_suffix)
    if not FLAGS.overwrite and os.path.isfile(result_file_name):
        return -1,-1,-1,-1,-1,-1,-1,-1,'-1'
    result_file = open(result_file_name, 'w')
    result_file.write('UAR: {}'.format(uar))
    # Perform grid search
    if FLAGS.eval_mode == 'estimate':
        # All evaluated threshold values
        threshold_vals = np.arange(FLAGS.min_threshold, FLAGS.max_threshold, FLAGS.inc_threshold)
        f1_results = []
        for threshold in threshold_vals:
            # Perform max search
            dets = max_search(probs, threshold, FLAGS.min_dist)
            # Calculate Stage II
            _, _, _, _, _, _, f1 = eval_stage_2(dets, labels)
            f1_results.append(f1)
        # Find best threshold
        best_threshold = threshold_vals[np.argmax(f1_results)]
        final_dets = max_search(probs, best_threshold, FLAGS.min_dist)
        tp, fn, fp_1, fp_2, prec, rec, f1 = eval_stage_2(final_dets, labels)
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
        tp, fn, fp_1, fp_2, prec, rec, f1 = eval_stage_2(dets, labels)
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
    result_file.close()
    return uar, tp, fn, fp_1, fp_2, prec, rec, f1, best_threshold
# Run
if __name__ == '__main__':
    absl.app.run(main=main)
