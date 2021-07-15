"""Add other labels to prob files."""

#import sys; sys.path.append('C:/H/PhD/ORIBA/Model/Code/ED/detection')

import argparse
import csv
import glob
import numpy as np
import os
#import tensorflow as tf
import operator
import eval
import utils
import vis_utils
import fusion_utils
from fusion_utils2 import *

CSV_SUFFIX = '*.csv'
MIN_DIST_SECOND = 2
VIDEO_SAMPLE_RATE = 8
IMU_SAMPLE_RATE = 8

def probs_fusion_max(vid_probs, imu_probs):
    probs_all = [i if i>v else v for v, i in zip(vid_probs, imu_probs)]
    return np.array(probs_all)

def probs_fusion_min(vid_probs, imu_probs):
    probs_all = [i if i<v else v for v, i in zip(vid_probs, imu_probs)]
    return np.array(probs_all)

def probs_fusion_average(vid_probs, imu_probs):
    probs_all = [(i+v)/2 for v, i in zip(vid_probs, imu_probs)]
    return probs_all

def probs_fusion_weighted_average(vid_probs, imu_probs, vid_threshold, imu_threshold):
    probs_all = [(imu_threshold*i+vid_threshold*v)/(imu_threshold+vid_threshold) for v, i in zip(vid_probs, imu_probs)]
    return probs_all

def labels_union(vid_labels, imu_labels):
    labels_all = [1 if i==1 or v==1 else 0 for v, i in zip(vid_labels, imu_labels)]
    return np.array(labels_all)

def detection_fusion(vid_labels, vid_probs, imu_labels, imu_probs, min_dist, vid_threshold, imu_threshold, all_threshold, fusion_method):
    vid_probs = np.array([float(f) for f in vid_probs])
    imu_probs = np.array([float(f) for f in imu_probs])
    vid_labels = np.array([int(i) for i in vid_labels])
    imu_labels = np.array([int(i) for i in imu_labels])
    detections_vid = eval.max_search(vid_probs, vid_threshold, min_dist)
    vid_tp, vid_fn, vid_fp_1, vid_fp_2, vid_prec, vid_rec, vid_f1 = eval.eval_stage_2(detections_vid, vid_labels)
    detections_imu = eval.max_search(imu_probs, imu_threshold, min_dist)
    imu_tp, imu_fn, imu_fp_1, imu_fp_2, imu_prec, imu_rec, imu_f1 = eval.eval_stage_2(detections_imu, imu_labels)  
    
    if fusion_method == 'max': # max score
        all_probs = probs_fusion_max(vid_probs, imu_probs)
    elif fusion_method == 'min': # min score
        all_probs = probs_fusion_min(vid_probs, imu_probs)
    elif fusion_method == 'sum': # simple sum
        all_probs = probs_fusion_average(vid_probs, imu_probs)
    elif fusion_method == 'wsum': # weighted sum
        all_probs = probs_fusion_weighted_average(vid_probs, imu_probs, vid_threshold, imu_threshold)
    else:
        raise Exception('Fusion method {0} is not supported!'.format(fusion_method))

    all_labels = labels_union(vid_labels, imu_labels)
    detections_all = eval.max_search(all_probs, all_threshold, min_dist)
    tp, fn, fp_1, fp_2, prec, rec, f1 = eval.eval_stage_2(detections_all, all_labels)
    return tp, fn, fp_1, fp_2, prec, rec, f1, vid_tp, vid_fn, vid_fp_1, vid_fp_2, vid_prec, vid_rec, vid_f1, imu_tp, imu_fn, imu_fp_1, imu_fp_2, imu_prec, imu_rec, imu_f1, all_probs, detections_all 

def calc_threshold(prob_merge_dir, vid_col_label, vid_col_prob, imu_col_label, imu_col_prob, min_dist, fusion_method, min_threshold = 0.5, max_threshold = 1, inc_threshold = 0.001):
    vid_files_probs, vid_files_labels = eval.import_probs_and_labels(prob_merge_dir, vid_col_label, vid_col_prob)
    imu_files_probs, imu_files_labels = eval.import_probs_and_labels(prob_merge_dir, imu_col_label, imu_col_prob)

    vid_threshold = calc_threshold_by_probs(vid_files_probs, vid_files_labels, min_dist)
    imu_threshold = calc_threshold_by_probs(imu_files_probs, imu_files_labels, min_dist)
            
    if fusion_method == 'max': # max score
        all_files_probs = probs_fusion_max(vid_files_probs, imu_files_probs)
    elif fusion_method == 'min': # min score
        all_files_probs = probs_fusion_min(vid_files_probs, imu_files_probs)
    elif fusion_method == 'sum': # simple sum
        all_files_probs = probs_fusion_average(vid_files_probs, imu_files_probs)
    elif fusion_method == 'wsum': # weighted sum
        all_files_probs = probs_fusion_weighted_average(vid_files_probs, imu_files_probs, vid_threshold, imu_threshold)
    else:
        raise Exception('Fusion method {0} is not supported!'.format(fusion_method))

    all_files_labels = labels_union(vid_files_labels, imu_files_labels)
    all_threshold = calc_threshold_by_probs(all_files_probs, all_files_labels, min_dist)

    return vid_threshold, imu_threshold, all_threshold

def calc_threshold_by_probs(probs, labels, min_dist, min_threshold = 0.5, max_threshold = 1, inc_threshold = 0.001):
    threshold_vals = np.arange(min_threshold, max_threshold, inc_threshold)
    f1_results = []
    for threshold in threshold_vals:
        # Perform max search
        dets = eval.max_search(probs, threshold, min_dist)
        # Calculate Stage II
        _, _, _, _, _, _, f1 = eval.eval_stage_2(dets, labels)
        f1_results.append(f1)
    # Find best threshold
    best_threshold = threshold_vals[np.argmax(f1_results)]
    return round(best_threshold,3)

def detection_fusion_files(vid_imu_prob_merge_dir, vid_imu_analysis_results_filename, min_dist, vid_threshold, imu_threshold, all_threshold, fusion_method):
    #if utils.is_file(vid_imu_analysis_results_filename):
    #    RuntimeError("file {} already exists!".format(vid_imu_analysis_results_filename))
    vid_imu_analysis_results_filename = utils.add_postfix_to_filepathname(vid_imu_analysis_results_filename, '_' + fusion_method)   
    vid_imu_prob_filenames = glob.glob(os.path.join(vid_imu_prob_merge_dir, CSV_SUFFIX))
    with open(vid_imu_analysis_results_filename, 'w') as results_file:
        results_file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21}'\
            .format('Id', 'TP', 'FN', 'FP1', 'FP2', 'Prec', 'Rec', 'F1', 'vid TP', 'vid FN', 'vid FP1', 'vid FP2', 'vid Prec', 'vid Rec', 'vid F1', 'imu TP', 'imu FN', 'imu FP1', 'imu FP2', 'imu Prec', 'imu Rec', 'imu F1'))
        results_file.write('\n')
        total_tp, total_fn, total_fp_1, total_fp_2 = 0, 0, 0, 0 
        total_vid_tp, total_vid_fn, total_vid_fp_1, total_vid_fp_2 = 0, 0, 0, 0 
        total_imu_tp, total_imu_fn, total_imu_fp_1, total_imu_fp_2 = 0, 0, 0, 0 
        for vid_imu_prob_filename in vid_imu_prob_filenames:
            vid_pIds, vid_frames, vid_labels, vid_probs, _, imu_pIds, imu_frames, imu_labels, imu_probs, _, imu_labels1, imu_labels2, imu_labels3, imu_labels4 = \
                fusion_utils.read_vid_imu_prob_file(vid_imu_prob_filename, False)
            #vid_frames = np.array([int(i) for i in vid_frames])
            #vid_probs = np.array([float(f) for f in vid_probs])
            #imu_probs = np.array([float(f) for f in imu_probs])
            #vid_labels = np.array([int(i) for i in vid_labels])
            #imu_labels = np.array([int(i) for i in imu_labels])
            tp, fn, fp_1, fp_2, prec, rec, f1, vid_tp, vid_fn, vid_fp_1, vid_fp_2, vid_prec, vid_rec, vid_f1, imu_tp, imu_fn, imu_fp_1, imu_fp_2, imu_prec, imu_rec, imu_f1, all_probs, detections_all = \
                detection_fusion(vid_labels, vid_probs, imu_labels, imu_probs, min_dist, vid_threshold, imu_threshold, all_threshold, fusion_method)
            results_file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21}'\
                .format(vid_pIds[0], tp, fn, fp_1, fp_2, prec, rec, f1, vid_tp, vid_fn, vid_fp_1, vid_fp_2, vid_prec, vid_rec, vid_f1, imu_tp, imu_fn, imu_fp_1, imu_fp_2, imu_prec, imu_rec, imu_f1))
            results_file.write('\n')
            total_tp += tp
            total_fn += fn
            total_fp_1 += fp_1
            total_fp_2 += fp_2
            total_vid_tp += vid_tp
            total_vid_fn += vid_fn
            total_vid_fp_1 += vid_fp_1
            total_vid_fp_2 += vid_fp_2
            total_imu_tp += imu_tp
            total_imu_fn += imu_fn
            total_imu_fp_1 += imu_fp_1
            total_imu_fp_2 += imu_fp_2

            #vis_utils.plot_probs_1(all_probs, vid_labels, vid_probs, imu_probs, vid_frames, vid_threshold, imu_threshold, utils.get_file_name_from_path(vid_imu_analysis_results_filename, True) ,vid_pIds[0])
            #vis_utils.plot_probs_2(all_probs, vid_labels, imu_labels, vid_probs, imu_probs, vid_frames, vid_threshold, imu_threshold, utils.get_file_name_from_path(vid_imu_analysis_results_filename, True) ,vid_pIds[0])

            print('{} is done!'.format(vid_pIds[0]))

        total_prec = utils.calc_precision(total_tp, total_fp_1 + total_fp_2)
        total_rec = utils.calc_recall(total_tp, total_fn)
        total_f1 = utils.calc_f1(total_prec, total_rec, 3)
        total_prec = round(total_prec, 3)
        total_rec = round(total_rec, 3)

        total_vid_prec = utils.calc_precision(total_vid_tp, total_vid_fp_1 + total_vid_fp_2)
        total_vid_rec = utils.calc_recall(total_vid_tp, total_vid_fn)
        total_vid_f1 = utils.calc_f1(total_vid_prec, total_vid_rec, 3)
        total_vid_prec = round(total_vid_prec, 3)
        total_vid_rec = round(total_vid_rec, 3)

        total_imu_prec = utils.calc_precision(total_imu_tp, total_imu_fp_1 + total_imu_fp_2)
        total_imu_rec = utils.calc_recall(total_imu_tp, total_imu_fn)
        total_imu_f1 = utils.calc_f1(total_imu_prec, total_imu_rec, 3)
        total_imu_prec = round(total_imu_prec, 3)
        total_imu_rec = round(total_imu_rec, 3)
        results_file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21}'\
            .format('All', total_tp, total_fn, total_fp_1, total_fp_2, total_prec, total_rec, total_f1, total_vid_tp, total_vid_fn, total_vid_fp_1, total_vid_fp_2, total_vid_prec, total_vid_rec, total_vid_f1, total_imu_tp, total_imu_fn, total_imu_fp_1, total_imu_fp_2, total_imu_prec, total_imu_rec, total_imu_f1))

        results_file.write('\n')

def main(args=None):
    min_dist = MIN_DIST_SECOND * VIDEO_SAMPLE_RATE
    vid_threshold, imu_threshold, all_threshold = calc_threshold(args.vid_imu_prob_merge_dir_eval_org, args.vid_col_label, args.vid_col_prob, args.imu_col_label, args.imu_col_prob, min_dist, args.fusion_method)
    detection_fusion_files(args.vid_imu_prob_merge_dir, args.vid_imu_analysis_results_filename, min_dist, vid_threshold, imu_threshold, all_threshold, args.fusion_method)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add other labels to prob files')
    parser.add_argument('--vid_col_label', type=int, default=2, nargs='?', help='Col number of label in csv for video')
    parser.add_argument('--vid_col_prob', type=int, default=3, nargs='?', help='Col number of probability in csv for video')
    parser.add_argument('--imu_col_label', type=int, default=7, nargs='?', help='Col number of label in csv for imu')
    parser.add_argument('--imu_col_prob', type=int, default=8, nargs='?', help='Col number of probability in csv for imu')
    parser.add_argument('--fusion_method', type=str, default='wsum', nargs='?', help='fusion method') #max,min,sum,wsum
    #phase 1 args
    #parser.add_argument('--vid_imu_prob_merge_dir_eval_org', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\resnet_slowfast_117000_inertial8_4000\eval_org', nargs='?', help='')
    #parser.add_argument('--vid_imu_prob_merge_dir', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\video_inertial\test', nargs='?', help='Directory to create merged video and imu prob files.')
    #parser.add_argument('--vid_imu_analysis_results_filename', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\resnet_slowfast_117000_imu8_4000_fusion_test_score_level_labelunion.csv', nargs='?', help='Directory to create merged video and imu prob files.')
    #phase 2 args
    parser.add_argument('--vid_imu_prob_merge_dir_eval_org', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\resnet_slowfast_162000_inertial8_4000\eval_org', nargs='?', help='')
    parser.add_argument('--vid_imu_prob_merge_dir', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\video_inertial\test', nargs='?', help='Directory to create merged video and imu prob files.')
    parser.add_argument('--vid_imu_analysis_results_filename', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\resnet_slowfast_162000_imu8_4000_fusion_test_score_level_labelunion.csv', nargs='?', help='Directory to create merged video and imu prob files.')


    args = parser.parse_args()
    main(args)
