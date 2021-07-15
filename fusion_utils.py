"""Fusion utility functions"""

import argparse
import csv
import glob
import numpy as np
import os
#import tensorflow as tf
import operator
import eval2
import utils
from scipy import signal

CSV_SUFFIX = '*.csv'


def round_percent(number, rounding_digits = 4):
    return round(number * 100, rounding_digits)


def read_vid_prob_file(filenamepath):
    pIds = []
    frames = []
    labels = []
    probs = []
    with open(filenamepath) as prob_file:
        for row in csv.reader(prob_file, delimiter=','):
            pId = row[0]
            if utils.IfStringRepresentsFloat(pId) or utils.IfStringRepresentsInt(pId) or isinstance(pId, float) or isinstance(pId, int):
               pId = str(int(float(pId)))+'_1'
            pIds.append(pId)
            frames.append(int(float(row[1])))
            labels.append(int(float(row[2])))
            probs.append(float(row[3]))
    return pIds, frames, labels, probs


def read_imu_prob_file(filenamepath, retrieve_extra = True):
    pIds = []
    frames = []
    labels = []
    probs = []
    labels1 = []
    labels2 = []
    labels3 = []
    labels4 = []
    with open(filenamepath) as prob_file:
        for row in csv.reader(prob_file, delimiter=','):
            frames.append(int(row[0]))
            labels.append(row[1])
            probs.append(row[2])
            if retrieve_extra:
                pIds.append(row[6]) #this should change to 3 later
                labels1.append(row[7])
                labels2.append(row[3])
                labels3.append(row[4])
                labels4.append(row[5])
    return pIds, frames, labels, probs, labels1, labels2, labels3, labels4


def upsample_vid_prob(pIds, frames, labels, probs, upsample_rate):
    upsample_pIds = []
    upsample_frames = []
    upsample_labels = []
    upsample_probs = []

    for i in range(0,len(frames)):
        for j in range(0, upsample_rate):
            upsample_frame = ((frames[0]) * upsample_rate) if i == 0 and j == 0 else upsample_frame + 1
            upsample_frames.append(upsample_frame)
            upsample_pIds.append(pIds[i])
            upsample_labels.append(labels[i])
            upsample_probs.append(probs[i])
    return upsample_pIds, upsample_frames, upsample_labels, upsample_probs


def downsample(probs, downsampling_rate):
    if downsampling_rate == 1:
        return probs
    probs = signal.decimate(probs, downsampling_rate, axis=0)
    return probs


def write_vid_prob_to_file(pIds, frames, labels, probs, filenamepath):
    with open(filenamepath, 'w') as prob_file:
        for i in range(0, len(frames)):
            prob_file.write('{0},{1},{2},{3}'.format(pIds[i],frames[i],labels[i],probs[i]))
            prob_file.write('\n')


def upsample_vid_prob_files(vid_prob_dir, upsample_dir, upsample_rate):
    utils.create_dir_if_required(upsample_dir)
    prob_filenames = glob.glob(os.path.join(vid_prob_dir, CSV_SUFFIX))
    for prob_filename in prob_filenames:
        prob_new_filename = os.path.join(upsample_dir, utils.get_file_name_from_path(prob_filename))
        if utils.is_file(prob_new_filename):
            continue
        pIds, frames, labels, probs = read_vid_prob_file(prob_filename)
        upsample_pIds, upsample_frames, upsample_labels, upsample_probs = upsample_vid_prob(pIds, frames, labels, probs, upsample_rate)
        write_vid_prob_to_file(upsample_pIds, upsample_frames, upsample_labels, upsample_probs, prob_new_filename)

def count_sync_labels_vid_imu(vid_labels, imu_labels):
    vid_labels = np.array(vid_labels)
    imu_labels = np.array(imu_labels)
    idxs_true_video = np.where(vid_labels == '1')[0]
    idxs_true_imu = np.where(imu_labels == '1')[0]
    return len([idx for idx in idxs_true_video if idx in idxs_true_imu])

def find_best_match_labels_vid_imu(vid_labels, imu_labels, max_search_window):
    matches = {}
    for i in range(0,max_search_window):
        matches[i] = count_sync_labels_vid_imu(vid_labels, imu_labels[i:])
    for i in range(1,max_search_window):
        matches[-1 * i] = count_sync_labels_vid_imu(vid_labels[i:], imu_labels)
    return max(matches, key=matches.get), matches

def sync_vid_imu(vid_pIds, vid_frames, vid_labels, vid_probs, imu_pIds, imu_frames, imu_labels, imu_probs, imu_labels1, imu_labels2, imu_labels3, imu_labels4, sync_extra_imu = True):
    #best_match, matches = find_best_match_labels_vid_imu(vid_labels,
    #imu_labels, 32)
    ##vis_utils.plot_dictionary(matches)
    #if best_match != 4:
    #    print(vid_pIds[0]+" bm:{} ,".format(best_match))
    #else:
    #    print(vid_pIds[0]+" bm:{} ,".format(best_match))
    #    return vid_pIds, vid_frames, vid_labels, vid_probs,
    #    imu_pIds[best_match:], imu_frames[best_match:],
    #    imu_labels[best_match:], imu_probs[best_match:],
    #    imu_labels1[best_match:], imu_labels2[best_match:],
    #    imu_labels3[best_match:], imu_labels4[best_match:]
    #else:
    #    best_match *= -1
    #    return vid_pIds[best_match:], vid_frames[best_match:],
    #    vid_labels[best_match:], vid_probs[best_match:], imu_pIds, imu_frames,
    #    imu_labels, imu_probs, imu_labels1, imu_labels2, imu_labels3,
    #    imu_labels4
    frames_def = imu_frames[0] - vid_frames[0]
    if sync_extra_imu:
        if frames_def == 0:
            return vid_pIds, vid_frames, vid_labels, vid_probs, imu_pIds, imu_frames, imu_labels, imu_probs, imu_labels1, imu_labels2, imu_labels3, imu_labels4
        elif frames_def >= 0: # e.g., 121 - 120
            return vid_pIds[frames_def:], vid_frames[frames_def:], vid_labels[frames_def:], vid_probs[frames_def:], imu_pIds, imu_frames, imu_labels, imu_probs, imu_labels1, imu_labels2, imu_labels3, imu_labels4
        else:
            frames_def *= -1
            return vid_pIds, vid_frames, vid_labels, vid_probs, imu_pIds[frames_def:], imu_frames[frames_def:], imu_labels[frames_def:], imu_probs[frames_def:], imu_labels1[frames_def:], imu_labels2[frames_def:], imu_labels3[frames_def:], imu_labels4[frames_def:]
    else:
        if frames_def == 0:
            return vid_pIds, vid_frames, vid_labels, vid_probs, imu_pIds, imu_frames, imu_labels, imu_probs, [], [], [], []
        elif frames_def >= 0: # e.g., 121 - 120
            return vid_pIds[frames_def:], vid_frames[frames_def:], vid_labels[frames_def:], vid_probs[frames_def:], imu_pIds, imu_frames, imu_labels, imu_probs, [], [], [], []
        else:
            frames_def *= -1
            return vid_pIds, vid_frames, vid_labels, vid_probs, imu_pIds[frames_def:], imu_frames[frames_def:], imu_labels[frames_def:], imu_probs[frames_def:], [], [], [], []

def read_vid_imu_prob_file(filenamepath, retrieve_extra = True):
    vid_pIds = []
    vid_frames = []
    vid_labels = []
    vid_probs = []
    vid_thresholds = []
    imu_pIds = []
    imu_frames = []
    imu_labels = []
    imu_probs = []
    imu_thresholds = []
    imu_labels1 = []
    imu_labels2 = []
    imu_labels3 = []
    imu_labels4 = []
    with open(filenamepath) as prob_file:
        for row in csv.reader(prob_file, delimiter=','):
            vid_pIds.append(row[0])
            vid_frames.append(row[1])
            vid_labels.append(row[2])
            vid_probs.append(row[3])
            vid_thresholds.append(row[4])
            imu_pIds.append(row[5])
            imu_frames.append(row[6])
            imu_labels.append(row[7])
            imu_probs.append(row[8])
            imu_thresholds.append(row[9])
            if retrieve_extra:
                imu_labels1.append(row[8])
                imu_labels2.append(row[9])
                imu_labels3.append(row[10])
                imu_labels4.append(row[11])
    return vid_pIds, vid_frames, vid_labels, vid_probs, vid_thresholds, imu_pIds, imu_frames, imu_labels, imu_probs, imu_thresholds, imu_labels1, imu_labels2, imu_labels3, imu_labels4


def write_vid_imu_prob_file(vid_pIds, vid_frames, vid_labels, vid_probs, imu_pIds, imu_frames, imu_labels, imu_probs, imu_labels1, imu_labels2, imu_labels3, imu_labels4, vid_imu_filenamepath, write_extra_imu):
    if imu_pIds == []:
        imu_pIds = vid_pIds
    with open(vid_imu_filenamepath, 'w') as merge_file:
        length = len(vid_frames) if len(vid_frames) < len(imu_frames) else len(imu_frames) 
        if(write_extra_imu):
            for i in range(0, length):
                merge_file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}'.format(vid_pIds[i], vid_frames[i], vid_labels[i], vid_probs[i], imu_pIds[i], imu_frames[i], imu_labels[i], imu_probs[i], imu_labels1[i], imu_labels2[i], imu_labels3[i], imu_labels4[i]))
                merge_file.write('\n')
        else:
            for i in range(0, length):
                merge_file.write('{0},{1},{2},{3},{4},{5},{6},{7}'.format(vid_pIds[i], vid_frames[i], vid_labels[i], vid_probs[i], imu_pIds[i], imu_frames[i], imu_labels[i], imu_probs[i]))
                merge_file.write('\n')

def merge_vid_imu_prob_file(vid_filenamepath, imu_filenamepath, vid_imu_filenamepath, retrieve_extra_imu = True):
    vid_pIds, vid_frames, vid_labels, vid_probs = read_vid_prob_file(vid_filenamepath)
    imu_pIds, imu_frames, imu_labels, imu_probs, imu_labels1, imu_labels2, imu_labels3, imu_labels4 = read_imu_prob_file(imu_filenamepath, retrieve_extra_imu)
    vid_pIds, vid_frames, vid_labels, vid_probs, imu_pIds, imu_frames, imu_labels, imu_probs, imu_labels1, imu_labels2, imu_labels3, imu_labels4 = \
        sync_vid_imu(vid_pIds, vid_frames, vid_labels, vid_probs, imu_pIds, imu_frames, imu_labels, imu_probs, imu_labels1, imu_labels2, imu_labels3, imu_labels4, retrieve_extra_imu)
    write_vid_imu_prob_file(vid_pIds, vid_frames, vid_labels, vid_probs, imu_pIds, imu_frames, imu_labels, imu_probs, imu_labels1, imu_labels2, imu_labels3, imu_labels4, vid_imu_filenamepath, retrieve_extra_imu)


def merge_vid_imu_prob_files(vid_prob_dir, imu_prob_dir, vid_imu_prob_merge_dir, retrieve_extra_imu = True):
    utils.create_dir_if_required(vid_imu_prob_merge_dir)
    vid_prob_filenames = glob.glob(os.path.join(vid_prob_dir, CSV_SUFFIX))
    for vid_prob_filename in vid_prob_filenames:
        vid_prob_filename_withoutpath = utils.get_file_name_from_path(vid_prob_filename)
        imu_prob_filename = os.path.join(imu_prob_dir, vid_prob_filename_withoutpath)
        assert utils.is_file(imu_prob_filename),"file {0} does not exist".format(imu_prob_filename)
        vid_imu_prob_filename = os.path.join(vid_imu_prob_merge_dir, vid_prob_filename_withoutpath)
        if utils.is_file(vid_imu_prob_filename):
            continue
        merge_vid_imu_prob_file(vid_prob_filename, imu_prob_filename, vid_imu_prob_filename, retrieve_extra_imu)


def get_label(split_idxs_true, sub_labels, label):
    return [split_idx for split_idx in split_idxs_true if next(sub_labels[idx] for idx in split_idx if sub_labels[idx] != 'Idle') == label]


def get_label2_count(split_idxs_true, imu_labels2):
    if len(split_idxs_true) == 0:
        return 0, 0, 0, 0, 0, 0
    return \
        round_percent(len(get_label(split_idxs_true, imu_labels2, 'Eat')) / len(split_idxs_true)), \
        round_percent(len(get_label(split_idxs_true, imu_labels2, 'Drink')) / len(split_idxs_true)), \
        round_percent(len(get_label(split_idxs_true, imu_labels2, 'Lick')) / len(split_idxs_true)), \
        len(get_label(split_idxs_true, imu_labels2, 'Eat')), \
        len(get_label(split_idxs_true, imu_labels2, 'Drink')), \
        len(get_label(split_idxs_true, imu_labels2, 'Lick'))

def get_label4_count(split_idxs_true, imu_labels4):
    if len(split_idxs_true) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    return \
        round_percent(len(get_label(split_idxs_true, imu_labels4, 'Spoon')) / len(split_idxs_true)), \
        round_percent(len(get_label(split_idxs_true, imu_labels4, 'Fork')) / len(split_idxs_true)), \
        round_percent(len(get_label(split_idxs_true, imu_labels4, 'Cup')) / len(split_idxs_true)), \
        round_percent(len(get_label(split_idxs_true, imu_labels4, 'Hand')) / len(split_idxs_true)), \
        round_percent(len(get_label(split_idxs_true, imu_labels4, 'Knife')) / len(split_idxs_true)), \
        round_percent(len(get_label(split_idxs_true, imu_labels4, 'Finger')) / len(split_idxs_true)), \
        len(get_label(split_idxs_true, imu_labels4, 'Spoon')), \
        len(get_label(split_idxs_true, imu_labels4, 'Fork')), \
        len(get_label(split_idxs_true, imu_labels4, 'Cup')), \
        len(get_label(split_idxs_true, imu_labels4, 'Hand')), \
        len(get_label(split_idxs_true, imu_labels4, 'Knife')), \
        len(get_label(split_idxs_true, imu_labels4, 'Finger'))

