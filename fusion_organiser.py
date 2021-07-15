import os
import eval
import argparse
import numpy as np
from decision_level_fusion_labelunion import *
import utils
from fusion_utils2 import *

CSV_SUFFIX = '*.csv'
MIN_DIST_SECOND = 2
VIDEO_SAMPLE_RATE = 8
IMU_SAMPLE_RATE = 8
EVAL_LIST_PHASE1 = ['1020_1','1043_1','1068_1','1091_1','1112_1']
EVAL_LIST_PHASE2 = ['1068_2','2018_2','2045_2','2068_2','2094_2']

def get_eval_list(phase_number):
    if phase_number == 1:
        return EVAL_LIST_PHASE1
    else:
        return EVAL_LIST_PHASE2

def rename_files_if_required(dir, phasenumber):
    filenames = glob.glob(os.path.join(dir, CSV_SUFFIX))
    for file_name in filenames:
        #FIX THE TYPE IN VIDEO FILE NAMES 1004_1..csv => 1004_1.csv
        if '..' in file_name:
            os.rename(file_name, file_name.replace('..csv', '.csv'))
        filename_withoutpath_withoutextention = utils.get_file_name_from_path(file_name, True)
        if len(filename_withoutpath_withoutextention) == 4:
            file_newname = utils.add_postfix_to_filepathname(file_name, '_' + str(phasenumber))
            os.rename(file_name, file_newname)

def get_subject_id(file_name):
    return file_name.split('_')[-2] + '_' + file_name.split('_')[-1]

def organise_folders(vid_imu_prob_merge_dir_eval_train, vid_imu_prob_merge_dir_train, vid_imu_prob_merge_dir_eval, vid_imu_prob_merge_dir_test, phase_number):
    utils.create_dir_if_required(vid_imu_prob_merge_dir_eval)
    utils.create_dir_if_required(vid_imu_prob_merge_dir_train)
    eval_filenames = glob.glob(os.path.join(vid_imu_prob_merge_dir_eval_train, CSV_SUFFIX))
    eval_list = get_eval_list(phase_number)
    for file_name in eval_filenames:
        filename_withoutpath_withoutextention = utils.get_file_name_from_path(file_name, True)
        subject_id = get_subject_id(filename_withoutpath_withoutextention)
        if subject_id in eval_list:
            utils.copy_file(file_name, vid_imu_prob_merge_dir_eval)    
            utils.copy_file(file_name, os.path.join(vid_imu_prob_merge_dir_eval+"_sub", filename_withoutpath_withoutextention))    
        else:
            utils.copy_file(file_name, vid_imu_prob_merge_dir_train)    
    test_filenames = glob.glob(os.path.join(vid_imu_prob_merge_dir_test, CSV_SUFFIX))
    for file_name in test_filenames:
        filename_withoutpath_withoutextention = utils.get_file_name_from_path(file_name, True)
        utils.copy_file(file_name, os.path.join(vid_imu_prob_merge_dir_test+"_sub", filename_withoutpath_withoutextention))    

def main(args=None):
    rename_files_if_required(args.vid_prob_dir_eval, args.phase_number)
    rename_files_if_required(args.vid_prob_dir_test, args.phase_number)
    # create the merged filse with 0 for thresholds for eval
    FusionDto.merge_vid_imu_prob_files(args.vid_prob_dir_eval, args.imu_prob_dir_eval, args.vid_imu_prob_merge_dir_eval_org, args.vid_imu_prob_merge_filename_prefix, 0, 0, args.phase_number, True, False)
    min_dist = MIN_DIST_SECOND * VIDEO_SAMPLE_RATE
    vid_threshold, imu_threshold, all_threshold = calc_threshold(args.vid_imu_prob_merge_dir_eval_org, args.vid_col_label, args.vid_col_prob, args.imu_col_label, args.imu_col_prob, min_dist)
    # recreate the merged files with the right values for thresholds for eval
    FusionDto.merge_vid_imu_prob_files(args.vid_prob_dir_eval, args.imu_prob_dir_eval, args.vid_imu_prob_merge_dir_eval_org, args.vid_imu_prob_merge_filename_prefix, vid_threshold, imu_threshold, args.phase_number, True, True)
    detection_fusion_files(args.vid_imu_prob_merge_dir_eval_org, args.vid_imu_analysis_results_filename_eval, min_dist, vid_threshold, imu_threshold, all_threshold)
    FusionDto.merge_vid_imu_prob_files(args.vid_prob_dir_test, args.imu_prob_dir_test, args.vid_imu_prob_merge_dir_test, args.vid_imu_prob_merge_filename_prefix, vid_threshold, imu_threshold, args.phase_number, True, True)
    detection_fusion_files(args.vid_imu_prob_merge_dir_test, args.vid_imu_analysis_results_filename_test, min_dist, vid_threshold, imu_threshold, all_threshold)
    if args.organise_folders_for_deeplearning:
        organise_folders(args.vid_imu_prob_merge_dir_eval_org, args.vid_imu_prob_merge_dir_train, args.vid_imu_prob_merge_dir_eval, args.vid_imu_prob_merge_dir_test, args.phase_number)
        #utils.delete_dir_and_all_contents(args.vid_imu_prob_merge_dir_eval_org)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add other labels to prob files')
    parser.add_argument('--vid_col_label', type=int, default=2, nargs='?', help='Col number of label in csv for video')
    parser.add_argument('--vid_col_prob', type=int, default=3, nargs='?', help='Col number of probability in csv for video')
    parser.add_argument('--imu_col_label', type=int, default=7, nargs='?', help='Col number of label in csv for imu')
    parser.add_argument('--imu_col_prob', type=int, default=8, nargs='?', help='Col number of probability in csv for imu')
    parser.add_argument('--organise_folders_for_deeplearning', type=bool, default=False, nargs='?', help='Devide eval files into training and validations sets if true')
    
    #phase 1 parameters
    parser.add_argument('--vid_prob_dir_eval', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\resnet_slowfast\resnet_slowfast_public_117000_valid', nargs='?', help='Directory that contains video prob files.')
    parser.add_argument('--vid_prob_dir_test', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\resnet_slowfast\resnet_slowfast_public_117000_test', nargs='?', help='Directory that contains video prob files.')
    parser.add_argument('--imu_prob_dir_eval', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\est.d4ks1357d2tl.valid.cl.b256.93.8_std_uni_no_smo.dis\best_checkpoints\4000\prob', nargs='?', help='Directory that contains inertial prob files.')
    parser.add_argument('--imu_prob_dir_test', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\est.d4ks1357d2tl.valid.cl.b256.93.8_std_uni_no_smo.dis\best_checkpoints\4000\prob_test', nargs='?', help='Directory that contains inertial prob files.')
    parser.add_argument('--vid_imu_prob_merge_dir_eval_org', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\resnet_slowfast_117000_inertial8_4000\alllabels\eval_org', nargs='?', help='')
    parser.add_argument('--vid_imu_prob_merge_dir_train', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\resnet_slowfast_117000_inertial8_4000\alllabels\train', nargs='?', help='Directory to create merged video and imu prob files.')
    parser.add_argument('--vid_imu_prob_merge_dir_eval', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\resnet_slowfast_117000_inertial8_4000\alllabels\eval', nargs='?', help='Directory to create merged video and imu prob files.')
    parser.add_argument('--vid_imu_prob_merge_dir_test', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\resnet_slowfast_117000_inertial8_4000\alllabels\test', nargs='?', help='Directory to create merged video and imu prob files.')
    parser.add_argument('--vid_imu_prob_merge_filename_prefix', type=str, default='', nargs='?', help='Directory to create merged video and imu prob files.') # resnet_slowfast_117000_inertial8_4000_
    parser.add_argument('--vid_imu_analysis_results_filename_eval', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\resnet_slowfast_117000_imu8_4000_fusion_eval_highprob_calcthreshold_labelunion.csv', nargs='?', help='Directory to create merged video and imu prob files.')
    parser.add_argument('--vid_imu_analysis_results_filename_test', type=str, default=r'<ROOT FOLDER>\data\oreba-dis probabilities\resnet_slowfast_117000_imu8_4000_fusion_test_highprob_calcthreshold_labelunion.csv', nargs='?', help='Directory to create merged video and imu prob files.')
    parser.add_argument('--phase_number', type=int, default=1, nargs='?', help='number of data collection phase')
     
    #phase 2 parameters
    #parser.add_argument('--vid_prob_dir_eval', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\resnet_slowfast\resnet_slowfast_sha_public_162000_valid', nargs='?', help='Directory that contains video prob files.')
    #parser.add_argument('--vid_prob_dir_test', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\resnet_slowfast\resnet_slowfast_sha_public_162000_test', nargs='?', help='Directory that contains video prob files.')
    #parser.add_argument('--imu_prob_dir_eval', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\est.d4ks1357d2tl.valid.cl.b256.93.8_std_uni_no_smo.sha\best_checkpoints\4000\prob', nargs='?', help='Directory that contains inertial prob files.')
    #parser.add_argument('--imu_prob_dir_test', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\est.d4ks1357d2tl.valid.cl.b256.93.8_std_uni_no_smo.sha\best_checkpoints\4000\prob_test', nargs='?', help='Directory that contains inertial prob files.')
    #parser.add_argument('--vid_imu_prob_merge_dir_eval_org', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\resnet_slowfast_162000_inertial8_4000\alllabels\eval_org', nargs='?', help='')
    ##parser.add_argument('--vid_imu_prob_merge_dir_train', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\resnet_slowfast_162000_inertial8_4000\alllabels\train', nargs='?', help='Directory to create merged video and imu prob files.')
    ##parser.add_argument('--vid_imu_prob_merge_dir_eval', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\resnet_slowfast_162000_inertial8_4000\alllabels\eval', nargs='?', help='Directory to create merged video and imu prob files.')
    #parser.add_argument('--vid_imu_prob_merge_dir_test', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\resnet_slowfast_162000_inertial8_4000\alllabels\test', nargs='?', help='Directory to create merged video and imu prob files.')
    #parser.add_argument('--vid_imu_prob_merge_filename_prefix', type=str, default='', nargs='?', help='Directory to create merged video and imu prob files.') # resnet_slowfast_162000_inertial8_4000_
    #parser.add_argument('--vid_imu_analysis_results_filename_eval', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\resnet_slowfast_162000_imu8_4000_fusion_eval_highprob_calcthreshold_labelunion.csv', nargs='?', help='Directory to create merged video and imu prob files.')
    #parser.add_argument('--vid_imu_analysis_results_filename_test', type=str, default=r'<ROOT FOLDER>\data\oreba-sha probabilities\resnet_slowfast_162000_imu8_4000_fusion_test_highprob_calcthreshold_labelunion.csv', nargs='?', help='Directory to create merged video and imu prob files.')
    #parser.add_argument('--phase_number', type=int, default=2, nargs='?', help='number of data collection phase')
   
    args = parser.parse_args()
    main(args)

