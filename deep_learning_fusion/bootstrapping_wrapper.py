import argparse
import csv
import glob
import numpy as np
import os
#import tensorflow as tf
import operator
import eval
import utils
import fusion_utils
import bootstrapping

CSV_SUFFIX = '*.csv'

def read_f1score_results_file(f1score_results_filename):
    model_names, checkpoints, thresholds, f1s, uars , model_paths= [], [], [], [], [], []
    with open(f1score_results_filename) as f1score_results:
        reader = csv.reader(f1score_results, delimiter=',')
        next(reader)
        for row in reader:
            model_names.append(row[0])
            checkpoints.append(row[1])
            thresholds.append(float(row[4]))
            f1s.append(row[5])
            uars.append(row[6])
            model_paths.append(row[7])
    return np.array(model_names), np.array(checkpoints), np.array(thresholds), np.array(f1s), np.array(uars), np.array(model_paths)


def bootstrap_wrapper_model(model_name, checkpoint, threshold, model_path, bootstrapping_samples_no, confidence_interval1, confidence_interval2):
    checkpoint_dir = os.path.join(model_path,'best_checkpoints',checkpoint)
    prob_dir = os.path.join(checkpoint_dir,'prob_test')
    bootstrapping_step1_filename = os.path.join(checkpoint_dir,'bootstrapping_step1_results.csv')
    if '.64_' in model_name:
        mindist = 128
    elif '.32_' in model_name:
        mindist = 64
    elif '.16_' in model_name:
        mindist = 32
    elif '.8_' in model_name:
        mindist = 16
    else:
        RuntimeError("Frequency cannot be obtained from model name: {}".format(model_name))
    quantile_lower_1, quantile_upper_1, quantile_lower_2, quantile_upper_2, pId_sample_items, sample_f1s = \
        bootstrapping.bootstrap(prob_dir, bootstrapping_step1_filename, threshold, mindist, bootstrapping_samples_no, confidence_interval1, confidence_interval2)
    return quantile_lower_1, quantile_upper_1, quantile_lower_2, quantile_upper_2, pId_sample_items, sample_f1s

def samples_are_equal(previous_sample_f1s, sample_f1s):
    if len(previous_sample_f1s) != len(sample_f1s):
        return False
    for i in range(0, len(sample_f1s)):
        if previous_sample_f1s[i] != sample_f1s[i]:
            return False
    return True


def bootstrap_wrapper(f1score_results_filename, bootstrapping_results_filename, bootstrapping_samples_no, confidence_interval1, confidence_interval2):
    #if utils.is_file(bootstrapping_results_filename):
    #    return
    model_names, checkpoints, thresholds, f1s, uars , model_paths = read_f1score_results_file(f1score_results_filename)
    sample_f1s_list = []
    with open(bootstrapping_results_filename, 'w') as results_file:
        results_file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}'.\
            format('model', 'checkpoint', 'threshold', 'F1-test', 'UAR-test', 'CI '+ str(confidence_interval1) +' lower', 'CI '+ str(confidence_interval1) +' upper', 'CI '+ str(confidence_interval2) +' lower', 'CI '+ str(confidence_interval2) +' upper', 'model path'))
        results_file.write('\n')
        previous_pId_sample_items = None
        for i in range(0, len(model_names)):
            quantile_lower_1, quantile_upper_1, quantile_lower_2, quantile_upper_2, pId_sample_items, sample_f1s = \
                bootstrap_wrapper_model(model_names[i], checkpoints[i], thresholds[i], model_paths[i], bootstrapping_samples_no, confidence_interval1, confidence_interval2)
            if i > 0:
                assert samples_are_equal(previous_pId_sample_items, pId_sample_items), 'samples are not equal,{0},{1}'.format(','.join(previous_pId_sample_items), ','.join(pId_sample_items))
            sample_f1s_list.append(sample_f1s)    
            results_file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}'.\
                format(model_names[i], checkpoints[i], thresholds[i], f1s[i], uars[i], quantile_lower_1, quantile_upper_1, quantile_lower_2, quantile_upper_2, model_paths[i]))
            results_file.write('\n')
            previous_pId_sample_items = pId_sample_items
        assert samples_are_equal(previous_pId_sample_items, pId_sample_items), 'samples are not equal,{0},{1}'.format(','.join(previous_pId_sample_items), ','.join(pId_sample_items))
    return model_names, f1s, sample_f1s_list

def write_bootstrap_details(bootstrapping_details_filename, f1s, model_names, sample_f1s_list):
    if bootstrapping_details_filename == '' or bootstrapping_details_filename == None:
        return
    with open(bootstrapping_details_filename, 'w') as details_file:
        details_file.write('Models:,')
        details_file.write(','.join(model_names))
        details_file.write('\n')
        details_file.write('F1:,')
        for i in range(0, len(f1s)):
            if i > 0:
                details_file.write(',')
            details_file.write(str(f1s[i]))
        details_file.write('\n')
        samples_no = len(sample_f1s_list[0])
        for j in range(0, samples_no):
            for i in range(0, len(model_names)):
                if i == 0:
                    details_file.write('S {0}:,'.format(j+1))
                else:
                    details_file.write(',')
                details_file.write(str(sample_f1s_list[i][j]))
            details_file.write('\n')
    

def main(args=None):
    model_names, f1s, sample_f1s_list = bootstrap_wrapper(args.f1score_results_filename, args.bootstrapping_results_filename, args.bootstrapping_samples_no, args.confidence_interval1, args.confidence_interval2)
    write_bootstrap_details(args.bootstrapping_details_filename, model_names, f1s, sample_f1s_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add other labels to prob files')
    parser.add_argument('--bootstrapping_samples_no', type=int, default=5000, nargs='?', help='number of samples in bootstrapping')
    parser.add_argument('--f1score_results_filename', type=str, default=r'', nargs='?', help='')
    parser.add_argument('--bootstrapping_results_filename', type=str, default=r'', nargs='?', help='')
    parser.add_argument('--bootstrapping_details_filename', type=str, default=r'', nargs='?', help='')
    parser.add_argument('--confidence_interval1', type=float, default=0.90, nargs='?', help='')
    parser.add_argument('--confidence_interval2', type=float, default=0.80, nargs='?', help='')
    args = parser.parse_args()
    main(args)
