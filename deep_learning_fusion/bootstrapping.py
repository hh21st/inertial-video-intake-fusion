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

CSV_SUFFIX = '*.csv'


def bootstrap_step1(labels, probs, threshold, mindist):
    probs = np.array([float(f) for f in probs])
    labels = np.array([int(i) for i in labels])
    detections = eval.max_search(probs, threshold, mindist)
    tp, fn, fp_1, fp_2, _, _, _ = eval.eval_stage_2(detections, labels)
    return tp, fn, fp_1, fp_2


def read_bootstrap_step1_file(bootstrapping_step1_filename):
    pIds, tps, fns, fp_1s, fp_2s = [], [], [], [], []
    with open(bootstrapping_step1_filename) as step1_file:
        reader = csv.reader(step1_file, delimiter=',')
        next(reader)
        for row in reader:
            pIds.append(row[0])
            tps.append(int(row[1]))
            fns.append(int(row[2]))
            fp_1s.append(int(row[3]))
            fp_2s.append(int(row[4]))
    return np.array(pIds), np.array(tps), np.array(fns), np.array(fp_1s), np.array(fp_2s)


def bootstrap_step2(bootstrapping_step1_filename, samples_no):
    pIds, tps, fns, fp_1s, fp_2s = read_bootstrap_step1_file(bootstrapping_step1_filename)
    sample_items_no = len(pIds)
    sample_f1s = []
    for i in range(samples_no):
        np.random.seed(i)
        pId_sample_items = np.random.choice(pIds, size = sample_items_no)
        sample_items_idxs = [np.where(pIds == pId)[0] for pId in pId_sample_items]
        sample_tps = np.sum([tps[idx] for idx in sample_items_idxs])
        sample_fns = np.sum([fns[idx] for idx in sample_items_idxs])
        sample_fp_1s = np.sum([fp_1s[idx] for idx in sample_items_idxs])
        sample_fp_2s = np.sum([fp_2s[idx] for idx in sample_items_idxs])
        sample_prec = utils.calc_precision(sample_tps, sample_fp_1s + sample_fp_2s)
        sample_rec = utils.calc_recall(sample_tps, sample_fns)
        sample_f1 = utils.calc_f1(sample_prec, sample_rec)
        sample_f1s.append(sample_f1)
    sample_f1s = np.array(sample_f1s)
    return pId_sample_items, sample_f1s


def bootstrap_step3(sample_f1s, confidence_interval, rounding_digits = None):
    sample_f1s = sorted(sample_f1s)
    confidence_interval_lower = (1-confidence_interval)/2
    confidence_interval_upper = (confidence_interval+1)/2
    quantile_lower = np.quantile(sample_f1s, confidence_interval_lower)
    quantile_upper = np.quantile(sample_f1s, confidence_interval_upper)
    if rounding_digits != None:
        quantile_lower = round(quantile_lower, rounding_digits)
        quantile_upper = round(quantile_upper, rounding_digits)
    return quantile_lower, quantile_upper


def bootstrap_step1_files(prob_dir, bootstrapping_step1_filename, threshold, mindist):
    if utils.is_file(bootstrapping_step1_filename):
        return
    prob_filenames = glob.glob(os.path.join(prob_dir, CSV_SUFFIX))
    with open(bootstrapping_step1_filename, 'w') as results_file:
        results_file.write('{0},{1},{2},{3},{4}'.format('Id', 'TP', 'FN', 'FP1', 'FP2'))
        results_file.write('\n')
        for prob_filename in prob_filenames:
            _, _, labels, probs, _, _, _, _ = fusion_utils.read_imu_prob_file(prob_filename, False)
            pId = utils.get_file_name_from_path(prob_filename, True)
            tp, fn, fp_1, fp_2 = bootstrap_step1(labels, probs, threshold, mindist)
            results_file.write('{0},{1},{2},{3},{4}'.format(pId, tp, fn, fp_1, fp_2))
            results_file.write('\n')
            print('{} is done!'.format(pId))


def bootstrap(prob_dir, bootstrapping_step1_filename, threshold, mindist, bootstrapping_samples_no, confidence_interval1, confidence_interval2):
    bootstrap_step1_files(prob_dir, bootstrapping_step1_filename, threshold, mindist)
    pId_sample_items, sample_f1s = bootstrap_step2(bootstrapping_step1_filename, bootstrapping_samples_no)
    quantile_lower_1, quantile_upper_1 = bootstrap_step3(sample_f1s, confidence_interval1)
    quantile_lower_2, quantile_upper_2 = bootstrap_step3(sample_f1s, confidence_interval2)
    return quantile_lower_1, quantile_upper_1, quantile_lower_2, quantile_upper_2, pId_sample_items, sample_f1s


#def main(args=None):
    #quantile_lower_1, quantile_upper_1, quantile_lower_2, quantile_upper_2 = \
    #    bootstrap(args.prob_dir, args.bootstrapping_step1_filename, args.threshold, args.mindist, args.bootstrapping_samples_no, ???, ???)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add other labels to prob files')
    parser.add_argument('--prob_dir', type=str, default=r'C:\H\OneDrive - The University Of Newcastle\H\PhD\ORIBA\Fusion\Phase1\est.d4ks1357d2tl.valid.cl.b256.93.64_std_uni_no_smo.fixed.25000\prob', nargs='?', help='')
    parser.add_argument('--mindist', type=int, default=128, nargs='?', help='mindist')
    parser.add_argument('--threshold', type=float, default=0.978, nargs='?', help='imu best threshold')
    parser.add_argument('--bootstrapping_samples_no', type=int, default=5000, nargs='?', help='number of samples in bootstrapping')
    parser.add_argument('--bootstrapping_step1_filename', type=str, default=r'C:\H\OneDrive - The University Of Newcastle\H\PhD\ORIBA\Phase1\Bootstrapping\bootstrapping_step1_results.csv', nargs='?', help='')

    args = parser.parse_args()
    main(args)
