import matplotlib.pyplot as plt
import matplotlib.figure as fig
import csv
import os
import numpy as np
from numpy import genfromtxt

import utils

def show_prob(probs, threshlod, figure_file_fullname, save_only, start = 0, end = 0):
    red = '#ea6262'
    blue = '#3E6182'
    gray = 'gray'
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    if start == 0 and end == 0:
        plt.plot(probs[:,0],probs[:,1], color=red)
        plt.plot(probs[:,0],probs[:,2], color=blue)
        plt.plot(probs[:,0],np.full((probs.shape[0]), threshlod), color=gray)
    else:
        plt.plot(probs[start:end,0],probs[start:end,1], color=red)
        plt.plot(probs[start:end,0],probs[start:end,2], color=blue)
        plt.plot(probs[start:end,0],np.full((end-start), threshlod), color=gray)
    if not save_only:
        plt.show()
    utils.delete_file_if_exists(figure_file_fullname)
    fig = plt.gcf()
    fig.set_size_inches(18,10)
    plt.savefig(figure_file_fullname, bbox_inches='tight')
    plt.close()

def show_prob_parts(probs, file_name, output_dir, threshlod, parts, save_only):
    if parts == 1:
        figure_file_name = utils.get_file_name_without_extension(file_name)+'.png'
        figure_file_fullname = os.path.join(output_dir, figure_file_name)
        show_prob(probs, threshlod, figure_file_fullname, save_only)
    else:    
        frame_end=probs.shape[0]+1
        interval =int(probs.shape[0]/parts)
        range_start=0
        range_end=0
        for i in range(1, parts+1):
            figure_file_name = utils.get_file_name_without_extension(file_name)+ '_' + str(i) + '.png'
            figure_file_fullname = os.path.join(output_dir, figure_file_name)
            range_start =range_end
            if i<parts:
                range_end+=interval
            else:
                range_end=frame_end-1
            show_prob(probs, threshlod, figure_file_fullname, save_only, range_start, range_end)

def show_prob_file(prob_file_fullname, output_dir, threshlod, parts, save_only):
    utils.create_dir_if_required(output_dir)
    file_name = utils.get_file_name_from_path(prob_file_fullname)
    print(file_name)
    probs = genfromtxt(prob_file_fullname, delimiter=',')
    show_prob_parts(probs, file_name, output_dir, threshlod , parts, save_only)

def show_prob_files(prob_dir, output_dir, threshlod, parts, save_only):
    utils.create_dir_if_required(output_dir)
    for prob_file_fullname in utils.get_immediate_files(prob_dir):
        show_prob_file(prob_file_fullname, output_dir, threshlod, parts, save_only)

if __name__ == '__main__':
    #prob_dir = r'\\10.2.224.9\c3140147\run\20191002\20191017\est.d4ks1357d2tl.valid.cl.b128.93.64_std_uni.smo_0.oldInput_\best_checkpoints\185000\prob'
    #threshlod = 0.987

    #prob_dir = r'\\10.2.224.9\c3140147\run\20190906\20190926\cl3_1_nf_92.64_std_uni.smo_0.125\best_checkpoints\95000\prob'
    #threshlod = 0.987

    #prob_dir = r'\\10.2.224.9\c3140147\run\20191002\20191017\est.kyritsis.93.64_std_uni.smo_0.oldInput\best_checkpoints\330000\prob'
    #threshlod = 0.987

    prob_dir = r'\\10.2.224.9\c3140147\run\20191002\20191017\est.d4ks1357d2tl.valid.cl.b128.93.64_std_uni.smo_0.oldInput_\best_checkpoints\185000\prob_test'
    threshlod = 0.987

    output_dir = prob_dir+'_figure' 
        
    show_prob_files(prob_dir, output_dir, threshlod, 1, True)
    #show_prob_file(prob_file_fullname, output_dir, threshlod, 4, True)

