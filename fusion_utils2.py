"""Fusion utility classes and functions"""

import argparse
import csv
import glob
import numpy as np
import os
import operator
import utils
from scipy import signal

CSV_SUFFIX = '*.csv'

class VideoDto(object):
    PId_index = 0
    Frame_index = 1
    Label_index = 2
    Prob_index = 3
    #__init__
    def __init__(self, pIds=None, frames=None, labels=None, probs=None, phase_number = None):
        self.pIds = pIds if pIds is not None else []
        self.frames = frames if frames is not None else []
        self.probs = probs if probs is not None else []
        self.labels = labels if labels is not None else []
        self.pId = 0
        self.phase_number = phase_number

    # read_from_file:
    def read_from_file(self, filenamepath):
        self.pIds = []; self.frames = []; self.probs =[]; self.labels =[]
        # this is becuase the video file sample rate in phase 2 is 7.5 Hz
        frequency_times_2 = 15
        frame_org = 0
        with open(filenamepath) as prob_file:
            for row in csv.reader(prob_file, delimiter = ','):
                pId = row[VideoDto.PId_index]
                if pId.endswith('_video_140p'):
                    pId = pId.replace('_video_140p','')
                if utils.IfStringRepresentsFloat(pId) or utils.IfStringRepresentsInt(pId) or isinstance(pId, float) or isinstance(pId, int):
                   pId = str(int(float(pId))) + '_' + str(self.phase_number)
                frame = int(float(row[VideoDto.Frame_index]))
                if self.phase_number == 2:
                    frame_org = frame if frame_org==0 else frame_org+1 
                    frame += (frame_org-1)//frequency_times_2
                label = int(float(row[VideoDto.Label_index]))
                prob = float(row[VideoDto.Prob_index])
                self.pIds.append(pId)
                self.frames.append(frame)
                self.labels.append(label)
                self.probs.append(prob)
                # this is becuase the video file sample rate in phase 2 is 7.5 Hz
                if self.phase_number == 2 and frame_org%frequency_times_2 == 0:
                    self.pIds.append(pId)
                    self.frames.append(frame+1)
                    self.labels.append(label)
                    self.probs.append(prob)
        self.pId = self.pIds[0]

class ImuDto(object):
    PId_index = 0
    Frame_index = 1
    Label_index = 2
    Prob_index = 3
    Label1_index = 4
    Label2_index = 5
    Label3_index = 6
    Label4_index = 7

    #__init__
    def __init__(self, pIds=None, frames=None, labels=None, probs=None, labels1=None, labels2=None, labels3=None, labels4=None):
        self.pIds = pIds
        self.frames = frames if frames is not None else []
        self.probs = probs if probs is not None else []
        self.labels = labels if labels is not None else []
        self.labels1 = labels1
        self.labels2 = labels2
        self.labels3 = labels3
        self.labels4 = labels4
        self.contain_extra = False
        self.pId = 0
    
    def read_from_file(self, filenamepath, retrieve_extra=False):
        self.frames = []; self.probs =[]; self.labels =[]
        self.contain_extra = retrieve_extra
        if self.contain_extra:
            self.pIds = []; self.labels1 = []; self.labels2 = []; self.labels3 = []; self.labels4 = []
        with open(filenamepath) as prob_file:
            for row in csv.reader(prob_file, delimiter=','):
                self.frames.append(int(row[ImuDto.Frame_index]))
                self.labels.append(row[ImuDto.Label_index])
                self.probs.append(row[ImuDto.Prob_index])
                if retrieve_extra:
                    self.pIds.append(row[ImuDto.PId_index]) #this should change to 3 later
                    self.labels1.append(row[ImuDto.Label1_index])
                    self.labels2.append(row[ImuDto.Label2_index])
                    self.labels3.append(row[ImuDto.Label3_index])
                    self.labels4.append(row[ImuDto.Label4_index])
        if self.pIds is not None and len(self.pIds) > 0:
            self.pId = self.pIds[0]
        else:
            self.pId = utils.get_file_name_from_path(filenamepath, True)

class FusionDto(object):
    def __init__(self, vid: VideoDto = None, imu: ImuDto = None, vid_threshold = 0, imu_threshold = 0, phase_number = None):
        self.vid = vid
        self.imu = imu
        self.vid_threshold = vid_threshold
        self.imu_threshold = imu_threshold
        self.phase_number = phase_number
    
    def __sync_vid_imu(self):
        frames_def = self.imu.frames[0] - self.vid.frames[0]
        if frames_def >= 0: # e.g., 121 - 120
            new_vid = VideoDto(self.vid.pIds[frames_def:], self.vid.frames[frames_def:], self.vid.labels[frames_def:], self.vid.probs[frames_def:], self.vid.phase_number)
            new_imu = ImuDto(self.imu.pIds, self.imu.frames, self.imu.labels, self.imu.probs, self.imu.labels1, self.imu.labels2, self.imu.labels3, self.imu.labels4)
        else:
            frames_def *= -1# e.g., 9 - 15
            new_vid = VideoDto(self.vid.pIds, self.vid.frames, self.vid.labels, self.vid.probs, self.phase_number)
            new_imu = ImuDto(None, self.imu.frames[frames_def:], self.imu.labels[frames_def:], self.imu.probs[frames_def:])
            if self.imu.contain_extra:
                new_imu.pIds = self.imu.pIds[frames_def:]
                new_imu.labels1 = self.imu.labels1[frames_def:] 
                new_imu.labels2 = self.imu.labels2[frames_def:] 
                new_imu.labels3 = self.imu.labels3[frames_def:]
                new_imu.labels4 = self.imu.labels4[frames_def:]
        new_vid.pId = self.vid.pId
        new_imu.pId = self.imu.pId
        new_imu.contain_extra = self.imu.contain_extra
        self.vid = new_vid
        self.imu = new_imu

    def write_into_file(self, filenamepath: str):
        assert self.vid.pId == self.imu.pId , 'video pId {0} and imu pId {1} are not the same'.format(self.vid.pId, self.imu.pId)
        if self.imu.pIds == None or self.imu.pIds == []:
            self.imu.pIds = self.vid.pIds
        else:
            assert self.vid.pIds[0] == self.imu.pIds[0] , 'video pId {0} and imu pId {1} are not the same'.format(self.vid.pIds[0], self.imu.pIds[0])
            
        with open(filenamepath, 'w') as fusion_file:
            length = len(self.vid.frames) if len(self.vid.frames) < len(self.imu.frames) else len(self.imu.frames) 
            if(self.imu.contain_extra):
                for i in range(0, length):
                    fusion_file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}'.format(self.vid.pIds[i], self.vid.frames[i], self.vid.labels[i], self.vid.probs[i], self.vid_threshold, self.imu.pIds[i], self.imu.frames[i], self.imu.labels[i], self.imu.probs[i], self.imu_threshold, self.imu.labels1[i], self.imu.labels2[i], self.imu.labels3[i], self.imu.labels4[i]))
                    fusion_file.write('\n')
            else:
                for i in range(0, length):
                    fusion_file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}'.format(self.vid.pIds[i], self.vid.frames[i], self.vid.labels[i], self.vid.probs[i], self.vid_threshold, self.imu.pIds[i], self.imu.frames[i], self.imu.labels[i], self.imu.probs[i], self.imu_threshold))
                    fusion_file.write('\n')

    def merge(self, vid_filenamepath, imu_filenamepath, retrieve_extra_imu = False):
        self.vid = VideoDto()
        self.vid.phase_number = self.phase_number
        self.vid.read_from_file(vid_filenamepath)
        self.imu = ImuDto()
        self.imu.read_from_file(imu_filenamepath, retrieve_extra_imu)
        self.__sync_vid_imu()

    def merge_vid_imu_prob_files(vid_prob_dir, imu_prob_dir, vid_imu_prob_merge_dir, filename_prefix = '', vid_threshold = 0, imu_threshold = 0, phase_number = None, overwrite = False, retrieve_extra_imu = False):
        utils.create_dir_if_required(vid_imu_prob_merge_dir)
        vid_prob_filenames = glob.glob(os.path.join(vid_prob_dir, CSV_SUFFIX))
        for vid_prob_filename in vid_prob_filenames:
            vid_prob_filename_withoutpath = utils.get_file_name_from_path(vid_prob_filename)
            imu_prob_filename = os.path.join(imu_prob_dir, vid_prob_filename_withoutpath)
            assert utils.is_file(imu_prob_filename),"file {0} does not exist".format(imu_prob_filename)
            vid_prob_filename_withoutpath = filename_prefix + vid_prob_filename_withoutpath
            vid_imu_prob_filename = os.path.join(vid_imu_prob_merge_dir, vid_prob_filename_withoutpath)
            if not overwrite and utils.is_file(vid_imu_prob_filename):
                continue
            vid_imu = FusionDto()
            vid_imu.vid_threshold = vid_threshold
            vid_imu.phase_number = phase_number
            vid_imu.imu_threshold = imu_threshold
            vid_imu.merge(vid_prob_filename, imu_prob_filename, retrieve_extra_imu)
            vid_imu.write_into_file(vid_imu_prob_filename)


