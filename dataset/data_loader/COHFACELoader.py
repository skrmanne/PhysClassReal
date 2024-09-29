"""The dataloader for COHFACE datasets.

Details for the COHFACE Dataset see https://www.idiap.ch/en/dataset/cohface
If you use this dataset, please cite the following publication:
Guillaume Heusch, André Anjos, Sébastien Marcel, “A reproducible study on remote heart rate measurement”, arXiv, 2016.
http://publications.idiap.ch/index.php/publications/show/3688
"""
import glob
import os
import re

import cv2
import h5py
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
import scipy
from scipy.signal import butter, detrend

def bandpass(signal, lo, hi, fs):
    [b, a] = butter(1, [lo / fs * 2, hi / fs * 2], btype='bandpass')
    signal = scipy.signal.filtfilt(b, a, np.double(signal))

    return signal

class COHFACELoader(BaseLoader):
    """The data loader for the COHFACE dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an COHFACE dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- 1/
                     |      |-- 0/
                     |          |-- data.avi
                     |          |-- data.hdf5
                     |      |...
                     |      |-- 3/
                     |          |-- data.avi
                     |          |-- data.hdf5
                     |...
                     |   |-- n/
                     |      |-- 0/
                     |          |-- data.avi
                     |          |-- data.hdf5
                     |      |...
                     |      |-- 3/
                     |          |-- data.avi
                     |          |-- data.hdf5
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.mode = config_data.PREPROCESS.MODE
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For COHFACE dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            for i in range(4):
                subject = os.path.split(data_dir)[-1]
                dirs.append({"index": int('{0}0{1}'.format(subject, i)),
                             "path": os.path.join(data_dir, str(i))})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values"""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs
        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []
        for i in choose_range:
            data_dirs_new.append(data_dirs[i])
        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        frames = self.read_video(
            os.path.join(data_dirs[i]['path'],"data.avi"))
        bvps = self.read_wave(
            os.path.join(data_dirs[i]['path'],"data.hdf5"))
        
        # Resample RR at video FPS
        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    """
    def preprocess_dataset(self, data_dirs, config_preprocess):
        #Preprocesses the raw data.
        file_num = len(data_dirs)
        for i in range(file_num):
            frames = self.read_video(
                os.path.join(
                    data_dirs[i]["path"],
                    "data.avi"))
            bvps = self.read_wave(
                os.path.join(
                    data_dirs[i]["path"],
                    "data.hdf5"))
            target_length = frames.shape[0]
            bvps = BaseLoader.resample_ppg(bvps, target_length)
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            self.preprocessed_data_len += self.save(frames_clips, bvps_clips, data_dirs[i]["index"])
    """

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while (success):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0  # TODO: maybe change into avg
            frames.append(frame)
            success, frame = VidObj.read()

        return np.asarray(frames)

    @staticmethod
    def get_classfication_labels(signal, lo, hi, fs):

        # convert the signals to classification labels based on this rule:
        # if signal[i] > signal[i-1]: label=1
        # else: label = 0

        # filter signals over their bandwidth
        signal = bandpass(signal, lo, hi, fs)

        classification_labels = np.array([0]*len(signal))
        for i in range(1, len(classification_labels)):
            if signal[i] > signal[i-1]:
                classification_labels[i] = 1.0
        
        return classification_labels

    def read_wave(self, bvp_file):
        """Reads a bvp signal file."""
        f = h5py.File(bvp_file, 'r')
        pulse = f["pulse"][:]
        resp = f['respiration'][:]

        if self.mode == "classification":
            resp = COHFACELoader.get_classfication_labels(resp, 0.08, 0.5, 256)
            pulse = COHFACELoader.get_classfication_labels(pulse, 0.75, 2.5, 256)  # 256 - readout frequency of COHFACE BVP signal

        #return resp
        return pulse   # for pulse rate estimation
