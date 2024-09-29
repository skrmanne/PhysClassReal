"""The dataloader for AIRFLOW dataset.

Dataset contains in-the-wild infant and adult videos with GT respiration waveforms manually annotated.
Same dataloader can be used to load flow videos if similar naming and directory structure is followed.
"""
import glob
import os
import re

import cv2
import h5py
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader


class AIRFLOWLoader(BaseLoader):
    """The data loader for the ACL dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an AIRFLOW dataloader.
            Args:
                data_path(str): path of a folder which stores raw/flow videos and gt data.
                -----------------
                     AIRFLOW/
                     |   |-- D01/
                     |      |-- **/
                     |          |-- *.mp4
                     |          |-- *.hdf5
                     |          |...
                     |          |-- *.mp4
                     |          |-- *.hdf5
                     |      |...
                     |      |-- **/
                     |          |-- *.mp4
                     |          |-- *.hdf5
                     |          |...
                     |          |-- *.mp4
                     |          |-- *.hdf5
                     |...
                     |   |-- Y01/
                     |      |-- **/
                     |          |-- *.mp4
                     |          |-- *.hdf5
                     |          |...
                     |          |-- *.mp4
                     |          |-- *.hdf5
                     |      |...
                     |      |-- **/
                     |          |-- *.mp4
                     |          |-- *.hdf5
                     |          |...
                     |          |-- *.mp4
                     |          |-- *.hdf5
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path (For AIR dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            dir_len = len(glob.glob(data_dir + os.sep + "*"))  # Supports arbitrary number of videos in each subject.
            for i in range(dir_len):
                subject = os.path.split(data_dir)[-1]
                #dirs.append({"index": int('{0}0{1}'.format(subject, i)),
                dirs.append({"index": '{0}_{1}'.format(subject, i),
                             "path": os.path.join(data_dir, str(i).zfill(3))})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values"""
        #assert begin==0 and end==1, "ACL does not splitting the data due to different number of videos per subject"

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

        video_fn = glob.glob(data_dirs[i]['path']+'/*.mp4')[0]
        label_fn = glob.glob(data_dirs[i]['path']+'/*.hdf5')[0]

        #print(data_dirs[i]['path'], video_fn, label_fn)
        frames = self.read_video(
            #os.path.join(data_dirs[i]['path'],"data.mp4"))
            video_fn)
        bvps = self.read_wave(
            #os.path.join(data_dirs[i]['path'],"data.hdf5"))
            label_fn)
        
        # Better resample using scipy resampling instead of numpy interpolation
        target_length = frames.shape[0]
        #bvps = BaseLoader.resample_air(bvps, target_length)
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
            # resize frame to 96x96 resolution
            frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2), interpolation=cv2.INTER_AREA) 
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0  # TODO: maybe change into avg
            frames.append(frame)
            success, frame = VidObj.read()

        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        f = h5py.File(bvp_file, 'r')
        resp = f['respiration'][:]
        return resp
        #return pulse
