import numpy as np
import pandas as pd
import torch
import glob
import cv2
from evaluation.post_process import *


def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)
    return np.reshape(sort_data.cpu(), (-1))


def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()

    # set adult or infant subject flag
    infant_flag = False
    if config.VALID.DATA.DATASET == 'AIRFLOW' or config.TEST.DATA.DATASET == 'AIRFLOW' or config.VALID.DATA.DATASET == 'AIR' or config.TEST.DATA.DATASET == 'AIR':
        infant_flag = True
        print('Post-processing configs set to infant subjects.')
    for index in predictions.keys():
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "Raw":
            diff_flag_test = False
        elif config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            diff_flag_test = True
        else:
            raise ValueError("Not supported label type in testing!")

        frame_rate = config.TEST.DATA.FS
        """
        # varying frame rate support for AIR dataset:
        if config.TEST.DATA.DATASET == 'AIR':
            sub_folder = str(index).split('_')[0] +'/' + str(int(str(index).split('_')[1])).zfill(3)
            vid_fn = glob.glob('/scratch/manne.sa/data/ACL_23/AIR/test/'+sub_folder+'/*.mp4')[0]
            cap = cv2.VideoCapture(vid_fn)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        """

        gt_hr_fft, pred_hr_fft = calculate_metric_per_video(
            prediction, label, diff_flag=diff_flag_test, fs=frame_rate, hr_method='FFT', 
            path=str(index), infant_flag=infant_flag, mode=config.MODEL.MODE)
        gt_hr_peak, pred_hr_peak = calculate_metric_per_video(
            prediction, label, diff_flag=diff_flag_test, fs=frame_rate, hr_method='Peak', 
            path=str(index), infant_flag=infant_flag, mode=config.MODEL.MODE)
        
        # Remove outliers for now. TODO: Add this back
        #if str(index) == '10106' or str(index) == '10300':
            #print('skipped: ', index)
            #continue
        gt_hr_fft_all.append(gt_hr_fft)
        predict_hr_fft_all.append(pred_hr_fft)
        predict_hr_peak_all.append(pred_hr_peak)
        gt_hr_peak_all.append(gt_hr_peak)
    predict_hr_peak_all = np.array(predict_hr_peak_all)
    predict_hr_fft_all = np.array(predict_hr_fft_all)
    gt_hr_peak_all = np.array(gt_hr_peak_all)
    gt_hr_fft_all = np.array(gt_hr_fft_all)
    for metric in config.TEST.METRICS:
        if metric == "MAE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                print("FFT MAE (FFT Label):{0}".format(MAE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "RMSE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                print("FFT RMSE (FFT Label):{0}".format(RMSE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "MAPE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                print("FFT MAPE (FFT Label):{0}".format(MAPE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "Pearson":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                print("FFT Pearson (FFT Label):{0}".format(Pearson_FFT[0][1]))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        else:
            raise ValueError("Wrong Test Metric Type")
