"""The post processing files for caluclating heart rate using FFT or peak detection.
The file also  includes helper funcs such as detrend, mag2db etc.
"""

import numpy as np
import scipy
import scipy.io
from scipy.signal import butter, detrend
from scipy.sparse import spdiags
import os
import h5py
import torch

def get_signal(scores):
    # run through scores and add a constant value if score==1 else subtract
    signal, eps = torch.zeros_like(scores), 0.1
    for b in range(scores.shape[0]):
        for t in range(1, scores.shape[1]):
            signal[b, t] = signal[b, t-1] + eps if scores[b, t] >= 0.5 else signal[b, t-1] - eps
        signal[b,:] = (signal[b,:]-torch.min(signal[b,:]))/(torch.max(signal[b,:])-torch.min(signal[b,:])) # normalize
    
    return signal

    """
    # conversion in numpy
    signal, eps = np.zeros_like(scores), 0.1
    for t in range(1, scores.shape[0]):
        signal[t] = signal[t-1] + eps if scores[t] >= 0.5 else signal[t-1] - eps
    
    #normalize the signal with min max
    print("min, max:", np.min(signal), np.max(signal))
    signal = (signal-np.min(signal))/(np.max(signal)-np.min(signal))
    return signal"""


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal


def mag2db(mag):
    """Convert magnitude to db."""
    return 20. * np.log10(mag)


def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak


def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, 
hr_method='FFT', path=None, infant_flag=False, mode='regression'):
    if path=='Y03_0':
        return 0, 0
    """Calculate video-level HR"""
    """
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        if len(predictions) < 100:
            predictions = _detrend(np.cumsum(predictions), 50)
            labels = _detrend(np.cumsum(labels), 50)
        else:
            predictions = _detrend(np.cumsum(predictions), 100)
            labels = _detrend(np.cumsum(labels), 100)
    else:
        pass
        #predictions = _detrend(predictions, 100)
        #labels = _detrend(labels, 100)
    """

    if mode == 'classification':
        #print("Post processing conversion to signal", predictions.shape)
        #predictions = _detrend(predictions, 100)
        predictions = detrend(predictions)

    if use_bandpass:

        """
        SCAMPS and other adult pulse rate configs.
        # bandpass filter between [0.75, 2.5] Hz equals [45, 150] beats per min
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        """

        if infant_flag:
            # ACL dataset has breath rate for normal infants in the range [30, 50] breaths per min.
            # bandpass filter between [0.1, 1.0] = [6, 60]
            #lo, hi  = 0.1, 1.0, 2.0
            lo, hi = 0.3, 0.8
        else:
            # bandpass filter between [0.08, 0.5] Hz equals [5, 30] breaths per min
            # SCAMPS dataset has breath rate drawn normally from [8, 24] bpm range.
            # COHFACE dataset breath rate distribution - same as SCAMPS.
            #lo, hi = 0.08, 0.5
            lo, hi = 0.75, 2.5

        [b, a] = butter(1, [lo / fs * 2, hi / fs * 2], btype='bandpass')    # SCAMPS and COHFACE dataset bandpass filter bw.

        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))
    if hr_method == 'FFT':
        # Original low, high frequency cutoffs for Adult Heart/Respiration rate estimation.
        #hr_pred = _calculate_fft_hr(predictions, fs=fs, low_pass=0.08, high_pass=0.5)
        #hr_label = _calculate_fft_hr(labels, fs=fs, low_pass=0.08, high_pass=0.5)
        
        # Changing the frequency for infant respiration rate estimation based on subject
        hr_pred = _calculate_fft_hr(predictions, fs=fs, low_pass=lo, high_pass=hi)
        hr_label = _calculate_fft_hr(labels, fs=fs, low_pass=lo, high_pass=hi)
    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')

    if path and hr_method=='FFT':
        # Dump the predicted and label waveforms used for metric calculation.
        hf = h5py.File(os.path.join('/scratch/manne.sa/data/ACL_23/ACL/post_output', path+'_output.hdf5'), 'w')
        hf.create_dataset('respiration', data=predictions)
        hf.close()
    
        hf = h5py.File(os.path.join('/scratch/manne.sa/data/ACL_23/ACL/post_output', path+'_label.hdf5'), 'w')
        hf.create_dataset('respiration', data=labels)
        hf.close()

    if hr_method=='FFT':
        print("Respiration rate for video: {0} - output: {1}; label: {2}; fps: {3}".format(path, hr_pred, hr_label, fs))
    return hr_label, hr_pred

