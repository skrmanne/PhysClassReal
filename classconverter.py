# Check the dataset pre/post-processing pipeline
import os, sys
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import spdiags
from scipy.signal import detrend
from scipy.signal import butter

def bandpass(signal, lo, hi, fs):
    [b, a] = butter(1, [lo / fs * 2, hi / fs * 2], btype='bandpass')
    signal = scipy.signal.filtfilt(b, a, np.double(signal))

    return signal

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

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

def standardized_data(data):
    """Z-score standardization for video data."""
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data

def get_signal(scores):
    # run through scores and add a constant value if score==1 else subtract
    signal, eps = torch.zeros_like(scores), 0.1
    for b in range(scores.shape[0]):
        for t in range(1, scores.shape[1]):
            signal[b, t] = signal[b, t-1] + eps if scores[b, t] >= 0.5 else signal[b, t-1] - eps
        signal[b,:] = (signal[b,:]-torch.min(signal[b,:]))/(torch.max(signal[b,:])-torch.min(signal[b,:])) # normalize
    
    return signal

def get_classfication_labels(signal):

    # convert the signals to classification labels based on this rule:
    # if signal[i] > signal[i-1]: label=1
    # else: label = 0
    classification_labels = np.array([0]*len(signal))
    for i in range(1, len(classification_labels)):
        if signal[i] > signal[i-1]:
            classification_labels[i] = 1.0
    
    return classification_labels

def read_somedata():
    # create a torch tensor of batchsize 10 and sequence length 180
    resp_tensor, pulse_tensor = torch.zeros(10, 1800), torch.zeros(10, 1800)

    for folder in range(1, 11):
        filename = "/scratch/manne.sa/data/ACL_23/COHFACE/cohface/" + str(folder) + "/0/data.hdf5"
        f = h5py.File(filename, 'r')
        pulse = f["pulse"][100:1900]
        resp = f['respiration'][100:1900]
        
        resp = standardized_data(resp)
        pulse = standardized_data(pulse)

        # convert to torch tensor
        pulse_tensor[folder-1, :] = torch.tensor(pulse).float()
        resp_tensor[folder-1, :] = torch.tensor(resp).float()

    return resp_tensor, pulse_tensor

def read_and_convert_somedata():
    # create a torch tensor of batchsize 10 and sequence length 180
    resp_tensor, pulse_tensor = torch.zeros(10, 1800), torch.zeros(10, 1800)

    for folder in range(1, 11):
        filename = "/scratch/manne.sa/data/ACL_23/COHFACE/cohface/" + str(folder) + "/0/data.hdf5"
        f = h5py.File(filename, 'r')
        pulse = f["pulse"][100:1900]
        resp = f['respiration'][100:1900]
        
        # filter signals over their bandwidth
        resp = bandpass(resp, 0.08, 0.5, 256)
        pulse = bandpass(pulse, 0.75, 2.5, 256)

        resp = get_classfication_labels(resp)
        pulse = get_classfication_labels(pulse)

        # convert to torch tensor
        pulse_tensor[folder-1, :] = torch.tensor(pulse).float()
        resp_tensor[folder-1, :] = torch.tensor(resp).float()

    return get_signal(resp_tensor), get_signal(pulse_tensor)

def visualize_data():
    resp_tensor, pulse_tensor = read_somedata()
    for i in range(0, 10):
        resp, pulse = resp_tensor[i, :], pulse_tensor[i, :]
        # convert to numpy for visualization
        resp, pulse = resp.numpy(), pulse.numpy()

        # filter signals over their bandwidth
        resp = bandpass(resp, 0.08, 0.5, 256)
        pulse = bandpass(pulse, 0.75, 2.5, 256)

        # get frequency from both singals
        #resp_rate = _calculate_fft_hr(resp, fs=256, low_pass=0.08, high_pass=0.5)
        #pulse_rate = _calculate_fft_hr(pulse, fs=256, low_pass=0.75, high_pass=2.5)
        resp_rate = _calculate_peak_hr(resp, fs=256)
        pulse_rate = _calculate_peak_hr(pulse, fs=256)

        # create a plot and draw resp and pulse one below another
        # clear the plot
        plt.clf()
        plt.plot(resp)
        plt.plot(pulse)

        # add labels with resp_rate and pulse_rate
        resp_label = "Respiration Rate: " + str(resp_rate)
        pulse_label = "Pulse Rate: " + str(pulse_rate)

        plt.xlabel("Time")
        plt.ylabel("Signal")
        plt.title("Respiration and Pulse Signal")
        plt.legend([resp_label, pulse_label])
        plt.savefig("fig_" + str(i) + ".png")

def visualize_converted_data():
    resp_tensor, pulse_tensor = read_and_convert_somedata()
    for i in range(0, 10):
        resp, pulse = resp_tensor[i, :], pulse_tensor[i, :]
        resp, pulse = resp.numpy(), pulse.numpy()

        # detrend
        #resp, pulse = _detrend(resp, 100), _detrend(pulse, 100)
        resp, pulse = detrend(resp), detrend(pulse)

        # get frequency from both singals
        #resp_rate = _calculate_fft_hr(resp, fs=256, low_pass=0.08, high_pass=0.5)
        #pulse_rate = _calculate_fft_hr(pulse, fs=256, low_pass=0.75, high_pass=2.5)
        resp_rate = _calculate_peak_hr(resp, fs=256)
        pulse_rate = _calculate_peak_hr(pulse, fs=256)

        # add labels with resp_rate and pulse_rate
        resp_label = "Respiration Rate: " + str(resp_rate)
        pulse_label = "Pulse Rate: " + str(pulse_rate)

        # create a plot and draw resp and pulse one below another
        # clear the plot
        plt.clf()
        plt.plot(resp)
        plt.plot(pulse)

        # add labels
        plt.xlabel("Time")
        plt.ylabel("Signal")
        plt.title("Respiration and Pulse Signal")
        plt.legend([resp_label, pulse_label])
        plt.savefig("fig_" + str(i) + "_c.png")

if __name__ == "__main__":
    visualize_data()
    visualize_converted_data()

    """
    TODO 1: Check the frequency of the signal from both variants.
    TODO 2: Check for respiration and pulse signal differences compare
             if higher frequency signal has any different behavior.
    TODO 3:
    """