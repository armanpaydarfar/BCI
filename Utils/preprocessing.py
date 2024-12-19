from scipy.signal import butter, filtfilt
import numpy as np

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def apply_car_filter(data):
    avg = np.mean(data, axis=0)
    return data - avg

