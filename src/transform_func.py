# -*- coding: utf-8 -*-
# -*- authors : Jan Zgraggen -*-
# -*- date : 2025-04-02 -*-
# -*- Last revision: 2025-05-13 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Functions to load the data-*-

# Import libraries
import numpy as np
from scipy import signal

bp_filter = signal.butter(4, (0.5, 30), btype="bandpass", output="sos", fs=250)


def time_filtering(x: np.ndarray) -> np.ndarray:
    """Filter signal in the time domain

    Args:
        x (np.ndarray): Input signal to be filtered.
    Returns:
        np.ndarray: Filtered signal.
    """
    return signal.sosfiltfilt(bp_filter, x, axis=0).copy()


def fft_filtering(x: np.ndarray) -> np.ndarray:
    """Compute FFT and only keep the frequencies between 0.5 and 30Hz

    Args:
        x (np.ndarray): Input signal to be filtered.
    Returns:
        np.ndarray: Filtered signal.
    """
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))

    win_len = x.shape[0]
    # Only frequencies b/w 0.5 and 30Hz
    return x[int(0.5 * win_len // 250) : 30 * win_len // 250]

def clean_input(x: np.ndarray) -> np.ndarray:
    """Apply processing to the input
    
    Args:
        x (np.ndarray): Input signal to be filtered.
    Returns:
        np.ndarray: Processed signal.
    """
    # Apply bandpass filter
    x = time_filtering(x)

    # Normalize: per-sample, per-channel standardization
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    x = (x - mean) / (std + 1e-6)

    return x