# -*- coding: utf-8 -*-
# -*- authors : Jan Zgraggen -*-
# -*- date : 2025-04-02 -*-
# -*- Last revision: 2025-06-09 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Functions to load the data-*-

# Import libraries
import numpy as np
from scipy import signal

def time_filtering(x: np.ndarray) -> np.ndarray:
    """Filter signal in the time domain

    Args:
        x (np.ndarray): Input signal to be filtered.
    Returns:
        np.ndarray: Filtered signal.
    """
    bp_filter = signal.butter(4, (0.5, 30), btype="bandpass", output="sos", fs=250)
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

def power_spectral_density(x: np.ndarray) -> np.ndarray:
    """Apply power spectral density and return band energies coefficients.

    Args:
        x (np.ndarray): Input signal of shape [time, channels].

    Returns:
        np.ndarray: band energies coefficients.
    """
    # Set wavelet type and decomposition level
    

    
    freqs, psd_signals = signal.welch(np.asarray(x.T, dtype=np.float32), fs=250)

    # remove frequencies outside the range of interest
    psd_signals = psd_signals[:, np.logical_and(freqs >= 0.5, freqs <= 30)]

    # Log transform for numerical stability
    psd_signals = np.where(np.abs(psd_signals) > 1e-8, np.abs(psd_signals), 1e-8).T

    return psd_signals