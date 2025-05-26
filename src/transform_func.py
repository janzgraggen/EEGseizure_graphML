# -*- coding: utf-8 -*-
# -*- authors : Jan Zgraggen -*-
# -*- date : 2025-04-02 -*-
# -*- Last revision: 2025-05-26 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Functions to load the data-*-

# Import libraries
import numpy as np
from scipy import signal
import mne
from mne.preprocessing import ICA
import mne_icalabel
import numpy as np

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

def ica(x: np.ndarray):

    # Example parameters
    sfreq = 250  # Hz
    n_channels = 19
    n_seconds = 12
    n_samples = sfreq * n_seconds

    data = x.T

    # Standard 10-20 channel names
    standard_19_ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                            'T7', 'C3', 'Cz', 'C4', 'T8',
                            'P7', 'P3', 'Pz', 'P4', 'P8',
                            'O1', 'O2']

    # Create MNE Info object
    info = mne.create_info(ch_names=standard_19_ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)

    # Create RawArray object
    raw = mne.io.RawArray(data, info, verbose=False)

    # Set the standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    # Filter data: bandpass 1–100 Hz and notch at 50 Hz (line noise)
    raw.filter(1., 100., verbose=False)
    raw.notch_filter(50., verbose=False)

    # Apply Common Average Reference (CAR)
    raw.set_eeg_reference('average', projection=False, verbose=False)

    # Fit ICA using extended infomax (picard supports this)
    ica = ICA(method='infomax', fit_params=dict(extended=True))
    ica.fit(raw, verbose=False)

    # Label components using ICLabel
    labels = mne_icalabel.label_components(raw, ica, method='iclabel')

    # Select components to exclude based on labels
    artifact_types = ['eye', 'muscle', 'other']
    ica.exclude = [i for i, label in enumerate(labels['labels']) if label in artifact_types]

    # Apply ICA to remove selected components
    raw_clean = ica.apply(raw.copy(), exclude=[i for i, label in enumerate(labels['labels']) if label in artifact_types], verbose=False)

    # Extract cleaned data as NumPy array (samples, channels)
    clean_data = raw_clean.get_data().T

    return clean_data