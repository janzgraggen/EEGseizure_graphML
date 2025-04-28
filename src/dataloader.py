# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-28 -*-
# -*- Last revision: 2025-04-28 by roduit -*-
# -*- python version : 3.11.11 -*-
# -*- Description: Functions to load the project-*-

# Import librairies
import pandas as pd
from seiz_eeg.dataset import EEGDataset
from torch.utils.data import DataLoader
import os
from transform_func import *

# Import modules
import constants

def load_data(
        cfg: dict,
    ) -> DataLoader:
    """ Load the data from the path and return a DataLoader.

    Args:
        cfg (dict): Configuration dictionary containing the informations about the data

    Returns:
        DataLoader: _description_
    """
    # Read clips
    path = cfg.get("data_pth", None)
    if path is None:
        raise ValueError("No data path provided in the configuration file.")
    clips_path = os.path.join(path, "segments.parquet")
    clips = pd.read_parquet(clips_path)

    # Get transform
    tfm_name = cfg.get("tfm", None)
    tfm = get_transform(tfm_name=tfm_name)

    # Get additional parameters
    batch_size = cfg.get("batch_size", constants.BATCH_SIZE)
    shuffle = cfg.get("shuffle", True)

    # Create dataset
    dataset = EEGDataset(
        clips,
        signals_root=path,
        signal_transform=tfm,
        prefetch=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader

def get_transform(tfm_name:str):
    """Get the transform function based on the name provided.

    Args:
        tfm_name (str): Name of the transform function to be used.

    Returns:
        function: The transform function corresponding to the name provided.
    """
    if tfm_name == 'fft':
        return fft_filtering
    else:
        return None