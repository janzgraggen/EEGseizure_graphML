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
from tqdm import tqdm

# Import modules
import constants
from transform_func import *

def parse_datasets(cfg: dict):
    """Parse the datasets from the configuration file.

    Args:
        cfg (dict): Configuration dictionary containing the informations about the data
    """

    num_datasets = len(cfg)
    if num_datasets == 0:
        raise ValueError("No datasets provided in the configuration file.")
    
    for i in tqdm(range(num_datasets), desc="Loading datasets", unit="dataset"):
        dataset = cfg[i]
        set_type = dataset.get("set", None)
        if set_type == 'train':
            loader_train = load_data(cfg=dataset)
        elif set_type == 'val':
            loader_val = load_data(cfg=dataset)
        elif set_type == 'test':
            loader_test = load_data(cfg=dataset)
        else:
            raise ValueError(f"Unknown dataset type: {set_type}")
    return loader_train, loader_val, loader_test

        



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
    path = cfg.get("path", None)
    if path is None:
        raise ValueError("No data path provided in the configuration file.")
    clips_path = os.path.join(path, "segments.parquet")
    clips = pd.read_parquet(clips_path)

    split = cfg.get("split", None)

    if split is not None:
        start = int(split[0] * len(clips))
        end = int(split[1] * len(clips))
        clips = clips.iloc[start:end]

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