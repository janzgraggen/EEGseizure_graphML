# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-28 -*-
# -*- Last revision: 2025-06-09 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Utils functions -*-

# Import libraries
import os
import random
import numpy as np
import torch
import yaml

# Import modules
from models.graph_models import GAT, GCN, LSTMGNN, LSTMGAT
from models.cnn import CNN
from models.fcn import FCN
from models.resnet import ResNet

def set_seed(seed: int = 42):
    """Set the seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(0)
    elif torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set PYTHONHASHSEED environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in cudnn (may slow down your training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_yml(cfg_file: str) -> dict:
    """Read a yaml configuration file.

    Args:
        cfg_file (str): Path to the yaml configuration file.

    Returns:
        dict : Configuration parameters as a dictionary.
    """
    with open(cfg_file, "r") as file:
        cfg = yaml.safe_load(file)
    return cfg


def choose_model(cfg: dict) -> torch.nn.Module:
    """Choose a model based on the model name.

    Args:
        model_name (str): Name of the model to choose.

    Returns:
        torch.nn.Module: The selected model.
    """
    model_name = cfg.get("name")
    model_cfg = cfg.get("config", {})
    if model_name == "CNN":
        return CNN.from_config(model_cfg=model_cfg)

    elif model_name == "FCN":
        return FCN.from_config(model_cfg=model_cfg)
    elif model_name == "ResNet":
        return ResNet.from_config(model_cfg=model_cfg)
    elif model_name == "GAT":
        return GAT.from_config(model_cfg=model_cfg)
    elif model_name == "LSTMGNN":
        return LSTMGNN.from_config(model_cfg=model_cfg)
    elif model_name == "LSTMGAT":
        return LSTMGAT.from_config(model_cfg=model_cfg)
    elif model_name == "GCN":
        return GCN.from_config(model_cfg=model_cfg)
    else:
        raise ValueError(f"Model {model_name} not found.")


def is_mostly_zero_record(eeg, threshold=0.2):
    """Check if the EEG record is mostly zero.
    This function checks if a given EEG record is mostly zero by calculating
    the cumulative sum of the absolute signal across channels and determining
    if the start of the signal exceeds a specified threshold.

    Args:
        eeg (np.ndarray): EEG data with shape (channels, time).
        threshold (float): Threshold for determining if the record is mostly zero.
                           Default is 0.2 (20%).
    Returns:
        bool: True if the record is mostly zero, False otherwise.
    """
    # Sum absolute signal across channels => shape: (time,)
    signal_magnitude = np.sum(np.abs(eeg), axis=1)
    # assert signal_magnitude.shape == (3000,), f"Expected shape (3000,), got {signal_magnitude.shape}"

    length = len(signal_magnitude)
    zeros = (signal_magnitude == 0).sum()

    return (zeros / length) > threshold
