# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-28 -*-
# -*- Last revision: 2025-05-26 by Caspar -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Utils functions -*-

# Import libraries
import os
import random
import numpy as np
import torch
import yaml

# Import modules
from models.graph_models import GAT, GCN
from models.cnn import CNN
from models.fcn import FCN


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
    elif model_name == "GAT":
        return GAT.from_config(model_cfg=model_cfg)
    
    elif model_name == "GCN":
        return GCN.from_config(model_cfg=model_cfg)
    
    else:
        raise ValueError(f"Model {model_name} not found.")