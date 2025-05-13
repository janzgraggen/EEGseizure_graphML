# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-28 -*-
# -*- Last revision: 2025-05-13 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Functions to train the model-*-

# Import libraries
from typing import Iterable
import torch
from torch import Tensor
from torch.optim import Optimizer


def get_criterion(criterion_name: str) -> torch.nn.Module:
    """
    Get the criterion based on the name provided.

    Args:
        criterion_name (str): The name of the criterion.

    Returns:
        torch.nn.Module: The criterion object.
    """
    if criterion_name == "BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss()
    elif criterion_name == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()
    elif criterion_name=="MSELoss":
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Criterion {criterion_name} not recognized.")


def get_optimizer(
    optimizer_name: str, parameters: Iterable[Tensor], learning_rate: float
) -> Optimizer:
    """
    Get the optimizer based on the class name provided.

    Args:
        optimizer_name (str): The name of the optimizer class.
        parameters (iterable): The parameters to optimize.
        learning_rate (float): The learning rate.

    Returns:
        torch.optim.Optimizer: The optimizer object.
    """
    if optimizer_name == "Adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(parameters, lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")
