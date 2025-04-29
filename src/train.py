# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-28 -*-
# -*- Last revision: 2025-04-28 by roduit -*-
# -*- python version : 3.11.11 -*-
# -*- Description: Functions to train the model-*-

# Import libraries
import torch

def get_criterion(criterion_name):
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
    else:
        raise ValueError(f"Criterion {criterion_name} not recognized.")
    
def get_optimizer(optimizer_class, parameters, learning_rate):
    """
    Get the optimizer based on the class name provided.
    
    Args:
        optimizer_class (str): The name of the optimizer class.
        model (torch.nn.Module): The model to optimize.
        learning_rate (float): The learning rate.
        
    Returns:
        torch.optim.Optimizer: The optimizer object.
    """
    if optimizer_class == "Adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    elif optimizer_class == "SGD":
        return torch.optim.SGD(parameters, lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_class} not recognized.")