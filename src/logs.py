# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-02 -*-
# -*- Last revision: 2025-05-06 by roduit -*-
# -*- python version : 3.11.11 -*-
# -*- Description: Functions to log parameters on Mlflow-*-

# Import libraries
import mlflow
from torchinfo import summary
from torch_geometric.nn import summary as pyg_summary
from models.graph_models import GAT, GCN
from models.cnn_base import CnnBase
import tempfile
from pathlib import Path

# Import modules
import constants


def log_cfg(cfg: dict):
    """Function used to log the configuration file on mlflow.

    Args:
        cfg (dict): Configuration file.
    """

    model_name = cfg.get("name", None)
    n_epochs = cfg.get("num_epochs", None)
    optimizer = cfg.get("optimizer", None)
    use_scheduler = cfg.get("use_scheduler", None)
    learning_rate = cfg.get("learning_rate", None)

    # Log parameters
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("n_epochs", n_epochs)
    mlflow.log_param("optimizer", optimizer)
    mlflow.log_param("use_scheduler", use_scheduler)
    mlflow.log_param("learning_rate", learning_rate)

def log_model_summary(model, data_sample):
    """Function used to log the model summary on mlflow.

    Args:
        model (torch.nn.Module): Model to log.
        input_size (tuple): Input size of the model.
    """
    GRAPH_MODELS = [GAT, GCN]
    CNN_MODELS = [CnnBase]
    if type(model) in GRAPH_MODELS:
        model_summary = pyg_summary(model, data_sample[0].to(constants.DEVICE))
    elif type(model) in CNN_MODELS:
        model_summary = summary(model)
    else:
        return
    
    print(model_summary)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir, "model_summary.txt")
        path.write_text(str(summary(model)))
        mlflow.log_artifact(path)