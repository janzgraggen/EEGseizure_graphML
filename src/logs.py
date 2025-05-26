# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-02 -*-
# -*- Last revision: 2025-05-26 by Caspar -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Functions to log parameters on Mlflow-*-

# Import libraries
import mlflow
from torchinfo import summary
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
    n_epochs = cfg.get("n_epochs", None)
    optimizer = cfg.get("optimizer", None)
    use_scheduler = cfg.get("use_scheduler", None)
    learning_rate = cfg.get("learning_rate", None)

    # Log parameters
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("n_epochs", n_epochs)
    mlflow.log_param("optimizer", optimizer)
    mlflow.log_param("use_scheduler", use_scheduler)
    mlflow.log_param("learning_rate", learning_rate)

def log_model_summary(model):
    """Function used to log the model summary on mlflow.

    Args:
        model (torch.nn.Module): Model to log.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir, "model_summary.txt")
        path.write_text(str(summary(model)))
        mlflow.log_artifact(path)