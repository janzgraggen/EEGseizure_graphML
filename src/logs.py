# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-02 -*-
# -*- Last revision: 2025-04-29 by roduit -*-
# -*- python version : 3.11.11 -*-
# -*- Description: Functions to log parameters on Mlflow-*-

# Import libraries
import mlflow


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
