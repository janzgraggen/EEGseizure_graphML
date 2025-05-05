# -*- coding: utf-8 -*-
# -*- authors : Jan Zgraggen -*-
# -*- date : 2025-04-02 -*-
# -*- Last revision: 2025-05-05 by roduit -*-
# -*- python version : 3.11.11 -*-
# -*- Description: Functions to run the project-*-

# Import libraries
import argparse
import mlflow
import matplotlib
import os

matplotlib.use("Agg")  # Use non interactive backend

# import modules
from utils import set_seed, read_yml, choose_model
from logs import log_cfg
from dataloader import parse_datasets
import constants

def main(args: argparse.Namespace):
    """Define the main function to run the project.

    Args:
        args (argparse.Namespace) : Arguments from the command line.
    """
    # Check if config file exists
    cfg_file = args.cfg
    seed = args.seed

    set_seed(seed=seed)
    cfg = read_yml(cfg_file=cfg_file)

    # Get main infos
    experiment = cfg.get("experiment", "Default")
    name = cfg.get("name", "debug")

    # Set mlflow informations
    mlflow.set_tracking_uri(uri="./project/mlruns")
    mlflow.set_experiment(experiment)
    run_name = "{}:{}:{}".format(experiment, name, seed)

    with mlflow.start_run(run_name=run_name):

        model_cfg = cfg.get("model", {})
        model = choose_model(cfg=model_cfg)

        print("Logging model...")
        log_cfg(cfg=model_cfg)

        datasets_cfg = cfg.get("datasets", {})
        loader_train, loader_val, loader_test = parse_datasets(cfg=datasets_cfg)

        print("Training model...")
        model.fit(
            loader_train,
            loader_val,
            num_epochs=model_cfg.get("n_epochs", constants.NUM_EPOCHS),
            learning_rate=model_cfg.get("learning_rate", constants.LEARNING_RATE),
            criterion_name=model_cfg.get("criterion", constants.CRITERION),
            optimizer_name=cfg.get("optimizer", constants.OPTIMIZER),
        )

        model.predict(loader=loader_train)

        model.create_submission(loader=loader_test)

if __name__ == "__main__":

    cfg_path = "project/config/exp/"

    # Use argument
    parser = argparse.ArgumentParser(description="Run grade computation")
    parser.add_argument("--cfg", type=str, default="basic_cnn_local_upsample.yml"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run_id", type=str, default=None)

    args = parser.parse_args()

    args.cfg = cfg_path + args.cfg

    main(args=args)
