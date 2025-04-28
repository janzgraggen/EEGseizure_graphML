# -*- coding: utf-8 -*-
# -*- authors : Jan Zgraggen -*-
# -*- date : 2025-04-02 -*-
# -*- Last revision: 2025-04-28 by roduit -*-
# -*- python version : 3.11.11 -*-
# -*- Description: Functions to run the project-*-

# Import libraries
import argparse
import mlflow
import matplotlib
import os
import sys
from pathlib import Path
matplotlib.use('Agg')  # Use non interactive backend

# import modules
from utils import set_seed, read_yml, choose_model
from logs import log_cfg
from dataloader import load_data
import constants

# Add base directory to sys.path
parent_dir = Path(__file__).resolve().parents[2]
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
os.chdir(parent_dir)

def main(args):
            
    # Check if config file exists
    cfg_file = args.cfg
    seed = args.seed

    set_seed(seed=seed)
    cfg = read_yml(cfg_file=cfg_file)

    # Get main infos
    experiment = cfg.get("experiment", "Default")
    name = cfg.get("name", "debug")
    
    # Set mlflow informations
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(experiment)
    run_name = "{}:{}:{}".format(experiment, name, seed)

    model_cfg = cfg.get("model", {})
    model = choose_model(cfg=model_cfg)

    print("Loading data...")
    loader_train = load_data(cfg.get("train"))
    loader_test = load_data(cfg.get("test"))

    print("Training model...")
    model.train_model(
            loader_train,
            num_epochs=model_cfg.get("n_epochs", constants.NUM_EPOCHS),
            learning_rate=model_cfg.get("learning_rate", constants.LEARNING_RATE),
            criterion_name=model_cfg.get("criterion", constants.CRITERION),
            optimizer_name=cfg.get("optimizer", constants.OPTIMIZER),
        )
    
    with mlflow.start_run(run_name=run_name):
        print('Logging model...')
        log_cfg(cfg=model_cfg)

if __name__ == '__main__':

    # Use argument
    parser = argparse.ArgumentParser(description='Run grade computation')
    parser.add_argument('--cfg', type=str, default='project/config/exp/cnn/basic_cnn.yml')    
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--run_id', type=str, default=None)

    args = parser.parse_args()
    
    main(args=args)