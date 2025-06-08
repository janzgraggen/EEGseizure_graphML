# -*- coding: utf-8 -*-
# -*- authors : Jan Zgraggen -*-
# -*- date : 2025-04-02 -*-
# -*- Last revision: 2025-05-26 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Functions to run the project-*-

# Import libraries
import argparse
import mlflow
import matplotlib
import os
import numpy as np

matplotlib.use("Agg")  # Use non interactive backend

# import modules
from utils import set_seed, read_yml, choose_model
from logs import log_cfg, log_model_summary
from dataloader import parse_datasets
import constants

def main(args: argparse.Namespace):
    cfg_file = args.cfg
    seed = args.seed

    set_seed(seed=seed)
    cfg = read_yml(cfg_file=cfg_file)

    experiment = cfg.get("experiment", "Default")
    name = cfg.get("name", "debug")
    model_cfg = cfg.get("model", {})
    datasets_cfg = cfg.get("datasets", [])
    config_dataset = cfg.get("config_dataset", {})
    n_splits = config_dataset.get("n_splits", 5)

    mlflow.set_tracking_uri(uri="./project/mlruns")
    mlflow.set_experiment(experiment)

    fold_metrics = {
        "train_accuracy": [],
        "train_f1_score": [],
        "val_accuracy": [],
        "val_f1_score": [],
    }

    for fold in range(n_splits):
        run_name = f"{experiment}:{name}:seed{seed}:fold{fold}"
        with mlflow.start_run(run_name=run_name):
            run_id = mlflow.active_run().info.run_id
            set_seed(seed + fold)


            model = choose_model(cfg=model_cfg)
            log_cfg(cfg=model_cfg)
            
            if fold == 0:
                print(f"\n========== Model Summary ==========")
                log_model_summary(model=model)

            print(f"\n========== Fold {fold + 1}/{n_splits} ==========")


            loader_train, loader_val, loader_test = parse_datasets(
                datasets=datasets_cfg,
                config=config_dataset,
                fold=fold
            )

    
            model.fit(
                loader_train,
                loader_val,
                num_epochs=model_cfg.get("n_epochs", constants.NUM_EPOCHS),
                learning_rate=float(model_cfg.get("learning_rate", constants.LEARNING_RATE)),
                criterion_name=model_cfg.get("criterion", constants.CRITERION),
                optimizer_name=model_cfg.get("optimizer", constants.OPTIMIZER),
            )
            tr_acc, tr_f1,_ =   model.predict(loader=loader_train)
            val_acc, val_f1,_ = model.predict(loader=loader_val)
            fold_metrics["train_accuracy"].append(tr_acc)
            fold_metrics["train_f1_score"].append(tr_f1)
            fold_metrics["val_accuracy"].append(val_acc)
            fold_metrics["val_f1_score"].append(val_f1)

    print(f"\n========== Fold Metrics ==========")
    for metric, values in fold_metrics.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        if metric.startswith("val_"):
            print(f"\033[1m\033[31m{metric}\033[0m: {mean_value:.2f} ± {std_value:.2f}")
        else:
            print(f"{metric}: {mean_value:.2f} ± {std_value:.2f}")
            
    print(f"\n========== Train on all  ==========")
    run_name = f"{experiment}:{name}:seed{seed}:all"
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        set_seed(seed + fold)
        model = choose_model(cfg=model_cfg)
        log_cfg(cfg=model_cfg)
        loader_train, loader_val, loader_test = parse_datasets(
                datasets=datasets_cfg,
                config=config_dataset,
                fold=fold,
                submission=True
            )
        model.fit(
                loader_train,
                None,
                num_epochs=model_cfg.get("n_epochs", constants.NUM_EPOCHS),
                learning_rate=float(model_cfg.get("learning_rate", constants.LEARNING_RATE)),
                criterion_name=model_cfg.get("criterion", constants.CRITERION),
                optimizer_name=model_cfg.get("optimizer", constants.OPTIMIZER),
                submission=True,
            )

        print(f"\n========== Create Submission ==========")
        model.create_submission(
            loader=loader_test,
            path=os.path.join(constants.SUBMISSION_DIR, f"{run_id}_all.csv")
        )

if __name__ == "__main__":

    # Use argument
    parser = argparse.ArgumentParser(description="Run model computation")
    parser.add_argument("--cfg", type=str, default="fcn/fcn_fft.yml"
    )
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    args.cfg = os.path.join(constants.CFG_DIR, args.cfg)

    main(args=args)