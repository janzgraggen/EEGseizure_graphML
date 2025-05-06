# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-02 -*-
# -*- Last revision: 2025-05-06 by roduit -*-
# -*- python version : 3.9.14 -*-
# -*- Description: Constants used in the project -*-

# Import libraries
from pathlib import Path
import torch

# ======================================================================================
# =====                                DIRECTORIES                                 =====
# ======================================================================================
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"

TRAIN_DIR = DATA_DIR / "train"
CLIPS_TRAIN_FILE = TRAIN_DIR / "segments.parquet"

TEST_DIR = DATA_DIR / "test"
CLIPS_TEST_FILE = TEST_DIR / "segments.parquet"

SUBMISSION_DIR = DATA_DIR / "submission"

if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
if not SUBMISSION_DIR.exists():
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

SUBMISSION_FILE = SUBMISSION_DIR / "submission.csv"

DISTANCE_3D_FILE = DATA_DIR / "distance_3d.csv"

# ======================================================================================
# =====                                MODEL PARAMS                                =====
# ======================================================================================
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)

NUM_EPOCHS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
CRITERION = "bce"
OPTIMIZER = "Adam"
