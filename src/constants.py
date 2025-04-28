# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-02 -*-
# -*- Last revision: 2025-04-28 by roduit -*-
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

# ======================================================================================
# =====                                MODEL PARAMS                                =====
# ======================================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

NUM_EPOCHS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
CRITERION = 'bce'
OPTIMIZER = 'Adam'