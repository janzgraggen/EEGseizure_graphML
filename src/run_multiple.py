# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-24 -*-
# -*- Last revision: 2025-06-09 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: File to run multiple experiments-*-

import subprocess
import os

config_folder = "project/config/exp/gcn"

config_files = os.listdir(config_folder)
# Filter out non-configuration files if necessary
config_files = [f for f in config_files if f.endswith(".yml")]
config_files = [os.path.join(config_folder.split("/")[-1], f) for f in config_files]

# Loop through each config file and run the command
for config in config_files:
    print(f"Running with config: {config}")
    subprocess.run(["python", "project/src/run.py", "--cfg", config])
