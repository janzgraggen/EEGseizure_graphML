<div align="center">
<img src="../resources/logo-epfl.png" alt="Example Image" width="192" height="108">
</div>

<div align="center">
Ecole Polytechnique Fédérale de Lausanne
</div> 
<div align="center">
EE452 - Network Machine Learning
</div> 

# Comparing time-series and graph-based models in EEG seizure detection

## Table of Contents

- [Abstract](#abstract)
- [Project Structure](#project-structure)
- [Data Structure](#data-structure)
- [Connection to SCITAS](#connection-to-scitas)
- [Mlflow](#mlflow)
- [Use the Models](#use-the-models)
- [Results](#results)
- [Contributors](#contributors)

## Abstract
Epilepsy is a neurological disorder characterized by abnormal neuronal activity that can cause sudden disruptions in brain function, known as seizures. Electroencephalography (EEG) is the primary tool for detecting these events. Machine learning methods can be effectively applied to detect seizures in EEG data. In this project, we compare the performance of time-series-based and graph-based machine learning architectures for seizure prediction. To this end, we experiment with different signal representations, graph construction methods, and model architectures. Our results show that hybrid models combining both temporal and graph-based processing of EEG data (such as GAT-LSTM) perform particularly well, achieving F1 scores of up to 80%.

## Project Structure
```
.
├── config
│   └── exp
├── data
│   ├── distances_3d.csv
│   ├── submission
│   ├── test
│   ├── train
│   └── val
├── documents
│   ├── NML_Project_Proposal.pdf
│   ├── report.pdf
├── README.md
├── requirements.txt
├── resources
│   ├── gat_lstm.png
│   ├── ...
│   └── small-gcn_metrics_report.csv
└── src
    ├── constants.py
    ├── dataloader.py
    ├── logs.py
    ├── models
    ├── plots.py
    ├── report.ipynb
    ├── run_multiple.py
    ├── run.job
    ├── run.py
    ├── train.py
    ├── transform_func.py
    └── utils.py
```

Some important files/folders:
- The [config](./config/) folder contains all the `.yml`files implementing the models explained in the report.
- The [report](./documents/report.pdf) summarizes the work done in this project.
- The [src](./src) folder contains all the code.
    - [run.py](./src/run.py): entry point to run the experiments.
    - [models](./src/models/) folder contains all the code related to the models.
    - [run.job](./src/run.job): file to use to submit a job on Scitas.
## Data Structure

Data are available using the following Kaggle command:
```
kaggle competitions download -c epfl-network-machine-learning-2025
```

The recommanded structure for the data is the following:
```
.
├── distances_3d.csv
├── sample_submission.csv
├── test
│   ├── segments.parquet
│   └── signals
└── train
    ├── segments.parquet
    └── signals
```

### Copy data to SCITAS cluster
If you need to copy data to the cluser (for instance distances_3d.csv), use the following command
```
scp path/to/data/distances_3d.csv username@izar.hpc.epfl.ch:/home/username/EE452-Network-Machine-Learning/project/data
```
and replace `path/to/data/` with your actual path and `username` with your Gaspar ID.


## Connection to SCITAS

This first part is important in case it is your first use of SCITAS.
- If you are not on campus, make sure you're connected to the EPFL VPN;
- Connect to the SCITAS Izar cluster using the following command : ```ssh -i /path/to/created/ssh/key 'username@izar.hpc.epfl.ch'```, where you replace /path/to/created/ssh/key with the file to the File document (not the .pub one) and username with your Gaspar username;
- Once you're on the server, do a git clone using SSH of this repository (you have to set up an SSH key inside the server. The .ssh repository is hidden, but it does exist at /home/username/);
- Setup a virtual environment to use packages inside the cluster : first, run ```module load gcc python``` (activates the correct Python version) and ```virtualenv --system-site-packages venvs/venv_project``` to create the environment. Then activate it by running ```source venvs/venv_project/bin/activate``` and then install missing packages on it by running ```pip install -r requirements.txt``` from the project/ folder.

This second part is what you have to follow when you want to run a job inside the cluster :

- To run a test, you simply need to do ```sbatch run.job  "cfg_folder/cfg_file.yml" seed_number``` from where the file is (both of these are not mandatory arguments). 
Warning: it is important to be in the same folder as the file when running the command, as there can be some package installation issues. ```cfg_folder```corresponds to the configuration folder of interest in ```config/exp```.

<!-- NOT FUNCTIONAL - If you want to run a so-called interactive job (basically one where you can directly see the outputs of your functions in the terminal), run ```Sinteract -p gpu -g gpu:1```. This creates a terminal directly on the node, which means that to execute code you have to run (for example in the src/ folder) ```python3 run.py --kwargs```.-->

## Mlflow
This project uses the [Mlflow library](https://mlflow.org) to keep track of the experiment.

### Lauching Mlflow server
By default, the experiments are saved in `./project/mlflow` folder. To see the results, in the terminal use the following command:
```
 mlflow ui --backend-store-uri ./project/mlruns
```

Then in your favorite browser, go to http://127.0.0.1:5000

### Recover experiment from scitas
When running a model on scitas, you can easily recover the experiment to your local machine by doing:

```
scp username@izar.hpc.epfl.ch:/home/username/EE452-Network-Machine-Learning/project/mlruns/path/to/experiment ~/path/to/destination
```

and replacing `/path/to/experiment` with the desired experiment path, `username` by your Gaspar ID and `~/path/to/destination` with the path to your destination location.

## Use the Models
If you want to use this project and test it, you can just do:
```
python project/src/run.py --cfg your/exp/folder/your_exp.yml
```

The code will run the experiment with the parameters you configured in the .yml file. You can use the one provided at the [exp folder](./config/).

# Results
The table below summarizes the results. For more informations about the methodology, please check the [report](./documents/report.pdf)

### Non-Graph-Based Models

| Model         | Signal | F1 (%)       | Acc. (%)     | F1ₖ (%)   |
|---------------|--------|--------------|--------------|-----------|
| ResNet        | FFT    | 87 ± 6       | 93 ± 4       | 70.35     |
| ResNet        | PSD    | 19 ± 2       | 22 ± 1       | -         |
| FCN           | BP     | 79 ± 6       | 85 ± 7       | -         |
| FCN           | FFT    | 77 ± 6       | 83 ± 7       | -         |
| FCN           | PSD    | 18 ± 0       | 20 ± 0       | -         |
| FCN           | -      | 75 ± 1       | 84 ± 1       | 69.62     |
| CNN           | -      | 71 ± 4       | 81 ± 3       | -         |
| CNN           | BP     | 69 ± 2       | 79 ± 3       | -         |
| CNN           | FFT    | 68 ± 9       | 74 ± 13      | -         |
| LSTM (Base)   | -      | 67.9         | -            | 67.9      |

### Graph-Based Models

| Model      | Graph [th] | Signal | F1 (%)       | Acc. (%)     | F1ₖ (%)   |
|------------|-------------|--------|--------------|--------------|-----------|
| GCN        | dst [0.8]   | FFT    | 64 ± 1       | 72 ± 1       | -         |
| GCN        | dst [0.8]   | BP     | 62 ± 1       | 73 ± 0       | -         |
| GCN        | dst [0.5]   | BP     | 61 ± 0       | 73 ± 0       | -         |
| GCN        | cor [0.5]   | FFT    | 65 ± 1       | 74 ± 2       | 76.09     |
| GCN        | coh [0.8]   | PSD    | 67 ± 2       | 75 ± 1       | 78.55     |
| LSTM-GCN   | coh [0.5]   | FFT    | 65 ± 1       | 72 ± 1       | -         |
| LSTM-GCN   | coh [0.5]   | PSD    | 77 ± 1       | 84 ± 1       | **83.55** |
| LSTM-GCN   | cor [0.8]   | PSD    | 78 ± 0       | 85 ± 1       | 80.32     |
| LSTM-GAT   | coh [0.5]   | PSD    | **80 ± 1**   | 87 ± 1       | 72.34     |

## Contributors
This project has been elaborated by Vincent Roduit, Caspar Henking, Aurel Mäder and Jan Zgraggen during the 2025 spring semester at EPFL.
