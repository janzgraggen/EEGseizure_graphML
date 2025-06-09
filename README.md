<div align="center">
<img src="../resources/logo-epfl.png" alt="Example Image" width="192" height="108">
</div>

<div align="center">
Ecole Polytechnique Fédérale de Lausanne
</div> 
<div align="center">
EE452 - Netork Machine Learning
</div> 

# Graph-based EEG Analysis

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
Epilepsy is a neurological disorder caused by abnormal neuronal activity. This irregular brain activity often leads to unpredictable disruptions of normal brain function, known as epileptic seizures.
The primary method for detecting these seizures is electroencephalography (EEG), which involves recording brain activity using multiple electrodes.
Manually detecting seizures—especially in long EEG recordings—can be time-consuming. To address this, Machine Learning (ML) techniques can assist in the diagnostic process. These techniques include traditional models such as CNNs, LSTMs, and ResNets, as well as more specialized architectures based on graph structures, such as Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs). This project aims to compare the different methods.

## Project Structure
```
.
├── README.md
├── data
│   ├── distances_3d.csv
│   ├── sample_submission.csv
│   ├── test
│   └── train
├── documents
│   └── NML_Project_Proposal.pdf
└── src
    └── example.ipynb
```
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

## Contributors
This project has been elaborated by Vincent Roduit, Caspar Henking, Aurel Mäder and Jan Zgraggen during the 2025 spring semester at EPFL.