<div align="center">
<img src="../resources/logo-epfl.png" alt="Example Image" width="192" height="108">
</div>

<div align="center">
Ecole Polytechnique Fédérale de Lausanne
</div> 
<div align="center">
EE452 - Netork Machine Learning
</div> 

# Project

## Table of Contents

- [Project Structure](#project-structure)
- [Data Structure](#data-structure)
- [Connection to SCITAS](#connection-to-scitas)
- [Contributors](#contributors)

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



## Connection to SCITAS

This first part is important in case it is your first use of SCITAS.
- If you are not on campus, make sure you're connected to the EPFL VPN;
- Connect to the SCITAS Izar cluster using the following command : ```ssh -i /path/to/created/ssh/key 'username@izar.hpc.epfl.ch'```, where you replace /path/to/created/ssh/key with the file to the File document (not the .pub one) and username with your Gaspar username;
- Once you're on the server, do a git clone using SSH of this repository (you have to set up an SSH key inside the server. The .ssh repository is hidden, but it does exist at /home/username/);
- Setup a virtual environment to use packages inside the cluster : first, run ```module load gcc python``` (activates the correct Python version) and ```virtualenv --system-site-packages venvs/venv_project``` to create the environment. Then activate it by running ```source venvs/venv_project/bin/activate``` and then install missing packages on it by running ```pip install -r requirements.txt``` from the project/ folder.

This second part is what you have to follow when you want to run a job inside the cluster :

- To run a test, you simply need to do ```sbatch run.job --cfg "path/to/cfg_file.yml" --seed seed_number --run_id run_id``` from where the file is.

<!-- NOT FUNCTIONAL - If you want to run a so-called interactive job (basically one where you can directly see the outputs of your functions in the terminal), run ```Sinteract -p gpu -g gpu:1```. This creates a terminal directly on the node, which means that to execute code you have to run (for example in the src/ folder) ```python3 run.py --kwargs```.-->


## Contributors
This project has been elaborated by Vincent Roduit, Caspar Henking, Aurel Mäder and Jan Zgraggen during the 2025 spring semester at EPFL.
