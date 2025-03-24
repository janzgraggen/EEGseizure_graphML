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
````

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

## Contributors
This project has been elaborated by Vincent Roduit, Caspar Henking, Aurel Mäder and Jan Zgraggen during the 2025 spring semester at EPFL.
