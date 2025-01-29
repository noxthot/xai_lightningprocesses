# Identifying Lightning Processes in ERA5 Soundings with Deep Learning

[![DOI](https://zenodo.org/badge/550170268.svg)](https://zenodo.org/badge/latestdoi/550170268)

This is the source code belonging to the paper of the same name [[1]](#1).

## Setup
While the code should be platform-independent, it was mainly tested using Ubuntu 20.04 and 22.04.

### Paths
Processed data is found in `Seafile/mlvapto/`. Simply create a symbolic link (Ubuntu) or link (Windows) to that directory and name the link `data`. Raw data needs to be put into `data_raw_aut` (Austria) and `data_raw_eu` (EU).

### Python
#### PDM
Use [`pdm`](https://pdm-project.org/) to install/sync all required modules:
```bash
pdm sync
```

To run code / `IPython` / `jupyter lab` in this environment:
```bash
pdm run python <SCRIPT.py>

pdm run ipython

pdm run jupyter lab
```

To add a package:
```bash
pdm add <PACKAGE_NAME>
```

### Java
Pyspark needs a specific openjdk version (`openjdk-8`).

To install in Ubuntu:
```
sudo apt install openjdk-8-jdk
```

For other systems, follow [this link](https://learn.microsoft.com/de-de/java/openjdk/download#openjdk-8).

### Hadoop
Our code uses Spark which takes use of hadoop. The code also works without having a local hadoop installation, but it prints a warning.
To install hadoop under Ubuntu, simply download the latest stable hadoop version [here](https://downloads.apache.org/hadoop/common/stable/). Unpack
the archive and add this line to your `~/.bashrc`:
```
export HADOOP_HOME="/path/to/your/unpacked/hadoop-x.x.x"
```

## Usage

The provided code is quite memory-consuming and was executed on a workstation with 32GB RAM.

### Data
How to retrieve the data is described in `data-preprocessing/README_preprocessing.md`.

### Runnable Files (Neural Network)
The order of the following list defines the order in which the scripts should be run.
- `etl.py`: This data pipeline transforms the raw data (see previous subsection) into the format that is required for training, testing and analysing with the following files.
- `train.py`: Trains a neural network on the transformed data.
- `test.py`: Evaluates the performance of the trained neural network on previously unseen test data. This file was used to compute the corresponding confusion matrix in Table 3.
- `test_shap.py`: Computes the shapley values using the trained model on the test data.
- `validation_scores.py`: Computes the classification threshold such that the diurnal cycle is least biased on the validation data.
- `analyse_shap_and_features.ipynb`: Visualizes the shap and real values of the vertical profiles distinguishing between true positives, false positives, false negatives, aswell as providing some plots regarding cloud top and bottom height. This file was used to generate figures 1, 2, 3 and 4 of the paper.
- `flash_case_study_final.ipynb`: Visualizes network classifications at a specific time on a map of austria. This file generates figure 5 of the paper.

### Runnable Files (Reference model):
- `reference_model.R`: Trains the reference model.
- `reference_valpred.R`: Stores the model output on the validation data (used for calculating the classification threshold later on).
- `reference_test.py`: Evaluates the trained reference model on previously unseen test data. This file was used to compute the corresponding confusion matrix in Table 3.

### Helper files
- `ccc.py`: Defining some global constants.
- `stats.py` and `stats_flash.py`: Used to calculate some of the constants defined in `ccc.py`.
- `utils*.py`: Helper functions containing various routines.

# References
<a id="1">[1]</a> Ehrensperger, G., Simon, T., Mayr, G. & Hell, T. (2024). Identifying Lightning Processes in ERA5 Soundings with Deep Learning. arXiv (https://arxiv.org/abs/2210.11529)
