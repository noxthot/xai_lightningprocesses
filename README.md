# Identifying Lightning Processes in ERA5 Soundings with Deep Learning

This is the source code accompanying the pre-print:

Gregor Ehrensperger, Tobias Hell, Georg J. Mayr, and Thorsten Simon (2022). Identifying Lightning Processes in ERA5 Soundings with Deep Learning. Available on arXiv, 2210.11529. DOI: [10.48550/ARXIV.2210.11529](https://doi.org/10.48550/arXiv.2210.11529)

## Setup (Ubuntu)
### Conda
Use conda to install all required modules (default environment: `meteorology_verticallearning`):\
```
conda env create -f environment.yml
```

Attention: Using the environment requires a CUDA-enabled graphics card (basically any Nvidia graphics card). Otherwise you might want to install the required packages manually.

In case you already got the environment and only need to update to the latest `environment.yml` use:\
```
conda activate meteorology_verticallearning
conda env update --file environment.yml --prune
```

After manually adding a package, update the `environment.yml` using this command:\
```
conda env export --name meteorology_verticallearning > environment.yml
```

### Java
Pyspark needs a specific openjdk version:
```
sudo apt install openjdk-8-jdk
```

### Hadoop
Our code uses Spark which takes use of hadoop. The code also works without having a local hadoop installation, but it prints a warning.
To install hadoop, simply download the latest stable hadoop version [here](https://downloads.apache.org/hadoop/common/stable/). Unpack
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
- `test.py`: Evaluates the performance of the trained neural network on previously unseen test data. This file was used to compute the corresponding confusion matrix in Table 2.
- `test_shap.py`: Computes the shapley values using the trained model on the test data.
- `validation_scores.py`: Computes the classification threshold such that the diurnal cycle is least biased on the validation data.
- `analyse_diurnal_cycles.py`: Uses the previously calculated classification threshold to generates plots that visualize how that threshold performs in reproducing the diurnal cycle on previously unseen test data. This file produces figure 2 of the paper.
- `analyse_shap_and_features.ipynb`: Visualizes the shap and real values of the vertical profiles distinguishing between true positives, false positives, false negatives, aswell as providing some plots regarding cloud top and bottom height. This file was used to generate figures 3, 4, 5 and 7 of the paper.
- `flash_case_study_final.ipynb`: Visualizes network classifications at a specific time on a map of austria. This file generates figure 6 of the paper.

### Runnable Files (Reference model):
- `reference_model.R`: Trains the reference model.
- `reference_valpred.R`: Stores the model output on the validation data (used for calculating the classification threshold later on).
- `reference_test.py`: Evaluates the trained reference model on previously unseen test data. This file was used to compute the corresponding confusion matrix in Table 2.

### Helper files
- `ccc.py`: Defining some global constants.
- `stats.py` and `stats_flash.py`: Used to calculate some of the constants defined in `ccc.py`.
- `utils*.py`: Helper functions containing various routines.
