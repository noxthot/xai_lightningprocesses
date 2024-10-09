from datetime import date
import os
from enum import Enum

USE_CUDA_IF_AVAILABLE = True

TORCH_NUM_WORKERS = 2
DATALOADER_NUM_WORKERS = 8 if USE_CUDA_IF_AVAILABLE else (8 - TORCH_NUM_WORKERS)

BATCH_SIZE = 1024

SEED_RANDOM = 1337  # Set to None to disable

DATA_ROOT_PATH = os.path.join('.', 'data', 'data_processed')
MODEL_ROOT_PATH = os.path.join('.', 'data', 'models')
DATASTATS_PATH = os.path.join(".", "data", "data_stats")
CACHE_PATH = os.path.join(".", "data", "cache")

INDEX_COLS = ['longitude', 'latitude', 'year', 'month', 'day', 'hour']

TRAIN_COLS_TMP = ["ciwc", "clwc", "crwc", "cswc", "q", "t", "u", "v", "w", "longitude", "latitude", "hour", "dayofyear", "topography"]  # Changes here also need to be reflected in transformrow() to take effect
LVL_TRAIN_COLS = ["ciwc", "clwc", "crwc", "cswc", "q", "t", "u", "v", "w"]

LVL_COLS = LVL_TRAIN_COLS + ["geoh", "pres"]

TRAIN_COLS = []

for traincol in TRAIN_COLS_TMP:
    if traincol in LVL_TRAIN_COLS:
        for idx in range(74):
            lvl = 64 + idx
            TRAIN_COLS.append(f"{traincol}_lvl{lvl}")
    else:
        TRAIN_COLS.append(traincol)

DATA_VAR_STATS = {'ciwc': {'min': -5.820766E-11, 'max': 0.0014211535, 'mean': 2.9080181E-06, 'std': 1.5918778E-05},
                  'clwc': {'min': -1.1641532E-10, 'max': 0.0018931772, 'mean': 7.880163E-06, 'std': 3.3968518E-05},
                  'crwc': {'min': -5.820766E-11, 'max': 0.0012295842, 'mean': 1.4961037E-06, 'std': 1.1229228E-05},
                  'cswc': {'min': -2.3283064E-10, 'max': 0.0046960115, 'mean': 5.119709E-06, 'std': 3.452298E-05},
                  'q': {'min': -0.00036671013, 'max': 0.02839908, 'mean': 0.0035570883, 'std': 0.0038681545},
                  't': {'min': 199.86183, 'max': 313.22552, 'mean': 258.9201, 'std': 27.922228},
                  'u': {'min': -48.20824, 'max': 75.00498, 'mean': 6.3445525, 'std': 9.254158},
                  'v': {'min': -63.93594, 'max': 68.34018, 'mean': -0.18657665, 'std': 8.218262},
                  'w': {'min': -12.683085, 'max': 5.9455957, 'mean': -0.008953505, 'std': 0.27458853},
                  'hour': {'min': 0, 'max': 23, 'mean': 11.5, 'std': 6.639528, 'delta': 1},
                  'dayofyear': {'min': 150.757507324219, 'max': 242.544998168945, 'mean': 196.608764648438, 'std': 26.5579566955566},
                  'longitude': {'min': 8.25, 'max': 16.75, 'mean': 12.5, 'std': 2.524876, 'delta': 0.25},
                  'latitude': {'min': 45.25, 'max': 49.75, 'mean': 47.5, 'std': 1.3693064, 'delta': 0.25},
                  'topography': {'min': -40, 'max': 2400, 'mean': 711, 'std': 528.3},
                  }

DATA_TARGET_STATS = {"flash": {"0": 14311772, "1": 351478},
                     "flash_windowed_max": {"0": 13948462, "1": 714788},
                     "flash_windowed_sum": {"0": 13948462, "1": 442704, "2": 205064, "3": 67020},
                    }

LOG_BATCH_INTERVAL = 100

# refdate to compute day_of_year accounding for leap days
REF_DATE = date.fromisoformat('2010-01-01')
START_DAY_HOUR = 6  # Data from 00:00 to this hour will be counted to the previous day

# Group ERA5 variables according their physical meaning
class VariableType(Enum):
    CLOUD = 1
    MASS = 2
    WIND = 3
    SETTING = 4
    PREDICTION = 5

VARTYPE_LOOKUP = {
    'ciwc': VariableType.CLOUD.name,
    'cswc': VariableType.CLOUD.name,
    'clwc': VariableType.CLOUD.name,
    'crwc': VariableType.CLOUD.name,
    'q': VariableType.MASS.name,
    't': VariableType.MASS.name,
    'u': VariableType.WIND.name,
    'v': VariableType.WIND.name,
    'w': VariableType.WIND.name,
    'topography': VariableType.SETTING.name,
    'latitude': VariableType.SETTING.name,
    'hour': VariableType.SETTING.name,
    'dayofyear': VariableType.SETTING.name,
    'longitude': VariableType.SETTING.name,
    'output': VariableType.PREDICTION.name,
}
