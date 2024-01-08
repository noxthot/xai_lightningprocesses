import json
import pickle
import os
import shap
import torch

import numpy as np
import pandas as pd
import torch.nn as nn

import ccc
import utils
import utils_shap
import utils_ui

from petastorm.spark import make_spark_converter

NR_ROWS_USED_TEST = None  # Limits the input data to this number of rows. Set to None to use full dataset
USE_NO_FLASH_AS_BACKGROUND = True

LON_RNG = utils.getVarRange('longitude')
LAT_RNG = utils.getVarRange('latitude')

use_cuda = ccc.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available()
devicestr = 'cuda' if use_cuda else 'cpu'
device = torch.device(devicestr)


target_mode = utils_ui.ask_targetmode()
modeldir, modelpath = utils_ui.ask_modeldir(target_mode)

with open(os.path.join(modelpath, 'train_monitor.pickle'), 'rb') as f:
    train_monitor = pickle.load(f)

_, best_epoch = utils.getOptThresholdFromVal(train_monitor)

model, model_name = utils.load_model(modeldir, device, best_epoch)
no_flash_postfix = "_no_flash" if USE_NO_FLASH_AS_BACKGROUND else ""
shap_path = os.path.join(modelpath, f"{model_name}_shap_parquet_bg_by_lon_lat{no_flash_postfix}")

if not os.path.isdir(shap_path):
    os.makedirs(shap_path)

with open(os.path.join(modelpath, 'data_cfg.json'), 'r') as f:
    config_data = json.load(f)

with open(os.path.join(modelpath, 'model_cfg.json'), 'r') as f:
    config_model = json.load(f)

add_cols = utils_shap.META_COLS

if not 'target_mode' in config_model or config_model['target_mode'] == 1:
    config_model['target_mode'] = 1
else:
    add_cols += ["flash_windowed_sum", "flash_windowed_max"]

fit_path = os.path.join(modelpath, f"{model_name}_test_df.pickle")
print("Loading fit (test_df) data", flush=True)
fitdf = pd.read_pickle(fit_path)
cols_drop = [col for col in fitdf.columns if col.startswith("hlayer_")] + ['target']
fitdf.drop(cols_drop, axis=1, inplace=True)
fitdf.rename(lambda c : utils_shap.colname_meta_infix(c), axis='columns', inplace=True)

print("Get test dataframe", flush=True)
testdf = utils.get_testdf_spark(config_data, add_cols, NR_ROWS_USED_TEST)

return_dfs = []

converter = make_spark_converter(testdf)

norm_fun = utils.getnormfun(config_model["norm_fun"])

batch_size = len(converter)

print(f"Get meta and data ({batch_size} rows)", flush=True)
with converter.make_torch_dataloader(batch_size=batch_size, transform_spec=utils.get_transform_spec(
                                        norm_fun, config_model['target_mode'],
                                        use_train_cols=ccc.TRAIN_COLS, return_cols=add_cols),
                                        num_epochs=1, workers_count=ccc.DATALOADER_NUM_WORKERS) as test_dataloader:
    test_dataloader_iter = iter(test_dataloader)
    
    batch = next(test_dataloader_iter)
    features_batch = batch['features'].to("cpu", non_blocking=True)
    del batch['features']
    del batch['label']

    meta = pd.DataFrame(batch)
    meta.rename(lambda c : utils_shap.colname_meta_infix(c), axis='columns', inplace=True)

for lon in LON_RNG:
    print(f"Filter for longitude {lon}", flush=True)
    meta_filt_lon = meta.query(f"{utils_shap.colname_meta_infix('longitude')} == {lon}")
    idx_filt_lon = meta_filt_lon.index
    meta_filt_lon.reset_index(drop=True, inplace=True)
    data_filt_lon = features_batch[idx_filt_lon,].to(device, non_blocking=True)

    for lat in LAT_RNG:
        print(f"Filter for longitude {lat}", flush=True)
        meta_filt_lat = meta_filt_lon.query(f"{utils_shap.colname_meta_infix('latitude')} == {lat}")
        idx_filt_lat = meta_filt_lat.index
        meta_filt_lat.reset_index(drop=True, inplace=True)
        data_filt_lat = data_filt_lon[idx_filt_lat,]

        if USE_NO_FLASH_AS_BACKGROUND:
            meta_filt_lat_no_flashes = meta_filt_lat.query(f"{utils_shap.colname_meta_infix('flash')} == 0")
            background = data_filt_lat[meta_filt_lat_no_flashes.index,]
        else:
            background = data_filt_lat

        print(f"Compute shapley values for lon {lon}, lat {lat}")
        e = shap.DeepExplainer(model, background)
        base_value = e.expected_value
        print(f"Base value: {base_value}")
        shap_values = e.shap_values(data_filt_lat)

        shap_df = pd.DataFrame(np.append(shap_values, np.full((np.shape(shap_values)[0], 1), base_value), axis=1), columns=[utils_shap.colname_shap_infix(col) for col in ccc.TRAIN_COLS] + ["shap_base_value"])
        shap_df = pd.concat([shap_df, meta_filt_lat], axis=1)
        return_dfs.append(shap_df)

print("Export results", flush=True)
return_df = pd.concat(return_dfs, ignore_index=True)
return_df = utils.joinDataframes(return_df, fitdf, True)

return_df.to_parquet(shap_path,
                    engine="pyarrow",
                    compression="snappy",
                    index=False,
                    partition_cols=[utils_shap.colname_meta_infix('flash'), utils_shap.colname_meta_infix('hour')])

print("Fin.", flush=True)

