# %%
import ccc
import utils
import utils_ui

import os
import pickle
import json
import torch

import numpy as np
import torch.nn as nn

from petastorm.spark import make_spark_converter

NR_CHUNKS = 19  # This has to be a divider of lenconvtest; otherwise we lose samples or look at some samples twice

target_mode = utils_ui.ask_targetmode()
model_dir, model_path = utils_ui.ask_modeldir(target_mode)

with open(os.path.join(model_path, 'data_cfg.json'), 'r') as f:
    config_data = json.load(f)

with open(os.path.join(model_path, 'model_cfg.json'), 'r') as f:
    config_model = json.load(f)

with open(os.path.join(model_path, 'train_monitor.pickle'), 'rb') as f:
    train_monitor = pickle.load(f)

target_col = utils.getTargetCol(config_model["target_mode"])
add_cols = ["longitude", "latitude", "hour", "day", "month", "year"]
sel_cols = list(set(ccc.TRAIN_COLS + add_cols + [target_col]))

df_val = utils.get_valdf_spark(config_data, sel_cols, None)

# %%
HOOK_ACT_NAME = 'hlayer'

converter_val = make_spark_converter(df_val)
lencon = len(converter_val)

if lencon % NR_CHUNKS != 0:
    raise Exception(f"NR_ITERATIONS (= {NR_CHUNKS}) has to be a divider of lencon (= {lencon}).")

print(f"Validation dataset: {lencon} samples")

norm_fun = utils.getnormfun(config_model["norm_fun"])

use_cuda = ccc.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available()
devicestr = 'cuda' if use_cuda else 'cpu'
device = torch.device(devicestr)

if not use_cuda:
    torch.set_num_threads(ccc.TORCH_NUM_WORKERS)
    
_, best_epoch = utils.getOptThresholdFromVal(train_monitor)

model, model_name = utils.load_model(model_dir, device, best_epoch)
hook_name_list = []

for idx in range(len(model.hidden_layers)):
    hook_name_list.append(f"{HOOK_ACT_NAME}_{idx}")
    model.hidden_layers[idx].register_forward_hook(utils.get_activation(f"{HOOK_ACT_NAME}_{idx}"))

loss_criterion = nn.BCEWithLogitsLoss(pos_weight=utils.calc_pos_weights(config_model["target_mode"], device))

with converter_val.make_torch_dataloader(batch_size=(lencon / NR_CHUNKS), transform_spec=utils.get_transform_spec(norm_fun, use_train_cols=ccc.TRAIN_COLS,
                                        return_cols=add_cols, target_mode=config_model["target_mode"]),
                                        num_epochs=1, workers_count=ccc.DATALOADER_NUM_WORKERS) as val_dataloader:
    val_dataloader_iter = iter(val_dataloader)

    val_loss, val_df, val_score = utils.test_one_epoch(model, device, val_dataloader_iter, NR_CHUNKS,
                                                    loss_criterion, config_model["target_mode"], hook_name_list)

# %%
pred_score = val_df['output']

expect = df_val[['flash']].toPandas()['flash'].mean()
opt_threshold = np.quantile(pred_score, 1 - expect)

val_score["flash_mean"] = float(expect)
val_score["opt_thr_calibration"] = float(opt_threshold)

json_out = json.dumps(val_score, indent=4)

json_file = open(os.path.join(model_path, f"{model_name}_val_scores.json"), 'w')
json_file.write(json_out)
json_file.close()

print(" * DONE.", flush=True)
