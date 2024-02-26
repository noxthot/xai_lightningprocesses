# %%
import ccc
import utils
import utils_ui
import utils_subdomains

import json
import matplotlib
import os
import pickle
import torch

import numpy as np
import pandas as pd
import seaborn as sns

import ccc


font = {'size' : 12}
matplotlib.rc('font', **font)


# %%
PATH_REFMODEL = os.path.join(ccc.MODEL_ROOT_PATH, 'targetmode_1', '2022_02_22__ALDIS_reference_gam')
PATH_REFMODEL_TESTDF = os.path.join(PATH_REFMODEL, 'test_predictions.parquet')
THRESHOLD_REFERENCE = 0.1259768351493519

font = {'size' : 12}

matplotlib.rc('font', **font)


# %%
target_mode = utils_ui.ask_targetmode()
model_dir, model_path = utils_ui.ask_modeldir(target_mode)

with open(os.path.join(model_path, 'train_monitor.pickle'), 'rb') as f:
    train_monitor = pickle.load(f)
    
_, best_epoch = utils.getOptThresholdFromVal(train_monitor, use_epoch=18)

_, model_name = utils.load_model(model_dir, torch.device("cpu"), best_epoch)

DIURNAL_CYCLE_PATH = os.path.join(model_path, "plots_diurnal_cycle")

TEST_DF = os.path.join(model_path, f"{model_name}_test_df.pickle")
SKIP_CAPE = True

# %%
with open(os.path.join(model_path, 'data_cfg.json'), 'r') as f:
    data_cfg_json = json.load(f)

DATA_PATH = os.path.join(ccc.DATA_ROOT_PATH, f'datamode_{data_cfg_json["datamode"]}', data_cfg_json['datasource'])

# %%
subdomains = utils_subdomains.get_subdomains().filter(['longitude', 'latitude', 'subdomain'])

# %%
test_df = pd.read_pickle(TEST_DF).filter(['output', 'target', 'longitude', 'latitude', 'hour'])

# %%
d_cape = pd.read_parquet(DATA_PATH, columns=['cape', 'cp', 'longitude', 'latitude', 'hour', 'year'])
d_cape = d_cape[d_cape['year'] == 2019].copy()
d_cape = d_cape.merge(subdomains, how='left', on=['longitude', 'latitude'])

d_cape['cape_released'] = d_cape['cape'] * np.where(d_cape['cp'] > 0.0, 1.0, 0.0)
d_cape['cape_released'] = np.where(d_cape['cape_released'] > 150.0, 1.0, 0.0)

d_cape.drop(['cape', 'cp', 'longitude', 'latitude', 'year'], axis=1, inplace=True)
cape_agg = d_cape.groupby(['hour', 'subdomain']).mean()

# %%
with open(os.path.join(model_path, f'{model_name}_val_scores.json'), 'r') as f:
    test_scores_json = json.load(f)
    THRESHOLD = test_scores_json["used_threshold"]

#%%
test_df_ref = pd.read_parquet(PATH_REFMODEL_TESTDF)
test_df_ref["output_reference"] = (test_df_ref["fit"] > THRESHOLD_REFERENCE) * 1.0
test_df_ref = test_df_ref[["longitude", "latitude", "output_reference", "hour"]]

# %%
test_df = test_df.merge(test_df_ref, how="left", on=["longitude", "latitude", "hour"]).merge(subdomains, how='left', on=['longitude', 'latitude'])
print(f"Using {THRESHOLD} as threshold...", flush=True)
test_df['output'] = np.where(test_df["output"] > THRESHOLD, 1.0, 0.0)
test_df.drop(['longitude', 'latitude'], axis=1, inplace=True)
df_agg = test_df.groupby(['hour', 'subdomain']).mean()

# %%
ofile = os.path.join(DIURNAL_CYCLE_PATH, "cycles_cape.png")
plt = sns.lineplot(data=cape_agg, x="hour", y="cape_released", hue="subdomain")
fig = plt.get_figure()
fig.savefig(ofile)
fig.clf()

# %%
ofile = os.path.join(DIURNAL_CYCLE_PATH, "cycles_output.png")
plt = sns.lineplot(data=df_agg, x="hour", y="output", hue="subdomain")
fig = plt.get_figure()
fig.savefig(ofile)
fig.clf()

# %%
ofile = os.path.join(DIURNAL_CYCLE_PATH, "cycles_target.png")
plt = sns.lineplot(data=df_agg, x="hour", y="target", hue="subdomain")
fig = plt.get_figure()
fig.savefig(ofile)
fig.clf()

# %%
df_long = pd.melt(df_agg.reset_index(), id_vars=['hour', 'subdomain'], var_name='type')

if SKIP_CAPE is False:
    cape_long = cape_agg.reset_index().rename({'cape_released': 'value'}, axis=1)
    cape_long['value'] = cape_long['value']
    cape_long['type'] = 'CAPE released'
    df_long = pd.concat([df_long, cape_long])

df_long['subdomain'].replace(utils_subdomains.LABELS, inplace=True)
df_long['type'].replace({'output': 'model', 'output_reference' : 'model_ref', 'target': 'observations'}, inplace=True)
df_long['value'] *= 100.0

ofile = os.path.join(DIURNAL_CYCLE_PATH, "cycles_paper.png")
g = sns.FacetGrid(df_long, col="subdomain", hue_kws={'color': ['#1f78b4', '#b2df8a', '#a6cee3'], "ls" : ["-", "-.", "--"]}, hue="type")
g.map(sns.lineplot, "hour", "value")
g.set_titles('{col_name}')
g.set_ylabels('Occurrence [%]')
g.add_legend()
g.savefig(ofile)