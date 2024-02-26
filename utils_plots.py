import datetime
import itertools
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ccc
import os
import utils_shap

from matplotlib.patches import Patch
from scipy import interpolate

CLUSTER_COLORS = [
    (0.69803921568627447, 0.87450980392156863, 0.54117647058823526),
    (0.2,                 0.62745098039215685, 0.17254901960784313),
    (0.8901960784313725,  0.10196078431372549, 0.10980392156862745),
    (0.98431372549019602, 0.60392156862745094, 0.6                ),
    (0.902,               0.902,               1.0                ),
]

COLUMN_UNITS = {
    "level" : "",
    "geopotential_altitude" : "m",
    'ciwc': "kg/kg",
    'cswc': "kg/kg",
    'clwc': "kg/kg",
    'crwc': "kg/kg",
    'q': "kg/kg",
    't': "K",
    'u': "m/s",
    'v': "m/s",
    'w': "Pa/s",
    'topography': "m",
}


def formatManyProfilesPlots(g, target_var, y_axis, clusternr):
    for ax in g.axes.ravel():
        ax.ticklabel_format(style='plain', axis='both')

        # set x label
        if target_var == "feature":
            varname = ax.get_title().split(" ")[-1]  # extract used variable from title
            xlabel = "value" + (f" [{COLUMN_UNITS[varname]}]" if varname in COLUMN_UNITS else "")
        else:
            xlabel = "shap value []"
        
        ax.set_xlabel(xlabel)

    ylabel = y_axis + (f" [{COLUMN_UNITS[y_axis]}]" if y_axis in COLUMN_UNITS else "")
    g.set_ylabels(ylabel.replace("_", " "))

    g.refline(y=0)

    if clusternr != "":
        g.fig.suptitle(f"Cluster {clusternr}")

    g.fig.autofmt_xdate()


def plot_many_profiles_internal_agg(dd_profiles_agg, target_var, y_axis, palette, plot_clusters, clusternr="", save_filepath=""):
    if target_var == "shap":        
        colprefix = "shapval_"
        sharex = True
    elif target_var == "feature":
        colprefix = "meta_"
        sharex = False
    else:
        raise Exception(f"{target_var} unknown")

    ylims = [min(dd_profiles_agg[y_axis]), max(dd_profiles_agg[y_axis])]

    dd_profiles_agg_q = dd_profiles_agg.query(f"cluster == {clusternr}") if clusternr != "" else dd_profiles_agg

    huecol = "cluster" if "cluster" in dd_profiles_agg_q.columns else "variable"
    hue_order = list(plot_clusters.keys()) if "cluster" in dd_profiles_agg_q.columns else None

    legend_keys = plot_clusters if clusternr == "" else [int(clusternr)]
    legend_elements = [Patch(color=(plt.colormaps[palette](idx) if isinstance(palette, str) else CLUSTER_COLORS[key]), label=plot_clusters[key]) for idx, key in enumerate(legend_keys)]
    
    dd_profiles_agg_q = dd_profiles_agg_q.sort_values(by=['variable', huecol, y_axis]).reset_index()

    g = sns.FacetGrid(dd_profiles_agg_q, col='variable', hue=huecol, hue_order=hue_order, height=7, aspect=0.7, sharex=sharex, sharey=True, palette=palette) 
    g.map_dataframe(sns.lineplot, sort=False, y=y_axis, x=f"{colprefix}qlow", hue_order=hue_order, alpha=0.1, estimator=None, err_style=None)
    g.map_dataframe(sns.lineplot, sort=False, y=y_axis, x=f"{colprefix}qhigh", hue_order=hue_order, alpha=0.1, estimator=None, err_style=None)
    g.map_dataframe(sns.lineplot, sort=False, y=y_axis, x=f"{colprefix}median", lw=3, hue_order=hue_order, estimator=None, err_style=None)
    g.map(plt.fill_betweenx, y_axis, f'{colprefix}qlow', f'{colprefix}qhigh', alpha=0.4)
    g.set(ylim=ylims)
    g.add_legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1, 0.5))

    formatManyProfilesPlots(g, target_var, y_axis, clusternr)

    if save_filepath != "":
        save_dir = os.path.dirname(save_filepath)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        g.savefig(f"{save_filepath}.pdf", dpi=600)
        g.savefig(f"{save_filepath}.png", dpi=600)
    else:
        g.fig.show()

    return g
    

def plot_many_profiles_internal_mult(dd_profiles, target_var, y_axis, palette, plot_clusters, clusternr="", save_filepath=""):
    nr_profiles = 1500

    if target_var == "shap":        
        coltarget = "shapval"
        sharex = True
    elif target_var == "feature":
        coltarget = "meta"
        sharex = False
    else:
        raise Exception(f"{target_var} unknown")

    dd_profiles_q = dd_profiles.query(f"cluster == {clusternr}") if clusternr != "" else dd_profiles
    dd_profiles_q["cluster"] = dd_profiles_q["cluster"].as_categorical()

    huecol = "cluster" if "cluster" in dd_profiles_q.columns else "variable"
    hue_order = list(plot_clusters.keys()) if "cluster" in dd_profiles_q.columns else None

    legend_keys = plot_clusters if clusternr == "" else [int(clusternr)]
    legend_elements = [Patch(color=(plt.colormaps[palette](idx) if isinstance(palette, str) else CLUSTER_COLORS[key]), label=plot_clusters[key]) for idx, key in enumerate(legend_keys)]
        
    print(f"Sampling {nr_profiles} unique profiles")
    event_ids = dd_profiles['unique_profile'].unique().tolist()
    sampled_ids = random.sample(event_ids, nr_profiles)

    print("Plotting")
    g = sns.FacetGrid(dd_profiles_q[dd_profiles_q.unique_profile.isin(sampled_ids)], col='variable', hue=huecol, hue_order=hue_order, height=7, aspect=0.7, sharex=sharex, sharey=True, palette=palette)    
    g.map_dataframe(sns.lineplot, y=y_axis, x=coltarget, units='unique_profile', hue_order=hue_order, alpha=0.05, estimator=None)
    g.add_legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1, 0.5))

    formatManyProfilesPlots(g, target_var, y_axis, clusternr)

    if save_filepath != "":
        g.savefig(save_filepath, dpi=600)

    return g


"""
Aggregated plots
ptype = q50, q90, mult
x_axis = level, geopotential_altitude
"""
def plot_many_profiles(df_many_cases, target_var, ptype='q90', y_axis='level', separate_clusters=False, save_path="", use_cache=True, plot_clusters=dict(), only_show_cols=[], write_cache=True, color_palette=None):
    if ptype not in ['q50', 'q90', 'mult']:
        raise Exception(f"ptype {ptype} unknown")

    if y_axis not in ['level', 'geopotential_altitude']:
        raise Exception(f"y_axis {y_axis} unknown")

    sns.set_theme(style="whitegrid", font_scale=1.5)
    
    cache_filename = "cached_profiles.pickle"
    cache_filepath = os.path.join(save_path, cache_filename)

    if use_cache and os.path.isfile(cache_filepath):
        print(f"Using cached file {cache_filepath}")
        dd_profiles = pd.read_pickle(cache_filepath)
    else:
        mycases = df_many_cases.copy()
        mycases.drop([c for c in mycases.columns if c.startswith("geoh")], axis="columns", inplace=True)

        idxcols = list(set(utils_shap.META_COLS_NOLVL) - set(["flash"]))

        mycases_wl = pd.wide_to_long(mycases,
                                stubnames=[f"{c}{infix}_lvl" for c in ccc.LVL_TRAIN_COLS for infix in {utils_shap.META_INFIX, utils_shap.SHAP_INFIX}],
                                i=idxcols,
                                j='level',
                                sep='')

        mycases_wl.rename(lambda c : c.replace('_lvl', ''), axis='columns', inplace=True)

        mycases_wl = mycases_wl.reset_index()
        rencols = {col : utils_shap.colname_meta_infix(col) for col in ccc.LVL_TRAIN_COLS}

        mycases_wl.rename(columns=rencols, inplace=True)
        mycases_wl.rename(lambda c: "_".join(c.split("_")[::-1]), axis='columns', inplace=True)  # wide_to_long needs suffixes; so we rename columns "foo_bar" to "bar_foo"
       
        dd_profiles = pd.wide_to_long(mycases_wl, [utils_shap.META_INFIX[1:], utils_shap.SHAP_INFIX[1:]], ccc.INDEX_COLS + ['level'], "variable", "_", r"\w+")

        if write_cache:
            with open(cache_filepath, 'wb') as handle:
                pickle.dump(dd_profiles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if y_axis == "geopotential_altitude":
        df_lvldefs = pd.read_csv(os.path.join("data", "netcdf_raw", "era5_model_level_definitions.csv"))
        df_lvldefs = df_lvldefs[["n", "geopotential_altitude"]]
        df_lvldefs["geopotential_altitude"] = df_lvldefs["geopotential_altitude"].round().astype(np.int32)
        
        dd_profiles.reset_index(drop=False, inplace=True)
        dd_profiles = dd_profiles.merge(df_lvldefs[["n", "geopotential_altitude"]], left_on="level", right_on="n", how="left")
        
        postfix = "_geopot"
    elif y_axis == "level":
        postfix = "_level"

    postfix += "_cl" + "-".join([str(c) for c in plot_clusters.keys()]) if len(plot_clusters) > 0 else ""
    postfix += "_" + "-".join(only_show_cols) if len(only_show_cols) > 0 else ""
    postfix += "" if write_cache else "_experimental"

    q05 = lambda x : x.quantile(0.05)
    q25 = lambda x : x.quantile(0.25)
    q75 = lambda x : x.quantile(0.75)
    q95 = lambda x : x.quantile(0.95)

    ggs = []

    dd_profiles.reset_index(drop=False, inplace=True)
    dd_profiles = dd_profiles[dd_profiles.variable.isin(ccc.LVL_TRAIN_COLS)]

    if len(only_show_cols) > 0:
        dd_profiles = dd_profiles[dd_profiles['variable'].isin(only_show_cols)]

    if len(plot_clusters) > 0:
        dd_profiles = dd_profiles[dd_profiles['cluster'].isin(plot_clusters.keys())]

    if color_palette is not None:
        palette = color_palette
    else:
        palette = sns.color_palette([CLUSTER_COLORS[col] for col in plot_clusters.keys()] if len(plot_clusters) > 0 else CLUSTER_COLORS)

    if ptype == "mult":
        print("Grouping unique profiles")
        dd_profiles['unique_profile'] = dd_profiles['longitude'].astype(str) + '_' + \
                                        dd_profiles['latitude'].astype(str) + '_' + \
                                        dd_profiles['year'].astype(str) + '_' + \
                                        dd_profiles['month'].astype(str) + '_' + \
                                        dd_profiles['day'].astype(str) + '_' + \
                                        dd_profiles['hour'].astype(str) + '_'

        if separate_clusters:
            clusters = dd_profiles["cluster"].unique()

            for clusternr in clusters:
                save_filepath = os.path.join(save_path, f"profiles_agg_{target_var}_{ptype}_cluster_{clusternr}{postfix}") if save_path != "" else ""
                ggs.append(plot_many_profiles_internal_mult(dd_profiles, target_var, y_axis, palette, plot_clusters, clusternr=clusternr, save_filepath=save_filepath))
        else:
            save_filepath = os.path.join(save_path, f"profiles_agg_{target_var}_{ptype}{postfix}") if save_path != "" else ""
            ggs.append(plot_many_profiles_internal_mult(dd_profiles, target_var, y_axis, palette, plot_clusters, save_filepath=save_filepath))
            
    elif ptype in ["q50", "q90"]:
        aggs = ["median"]

        if ptype == "q90":
            aggs += [('qlow', q05), ('qhigh', q95)]
        elif ptype == "q50":
            aggs += [('qlow', q25), ('qhigh', q75)]

        groupbycols = ["variable", y_axis]

        if "cluster" in dd_profiles.columns:
            groupbycols.append("cluster")

        dd_profiles_agg = dd_profiles.groupby(groupbycols).aggregate(aggs)

        dd_profiles_agg.reset_index(drop=False, inplace=True)
        dd_profiles_agg['type'] = [ccc.VARTYPE_LOOKUP[x] for x in dd_profiles_agg['variable'].values]

        dd_profiles_agg.columns = ['_'.join(col).rstrip('_') for col in dd_profiles_agg.columns]  # MultiIndex columns get flattened

        dd_profiles_agg.query(f'type != "{ccc.VariableType.SETTING.name}" and type != "{ccc.VariableType.PREDICTION.name}"', inplace=True)

        if separate_clusters:
            clusters = dd_profiles_agg["cluster"].unique()

            for clusternr in clusters:
                save_filepath = os.path.join(save_path, f"profiles_agg_{target_var}_{ptype}_cluster_{clusternr}{postfix}_geopotheight") if save_path != "" else ""
                ggs.append(plot_many_profiles_internal_agg(dd_profiles_agg, target_var, y_axis, palette, plot_clusters, clusternr=clusternr, save_filepath=save_filepath))
        else:
            save_filepath = os.path.join(save_path, f"profiles_agg_{target_var}_{ptype}{postfix}_geopotheight") if save_path != "" else ""
            ggs.append(plot_many_profiles_internal_agg(dd_profiles_agg, target_var, y_axis, palette, plot_clusters, save_filepath=save_filepath))

    for g in ggs:    
        g.fig.clf()