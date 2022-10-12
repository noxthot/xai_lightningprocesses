import os
import torch

import numpy as np
import pandas as pd
import pyspark
import sklearn.metrics as scores
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from petastorm import TransformSpec
from petastorm.spark import SparkDatasetConverter
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

import ccc
import utils_shap


SPARK_CACHE_DIR = "file://" + os.path.join(os.getcwd(), "tmp_spark_cache")
SPARK_CORES = 8
SPARK_RAM_GB = 20
SPARK_MAX_RESULT_SIZE_GB = 8


def getsparksession():
    spark = SparkSession \
        .builder \
        .master(f"local[{SPARK_CORES}]") \
        .appName("spark mlvapto") \
        .config("spark.driver.memory", str(SPARK_RAM_GB) + "g") \
        .config("spark.driver.maxResultSize", str(SPARK_MAX_RESULT_SIZE_GB) + "g") \
        .config(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, SPARK_CACHE_DIR) \
        .getOrCreate()


    return spark


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, dropout_p):
        super(Classifier, self).__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(input_dim, hidden_layers[0])])
        self.hidden_layers.extend([nn.Linear(hidden_layers[idx - 1], hidden_layers[idx]) for idx in range(1, len(hidden_layers))])
        
        self.outputlayer = nn.Linear(hidden_layers[-1], output_dim)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        for hlayer in self.hidden_layers:
            x = F.leaky_relu(hlayer(x))
            x = self.dropout(x)

        x = self.outputlayer(x)

        return x


def initialize_weights(m):
  if isinstance(m, nn.Linear):
      nn.init.kaiming_normal_(m.weight.data, a=0.01, nonlinearity='leaky_relu')
      nn.init.constant_(m.bias.data, 0)
  elif not (isinstance(m, nn.Dropout) or isinstance(m, nn.ModuleList) or isinstance(m, nn.Module)):
      raise Exception("Unknown layer for weight initialization")


def load_model(modeldir, device, epoch=None):
    modelpath = os.path.join(ccc.MODEL_ROOT_PATH, modeldir)
    files = [f for f in os.listdir(modelpath) if f.endswith(".pt")]

    model_of_choice = None

    if epoch is None:
        improved_models = [f for f in files if f.startswith("model_")]
        improved_models.sort(reverse=True)
        model_of_choice = improved_models[0]
    else: 
        epochstr = str(epoch).zfill(5)

        for modelfname in [f"model_{epochstr}.pt", f"_model_{epochstr}.pt"]:
            if modelfname in files:
                model_of_choice = modelfname
                break

    if model_of_choice is None:
        raise Exception(f"Model not found")

    modelfilepath = os.path.join(modelpath, model_of_choice)
    
    print(f"Loading model from {modelfilepath}")

    structure = torch.load(modelfilepath, device)
    model = Classifier(
                input_dim=structure['input_dim'],
                hidden_layers=structure['hidden_layers'],
                output_dim=structure['output_dim'],
                dropout_p=structure['dropout_p'],
            )

    model.load_state_dict(structure['model_state_dict'])
    model.to(device, non_blocking=True)

    return model, model_of_choice.replace(".pt", "")


def calc_pos_weights(target_mode, device):
    if target_mode == 1:
        weights = torch.tensor(ccc.DATA_TARGET_STATS["flash"]["0"] / ccc.DATA_TARGET_STATS["flash"]["1"])
    elif target_mode == 2:
        f0 = ccc.DATA_TARGET_STATS["flash_windowed_sum"]["0"]
        f1 = ccc.DATA_TARGET_STATS["flash_windowed_sum"]["1"]
        f2 = ccc.DATA_TARGET_STATS["flash_windowed_sum"]["2"]
        f3 = ccc.DATA_TARGET_STATS["flash_windowed_sum"]["3"]

        nr_events = f0 + f1 + f2 + f3

        weights = torch.tensor(
                    [
                        (nr_events - (f1 + f2 + f3)) / (f1 + f2 + f3),  # due to encoding, 1st output node equals 1 when sum in {1, 2, 3}
                        (nr_events - (f2 + f3)) / (f2 + f3),            # due to encoding, 2nd output node equals 1 when sum in {2, 3}
                        (nr_events - f3) / f3,
                    ])
    elif target_mode == 3:
        weights = torch.tensor(ccc.DATA_TARGET_STATS["flash_windowed_max"]["0"] / ccc.DATA_TARGET_STATS["flash_windowed_max"]["1"])
    else:
        raise Exception(f"target_mode {target_mode} not implemented")

    return weights.to(device)


def norm_minmax_col(pdseries, col):
    return (pdseries - ccc.DATA_VAR_STATS[col]['min']) / (ccc.DATA_VAR_STATS[col]['max'] - ccc.DATA_VAR_STATS[col]['min'])


def norm_meanstd_col(pdseries, col):
    return (pdseries - ccc.DATA_VAR_STATS[col]['mean']) / ccc.DATA_VAR_STATS[col]['std']


def norm_disabled_col(pdseries, _):
    return pdseries


def transformrow(pd_batch, norm_fun, use_train_cols, target_mode, return_cols):
    target_col = getTargetCol(target_mode)
    normed_cols = pd.DataFrame({colname: norm_fun(pd_batch[colname], colname.split("_")[0]) for colname in use_train_cols}, dtype=np.float32)

    pd_batch['features'] = normed_cols.to_numpy(dtype=np.float32).tolist()

    if target_mode in {1, 3}:
        pd_batch['label'] = pd_batch[target_col].map(lambda x: np.array([x], dtype=np.float32))
    else:
        pd_batch['label'] = getOrderedEncoding(pd_batch, target_col).tolist()

    droplabels = set(use_train_cols + [target_col]) - set(return_cols)
    pd_batch = pd_batch.drop(labels=droplabels, axis=1)

    pd_batch['features'] = pd_batch['features'].map(lambda x: np.array(x, dtype=np.float32))
    return pd_batch


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def test_one_epoch(model, device, test_dataloader_iter, steps_per_epoch, loss_criterion, target_mode, hook_name_list=[], threshold=None):
    model.eval()
    nr_data_rows = 0
    test_loss = 0
    return_dfs = []

    with torch.no_grad():
        for i in range(1, steps_per_epoch + 1):
            if i % ccc.LOG_BATCH_INTERVAL == 0:
                print(f"Val/Test Step {i} / {steps_per_epoch}")

            batch = next(test_dataloader_iter)
            data = batch['features'].to(device, non_blocking=True)
            target = batch['label'].to(device, non_blocking=True)

            nr_data_rows += len(data)

            output = model(data)
            loss = loss_criterion(output, target)
            test_loss += loss.item() * data.size(0)

            if target_mode in {1, 3}:
                output_dec = torch.sigmoid(output)[:, 0].tolist()
                target_dec = target[:, 0].tolist()
            elif target_mode == 2:
                output_dec = getClassFromOrderedEncoding(torch.sigmoid(output).cpu().numpy()).tolist()
                target_dec = getClassFromOrderedEncoding(target.cpu().numpy()).tolist()
            else:
                raise Exception(f"target_mode {target_mode} not known")

            return_dict = {
                            "output": output_dec,
                            "target": target_dec,
                            "longitude": batch["longitude"],
                            "latitude": batch["latitude"],
                            "hour": batch["hour"],
                            "day": batch["day"],
                            "month": batch["month"],
                            "year": batch["year"],
                          }

            for hook_name in hook_name_list:
                return_dict[hook_name + "_lrelu"] = [F.leaky_relu(row).cpu().numpy() for row in activation[hook_name]]

            return_dfs.append(pd.DataFrame(return_dict))

    test_loss /= nr_data_rows

    print('Validation Loss: {:.6f}'.format(test_loss), flush=True)

    return_df = pd.concat(return_dfs, ignore_index=True)

    pred_score = return_df["output"]

    if target_mode in {1, 3}:
        roc_auc = scores.roc_auc_score(return_df["target"], pred_score, multi_class="ovr")

        precisions, recalls, prc_thresholds = scores.precision_recall_curve(return_df["target"], pred_score)
        prc_auc = scores.auc(recalls, precisions)

        fscores = (2 * precisions * recalls) / (precisions + recalls)
        opt_threshold = prc_thresholds[np.argmax(fscores)]
        used_threshold = opt_threshold if threshold is None else threshold
        pred_class = np.where(pred_score > used_threshold, 1, 0)

        class_rep = scores.classification_report(return_df["target"], pred_class)
        conf_mat = scores.confusion_matrix(return_df["target"], pred_class)

        test_scores = {
            "class_rep": class_rep,
            "conf_mat": np.array2string(conf_mat),
            "opt_threshold": opt_threshold,
            "used_threshold": used_threshold,
            "loss": test_loss,
        }

        n = conf_mat.sum()
        TP = conf_mat[1, 1]
        TN = conf_mat[0, 0]
        FP = conf_mat[0, 1]
        FN = conf_mat[1, 0]

        mcc = scores.matthews_corrcoef(return_df["target"], pred_class)
        test_scores["prc_auc"] = prc_auc
        test_scores["roc_auc"] = roc_auc
        test_scores["accuracy"] = (TP + TN) / n
        test_scores["false_alarm_rate"] = FP / (FP + TP)
        test_scores["true_negative_rate"] = TN / (TN + FP)
        test_scores["true_positive_rate"] = TP / (FN + TP)
        test_scores["critical_success_index"] = TP / (TP + FN + FP)
        test_scores["mcc"] = mcc

        print('Loss: {:.6f}'.format(test_loss), flush=True)
        print('Area under PRC: {:.6f}'.format(prc_auc), flush=True)
        print('Area under ROC: {:.6f}'.format(roc_auc), flush=True)
        print(f'Optimal threshold: \n{opt_threshold}', flush=True)
        print(f'Used threshold: \n{used_threshold}', flush=True)
        print(f'Classification report: \n{class_rep}', flush=True)
        print(f'Confusion matrix report: \n{conf_mat}', flush=True)
        print('Accuracy: {:.6f}'.format(test_scores["accuracy"]), flush=True)
        print('False-alarm-rate: {:.6f}'.format(test_scores["false_alarm_rate"]), flush=True)
        print('True-negative-rate: {:.6f}'.format(test_scores["true_negative_rate"]), flush=True)
        print('True-positive-rate: {:.6f}'.format(test_scores["true_positive_rate"]), flush=True)
        print('Critical-success-index: {:.6f}'.format(test_scores["critical_success_index"]), flush=True)
        print('Matthews correlation coefficient: {:.6f}'.format(test_scores["mcc"]), flush=True)

    return test_loss, return_df, test_scores


def get_transform_spec(norm_fun, target_mode, use_train_cols=[], return_cols=[]):
    selected_fields = ["features", "label"]
    selected_fields += return_cols

    return TransformSpec(func=partial(transformrow, norm_fun=norm_fun, use_train_cols=use_train_cols, target_mode=target_mode, return_cols=return_cols), selected_fields=selected_fields)


def getnormfun(norm_fun_str):
    if norm_fun_str == "minmax":
        norm_fun = norm_minmax_col
    elif norm_fun_str == "meanstd":
        norm_fun = norm_meanstd_col
    elif norm_fun_str == "disabled":
        norm_fun = norm_disabled_col
    else:
        raise Exception(f"Normalization function {norm_fun_str} not implemented")

    return norm_fun


def get_testdf_spark(data_cfg, cols, limit_rows):
    spark = getsparksession()

    datamode = data_cfg['datamode']

    data_path = os.path.join(ccc.DATA_ROOT_PATH, f"datamode_{datamode}", data_cfg["datasource"])

    if datamode == 1:
        df_test = spark.read.parquet(os.path.join(data_path, 'test')).select(*cols)
    elif datamode in {2, 3}:
        df_test1 = spark.read.parquet(os.path.join(data_path, 'train'))
        df_test2 = spark.read.parquet(os.path.join(data_path, 'val'))
        
        df_test1 = df_test1.filter(df_test1["year"] == data_cfg["test_year"]).select(*cols)
        df_test2 = df_test2.filter(df_test2["year"] == data_cfg["test_year"]).select(*cols)

        df_test = df_test1.union(df_test2)
    else:
        raise Exception(f"DATA_MODE {datamode} unknown")
    
    if limit_rows is not None:
        df_test = df_test.limit(limit_rows)

    df_test = df_test.withColumn('features', lit(0))  # We need to init column here, otherwise data loader fails
    df_test = df_test.withColumn('label', lit(0))  # We need to init column here, otherwise data loader fails

    df_test = df_test.repartition(16)

    return df_test

def get_valdf_spark(data_cfg, cols, limit_rows):
    spark = getsparksession()

    datamode = data_cfg['datamode']

    data_path = os.path.join(ccc.DATA_ROOT_PATH, f"datamode_{datamode}", data_cfg["datasource"])

    if datamode == 1:
        raise Exception(f"DATA_MODE {datamode} not supported")
    elif datamode in {2, 3}:
        df_val = spark.read.parquet(os.path.join(data_path, 'val'))
        
        df_val = df_val.filter(df_val["year"] != data_cfg["test_year"]).select(*cols)
    else:
        raise Exception(f"DATA_MODE {datamode} unknown")
    
    if limit_rows is not None:
        df_val = df_val.limit(limit_rows)

    df_val = df_val.withColumn('features', lit(0))  # We need to init column here, otherwise data loader fails
    df_val = df_val.withColumn('label', lit(0))  # We need to init column here, otherwise data loader fails

    df_val = df_val.repartition(16)

    return df_val

def fit_to_batchsize(df):
    nr_avail_rows = df.count()
    points_lost = nr_avail_rows % ccc.BATCH_SIZE

    print(f"Dropping {points_lost} rows of dataset due to batch size.")

    return df.limit(nr_avail_rows - points_lost)


def getTargetCol(target_mode):
    if target_mode == 1:
        return "flash"
    elif target_mode == 2:
        return "flash_windowed_sum"
    elif target_mode == 3:
        return "flash_windowed_max"
    else:
        raise Exception(f"Target mode {target_mode} unknown")


def getOrderedEncoding(pdbatch, target_col):

    encarr = np.zeros((len(pdbatch), 3), dtype=np.float32)

    encarr[pdbatch[target_col] >= 1, 0] = 1
    encarr[pdbatch[target_col] >= 2, 1] = 1
    encarr[pdbatch[target_col] >= 3, 2] = 1

    return encarr


def getClassFromOrderedEncoding(encoded_array):
    return (encoded_array > 0.5).cumprod(axis=1).sum(axis=1)

def getVarRange(varname):
    start = ccc.DATA_VAR_STATS[varname]['min']
    step = ccc.DATA_VAR_STATS[varname]['delta']
    end = ccc.DATA_VAR_STATS[varname]['max'] + step
    return np.arange(start, end, step)


def joinDataframes(df1, df2, useMetaInfix=False, how="inner"):
    joincols = ccc.INDEX_COLS

    if useMetaInfix:
        joincols = [utils_shap.colname_meta_infix(c) for c in joincols]

    df1cols = set(df1.columns) - set(joincols)
    df2cols = set(df2.columns) - set(joincols)

    colsIntersect = df1cols.intersection(df2cols)

    if len(colsIntersect) > 0:
        print(f"WARNING: The following columns are available in both dataframes: {colsIntersect}.")

    if type(df1) == pyspark.sql.dataframe.DataFrame:
        df = df1.join(df2, on=joincols, how="left")
    else:
        df = pd.merge(df1, df2, on=joincols, how=how, validate="one_to_one", suffixes=("_left", "_right"))

    return df

def addCloudHeight(df):
    geoh_colnames = [f'geoh_lvl{l}' for l in range(64, 138)]

    geoh = df[geoh_colnames]
    output_df = geoh.sub(df['cbh'], axis='index').div(df['cth'] - df['cbh'], axis='index')
    output_df[output_df > 1] = np.nan
    output_df[output_df < 0] = np.nan
    output_df.rename(lambda c : c.replace('geoh', 'cloudscale'), axis='columns', inplace=True)
    
    return_df = pd.concat([df, output_df], axis=1)

    return return_df


def getOptThresholdFromVal(train_monitor, use_epoch=None):
    opt_threshold = float('inf')
    max_mcc = float('-inf')
    best_epoch = -1

    for scorebundle in train_monitor["val_scores"]:
        if use_epoch is not None:
            if use_epoch == scorebundle["epoch"]:
                opt_threshold = scorebundle["opt_threshold"]
                best_epoch = use_epoch
                break

        if scorebundle["mcc"] > max_mcc:
            max_mcc = scorebundle["mcc"]
            opt_threshold = scorebundle["opt_threshold"]
            best_epoch = scorebundle["epoch"]

    return opt_threshold, best_epoch


def getVeryConfidentThreshold(used_threshold):
    return (1 + used_threshold) / 2