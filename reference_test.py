import os
import json
import numpy as np
import pandas as pd
import sklearn.metrics as scores

THRTYPE = "calibration" # "fscore"

DATA_PATH = "data/models/targetmode_1/2022_02_22__reference"
FILE = os.path.join(DATA_PATH, "test_predictions.parquet")
FILE_VAL = os.path.join(DATA_PATH, "val_predictions.parquet")

print("Find opt_threshold on val scores", flush=True)
val_df = pd.read_parquet(FILE_VAL)
val_df.rename(columns={"flash": "target", "fit": "output"}, inplace=True)

pred_score = val_df["output"]

if THRTYPE == "fscore":
    ofile = "test_scores.json"
    
    roc_auc = scores.roc_auc_score(val_df["target"], pred_score, multi_class="ovr")

    precisions, recalls, prc_thresholds = scores.precision_recall_curve(val_df["target"], pred_score)
    prc_auc = scores.auc(recalls, precisions)

    fscores = (2 * precisions * recalls) / (precisions + recalls)
    opt_threshold = prc_thresholds[np.argmax(fscores)]
elif THRTYPE == "calibration":
    ofile = "test_scores_calibration.json"

    expect = val_df['target'].mean()
    opt_threshold = np.quantile(pred_score, 1 - expect)

print("Compute test scores", flush=True)
return_df = pd.read_parquet(FILE)
return_df.rename(columns={"flash": "target", "fit": "output"}, inplace=True)

pred_score = return_df["output"]

roc_auc = scores.roc_auc_score(return_df["target"], pred_score, multi_class="ovr")

precisions, recalls, prc_thresholds = scores.precision_recall_curve(return_df["target"], pred_score)
prc_auc = scores.auc(recalls, precisions)
pred_class = np.where(pred_score > opt_threshold, 1, 0)

class_rep = scores.classification_report(return_df["target"], pred_class)
conf_mat = scores.confusion_matrix(return_df["target"], pred_class)

test_scores = {
    "class_rep": class_rep,
    "conf_mat": np.array2string(conf_mat),
    "opt_threshold": opt_threshold,
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

print('Area under PRC: {:.6f}'.format(prc_auc), flush=True)
print('Area under ROC: {:.6f}'.format(roc_auc), flush=True)
print(f'Optimal threshold: \n{opt_threshold}', flush=True)
print(f'Classification report: \n{class_rep}', flush=True)
print(f'Confusion matrix report: \n{conf_mat}', flush=True)
print('Accuracy: {:.6f}'.format(test_scores["accuracy"]), flush=True)
print('False-alarm-rate: {:.6f}'.format(test_scores["false_alarm_rate"]), flush=True)
print('True-negative-rate: {:.6f}'.format(test_scores["true_negative_rate"]), flush=True)
print('True-positive-rate: {:.6f}'.format(test_scores["true_positive_rate"]), flush=True)
print('Critical-success-index: {:.6f}'.format(test_scores["critical_success_index"]), flush=True)
print('Matthews correlation coefficient: {:.6f}'.format(test_scores["mcc"]), flush=True)

json_test_scores = json.dumps(test_scores, indent=4)
json_file = open(os.path.join(DATA_PATH, ofile), 'w')
json_file.write(json_test_scores)
json_file.close()
