import ccc
import utils

import os

import numpy as np

DATA_MODE = 3  # errors for others; but 3 exports a subset which can be used for 1-3.

if __name__ == '__main__':
    if DATA_MODE != 3:
        raise Exception(f"Not implemented for DATA_MODE {DATA_MODE}")

    print("Set up spark session.", flush=True)
    spark = utils.getsparksession()

    datamode_root = os.path.join(ccc.DATA_ROOT_PATH, f"datamode_{DATA_MODE}")

    data_dirs = os.listdir(datamode_root)
    data_dirs.sort(reverse=True)
    datadir = data_dirs[0]

    data_path = os.path.join(datamode_root, datadir)

    cols = ["flash", "flash_windowed_sum", "flash_windowed_max"]

    print("Read parquet and select columns.", flush=True)
    df_train = spark.read.parquet(os.path.join(data_path, 'train')).select(cols)
    df_val = spark.read.parquet(os.path.join(data_path, 'val')).select(cols)
    
    print("Merge train/val set and transform to pandas.", flush=True)
    df = df_train.union(df_val).toPandas().astype(np.int32)

    for col in cols:
        print(f"Calc {col} and write to CSV.", flush=True)
        df[col].value_counts().to_csv(os.path.join(ccc.DATASTATS_PATH, f"statsvals_{col}.csv"))

    print("FIN.", flush=True)
