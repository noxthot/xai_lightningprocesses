import ccc
import datetime
import glob
import os
import utils

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.model_selection import train_test_split


DATA_MODE = 3  # 1 = split into train/val/test partion by hour, year; 2 = split into train/val partition by year, hour; 3 = split into train/val partition by year and simday

LIMIT_NR_FILES = 0  # Limit number of read files; just for testing purposes. Leave to 0 for processing everything.

PATH_DATA_SOURCE = os.path.join('.', 'data', 'netcdf_raw')
PATH_TMP_DATA_TARGET = os.path.join('.', 'tmpdata', "dataparquet_" + datetime.date.today().strftime("%Y_%m_%d"))
PATH_DATA_TARGET = os.path.join('.', 'data', 'data_processed', f'datamode_{DATA_MODE}',
                                "dataparquet_" + datetime.date.today().strftime("%Y_%m_%d"))

CONVERT_CDF_TO_PARQUET = True  # Should be True except for debugging purposes
SPLIT_PARQUET_TRAIN_VAL = True  # Should be True except for debugging purposes

if __name__ == '__main__':
    if DATA_MODE not in {1, 2, 3}:
        raise Exception(f"DATA_MODE {DATA_MODE} not implemented")

    if DATA_MODE == 1:
        partition_cols = ["hour", "year"]
        split_vals = [0.8, 0.15, 0.05]
    elif DATA_MODE == 2:
        partition_cols = ["year", "hour"]
        split_vals = [0.8, 0.2]
    elif DATA_MODE == 3:
        partition_cols = ["year"]
        split_vals = [0.8, 0.2]

    if CONVERT_CDF_TO_PARQUET:
        ls_era_files = glob.glob(os.path.join(PATH_DATA_SOURCE, "era5", "ERA5_ml_2*.nc"))
        ls_flash_files = glob.glob(os.path.join(PATH_DATA_SOURCE, "flash", "flash_2*.nc"))
        ls_coord_files = glob.glob(os.path.join(PATH_DATA_SOURCE, "era5", "ERA5_ml_vertical_coords_2*.nc"))
        ls_erasl_files = glob.glob(os.path.join(PATH_DATA_SOURCE, "era5sl", "*.nc"))

        ls_era_files.sort()
        ls_flash_files.sort()
        ls_coord_files.sort()
        ls_erasl_files.sort()

        if LIMIT_NR_FILES > 0:
            ls_era_files = ls_era_files[:LIMIT_NR_FILES]
            ls_flash_files = ls_flash_files[:LIMIT_NR_FILES]
            ls_coord_files = ls_coord_files[:LIMIT_NR_FILES]
            ls_erasl_files = ls_erasl_files[:LIMIT_NR_FILES]

        # sanity check
        for erafile, flashfile, coordfile, slfile in zip(ls_era_files, ls_flash_files, ls_coord_files, ls_erasl_files):
            str_date = erafile[-10:-3]

            if str_date != flashfile[-10:-3]:
                raise Exception(f"Files do not match: {erafile}, {flashfile}.")

            if str_date != coordfile[-10:-3]:
                raise Exception(f"Files do not match: {erafile}, {coordfile}.")

            if str_date != slfile[-10:-3]:
                raise Exception(f"Files do not match: {erafile}, {slfile}.")

        for idx, files in enumerate(zip(ls_era_files, ls_flash_files, ls_coord_files, ls_erasl_files)):
            erafile, flashfile, coordfile, slfile = files
            print(f"READING FILES: {erafile}, {flashfile}, {coordfile}, {slfile} ({idx + 1} of {len(ls_era_files)})", flush=True)

            dera = xr.open_mfdataset(erafile, parallel=True).persist()
            dflash = xr.open_mfdataset(flashfile, parallel=True).persist()
            dcoord = xr.open_mfdataset(coordfile, parallel=True).persist()
            dsl = xr.open_mfdataset(slfile, parallel=True).persist()

            dtopo = dcoord.sel(level=137.0).rename_vars({"geoh": "topography"})[["topography"]]

            print("MERGING", flush=True)
            mergedds = xr.merge([dera, dflash, dtopo, dcoord, dsl], join="inner")  # Merge along latitude, longitude and time
            mergeddsstack = mergedds.stack(latlongtime=("latitude", "time", "longitude"))

            del dera
            del dflash
            del dcoord 
            del mergedds

            print("TRANSFORMING", flush=True)
            df = pd.DataFrame()

            for c in ccc.LVL_COLS:
                colsaslist = mergeddsstack[c].transpose().values.astype(np.float32).tolist()
                colnames = [f"{c}_lvl{idx + 64}" for idx in range(74)]

                df[colnames] = pd.DataFrame(colsaslist).astype(np.float32)

            for c in ["longitude", "latitude", "flash", "topography", "t2m", "cbh", "cswc2040", "cth", "cape", "cp", "wvc1020", "ishf", "mcc", "tcslw"]:
                df[c] = mergeddsstack[c].values.astype(np.float32)

            tmpdatetime = pd.DatetimeIndex(mergeddsstack['time'].values)

            tdiff = tmpdatetime.date - ccc.REF_DATE
            daysim = (tdiff - datetime.timedelta(hours=ccc.START_DAY_HOUR)) / datetime.timedelta(days=1)

            df["hour"] = tmpdatetime.hour.values.astype(np.int32)
            df["day"] = tmpdatetime.day.values.astype(np.int32)
            df["dayofyear"] = np.mod(tdiff / datetime.timedelta(days=1), 365.2425).astype(np.float32)
            df["daysim"] = daysim.astype(np.int32)
            df["month"] = tmpdatetime.month.values.astype(np.int32)
            df["year"] = tmpdatetime.year.values.astype(np.int32)

            del mergeddsstack
            del tmpdatetime

            df.sort_values(by=["month", "day", "hour"], inplace=True)
            winsum = lambda x: x.rolling(window=3, min_periods=2, center=True).sum()
            winmax = lambda x: x.rolling(window=3, min_periods=2, center=True).max()

            df["flash_windowed_sum"] = df.groupby(["longitude", "latitude", "year"]).flash.apply(winsum).astype(np.int32)
            df["flash_windowed_max"] = df.groupby(["longitude", "latitude", "year"]).flash.apply(winmax).astype(np.int32)

            print("CONVERTING TO DASK", flush=True)
            dfdask = dd.from_pandas(df, chunksize=1000000)
            del df

            print("WRITING TO PARQUET", flush=True)
            dfdask.to_parquet(PATH_TMP_DATA_TARGET, partition_on=partition_cols, append=(idx != 0), write_index=False, engine="pyarrow", compression="snappy")
            del dfdask

        print("FINISHED CREATING TEMP DATA", flush=True)

    if SPLIT_PARQUET_TRAIN_VAL:
        print("SETTING UP SPARK SESSION", flush=True)
        spark = utils.getsparksession()

        sc = spark.sparkContext

        print("READING TEMP DATA", flush=True)
        sparkdf = spark.read.parquet(PATH_TMP_DATA_TARGET)

        print("SPLITTING DATASET", flush=True)
        if DATA_MODE == 3:
            distinct_daysims = [x.daysim for x in sparkdf.select('daysim').distinct().collect()]
            daysims_train, daysims_val = train_test_split(distinct_daysims, test_size=split_vals[1], random_state=ccc.SEED_RANDOM)
            dfs = []
            dfs.append(sparkdf.filter(sparkdf.daysim.isin(daysims_train)))
            dfs.append(sparkdf.filter(sparkdf.daysim.isin(daysims_val)))
        else:
            dfs = sparkdf.randomSplit(split_vals, seed=ccc.SEED_RANDOM)

        print("REPARTITIONING", flush=True)
        for df in dfs:
            df.repartition(1)

        datasets = {'train': dfs[0], 'val': dfs[1]}

        if DATA_MODE == 1:
            datasets["test"] = dfs[2]

        for key in datasets:
            print(f"WRITING {key.upper()} TO PARQUET", flush=True)
            datasets[key].write.partitionBy(*partition_cols) \
                .mode("overwrite") \
                .parquet(os.path.join(PATH_DATA_TARGET, key))
    
        datasets.clear()

    print("FIN.", flush=True)
