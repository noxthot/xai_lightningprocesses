import ccc

import datetime
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

if __name__ == '__main__':
    PATH_DATA_SOURCE = os.path.join('.', 'data', 'netcdf_raw')
    ls_era_files = glob.glob(os.path.join(PATH_DATA_SOURCE, "era5", "ERA5_ml_2*.nc"))
    dera = xr.open_mfdataset(ls_era_files, parallel=True)

    statvalsdf = pd.DataFrame(columns=['data_var', 'min', 'max', 'mean', 'std'])

    for varname in list(dera.coords) + list(dera.data_vars):
        print(f"Collect info for {varname}")

        if varname == "time":
            tdiff = (pd.DatetimeIndex(dera[varname].to_series()).date - ccc.REF_DATE) / datetime.timedelta(days=1)
            dayofyear = np.mod(tdiff, 365.2425).astype(np.float32)
            dmin = dayofyear.min()
            dmax = dayofyear.max()
            dmean = dayofyear.mean()
            dstd = dayofyear.std()
        else:
            dmin = dera[varname].min().compute().values
            dmax = dera[varname].max().compute().values
            dmean = dera[varname].mean().compute().values
            dstd = dera[varname].std().compute().values

        statvalsdf = statvalsdf.append({'data_var': varname, 'min': dmin, 'max': dmax, 'mean': dmean, 'std': dstd},
                                         ignore_index=True)

    statvalsdf.to_csv(os.path.join(ccc.DATASTATS_PATH, 'statsvals.csv'))

    print("FIN.", flush=True)
