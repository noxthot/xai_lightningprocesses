import numpy as np
import xarray as xr
import os

PATH_DATA_SOURCE = os.path.join('.', 'data', 'netcdf_raw')

LABELS = {
    'FL':  'Flatlands',
    'HA':  'High Alps',
    'NMR': 'Northern Alpine Rim',
    'SMR': 'Southern Alpine Rim',
}

def get_subdomains():
    ds = xr.open_dataset(os.path.join(PATH_DATA_SOURCE, 'ERA5_altitude.nc'))
    df = ds.to_dataframe()
    df.reset_index(inplace=True)
    df['subdomain'] = np.select(
        [
            (df['altitude'] >= 1200),
            (df['altitude'] < 1200) & (df['altitude'] >= 500) & (df['longitude'] < 12) & (df['latitude'] >= 47),
            (df['altitude'] < 1200) & (df['altitude'] >= 500) & (df['longitude'] >= 12) & (df['latitude'] >= 47.5),
            (df['altitude'] < 1200) & (df['altitude'] >= 500) & (df['longitude'] < 12) & (df['latitude'] < 47),
            (df['altitude'] < 1200) & (df['altitude'] >= 500) & (df['longitude'] >= 12) & (df['latitude'] < 47.5),
            (df['altitude'] < 500),
        ],
        [
            'HA',
            'NMR',
            'NMR',
            'SMR',
            'SMR',
            'FL',
        ]
    )
    return df