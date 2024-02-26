import numpy as np
import pandas as pd
import os

PATH_DATA_SOURCE = os.path.join('resources')

LABELS = {
    'FL':  'Flatlands',
    'HA':  'High Alps',
    'NMR': 'Northern Alpine Rim',
    'SMR': 'Southern Alpine Rim',
}

def get_subdomains():
    df = pd.read_csv(os.path.join(PATH_DATA_SOURCE, 'era5_altitudes.csv'))
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