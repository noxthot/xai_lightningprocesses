import glob
import os

import rasterio as rio

from pathlib import Path
from rasterio.merge import merge

input_path = 'input'
output_path = Path('output')

output_path.mkdir(parents=True, exist_ok=True)

raster_files = glob.glob(os.path.join(input_path), '*.tif')

raster_to_mosiac = []

for p in raster_files:
    raster = rio.open(p)
    raster_to_mosiac.append(raster)

mosaic, output = merge(raster_to_mosiac)

output_meta = raster.meta.copy()
output_meta.update(
    {"driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": output,
    })

with rio.open(os.path.join(output_path, "mosaic_output.tif"), "w", **output_meta) as m:
    m.write(mosaic)

