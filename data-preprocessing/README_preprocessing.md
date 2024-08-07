# How to obtain and pre-process the ERA5 model level data

ERA5 data on model level are an essential ingredient of the study.
Here, we provide guidance through the most important steps to
obtain and preprocess the ERA5 model level data up to the point
where the data are in the right format for entering the `etl.py`.

## CDS retrieval

The Climate Data Store (CDS) is the operational service of the
Copernicus Climate Change Service (C3S). Its purpose is to enable
access to scientific data that monitor the state of the climate.
To access data from the CDS an account is mandatory, cf.
`https://cds.climate.copernicus.eu/api-how-to`. The model level data
used in our study is archived on tapes at the CDS
(`https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5#HowtodownloadERA5-4-DownloadERA5familydatathroughtheCDSAPI`).
Thus, one has to be patient while requesting a download. Our
personal experience says that a single CDS user is able to obtain
a single file within 24 hours. We requested the data in chunks, where
each chunk covers one month. E.g., in our case 10 years times 3 summer
month (June, July, August) requires patience of roughly 30 days.
Our script sending the request, `cds_retrival_ml.py`, can be called
as follows:

```bash
(meteorology_verticallearning) $ python cds_retrival_ml.py -y 2022 -m 06
```

## From grib to netcdf format

**grib** and **netcdf** are binary data formats common in the weather
community and beyond. While the ERA5 model level data is only accessible
in **grib**, we convert the data files to **netcdf** format, which we
consider to be a bit more handy. The conversion is performed using
**eccodes** (https://confluence.ecmwf.int/display/ECC/ecCodes+Home),
a package developed by the ECMWF which provides a set of tools for
decoding and encoding messages between **grib** and **netcdf**.
Here we need the `grib_to_netcdf` (https://confluence.ecmwf.int/display/ECC/grib_to_netcdf) tool:

```bash
(meteorology_verticallearning) $ grib_to_netcdf -o ERA5_ml_2022-06.nc ERA5_ml_2022-06.grib
```

The `etl.py` assumes to find the data in the **netcdf** format in `./data/netcdf_raw/era5` and `./data/netcdf_raw/era5sl`.

## Computation of vertical coordinates

In order to be able to interpret the results on physical
vertical coordinates one can derive geopotential height
and pressure using `compute_vert_coords.py`. This is the
script we used. As an alternative one can follow the
instructions on https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height.

# How to obtain ALDIS lightning location system

The ALDIS data, which are the second important source of data, can be downloaded from [[1]](#1).

The `etl.py` assumes to find the data in the **netcdf** format in `./data/netcdf_raw/flash`.

# How to get elevation data (optional)
The elevation data, sourced from the TanDEM-X project in geotiff format, is utilized solely for depicting topography within background layers of select figures.
This data is available for download from platforms such as [data.europe.eu](https://data.europa.eu/data/datasets/2846908f-74fa-4d64-95df-7bc14959ab42?locale=en) and is typically provided in tiled segments that require merging. The code for merging can be found in `merge_geotiffs.py`.

# References
<a id="1">[1]</a>
[1] Simon, T., Schulz, W., Ehrensperger, G., & Mayr, G. (2024). ALDIS cloud to ground lightning strike occurrence aggregated to spatiotemporal ERA5 cells (summer months 2010 to 2019) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13164463