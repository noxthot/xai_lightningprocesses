#!/usr/bin/env python3
#
# compute_vert_coords.py --- computes geopotential on model levels
# Copyright (C) 2020  Deborah Morgenstern, Johannes Horak
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# OneLineDesc   : Computes final_geopotential on model levels
#
# Description   : Computes final_geopotential on model levels
#                 for era5 data on netCDF4.
#                 All formulas are taken and referenced to
#                 IFS Documentation Cy41r2, Part III: Dynamics
#                 and numerical procedures by ECMWF May 2015
#                 https://www.ecmwf.int/en/elibrary/16647-part-iii-dynamics-and-numerical-procedures
#
# Inputs       : levDef.csv     csv-file containing level
#                               definitions a and b
#                               for download use file:
#                               download_model_definition.py
#              : ml.nc          a netCDF file containing
#                               t and q on model level
#                               serves as basis for cutting
#                               other data
#              : sp.nc          a netCDF file containing
#                               surface pressure on pressure level
#                               or lnsp
#              : z.nc           a netCDF file containing
#                               surface geopotential for one day
#                               on surface level
# Returns      : outfile_name.nc     geopotential, geopotential height
#                               and pressure on model level
#                               appended to existing outfile_name.nc

import pandas as pd
import xarray as xa
import numpy as np
import netCDF4
from netCDF4 import date2num
from optparse import OptionParser
import os
import time as tim

# ---------------------------------------------------------------------------------
# Input files are parsed. Give path relative to this script.
# ---------------------------------------------------------------------------------
parser = OptionParser()
parser.add_option("-l", "--levDef", dest="LEVEL_DEFINITION",
                  help="*.csv-file where ERA5 model level definition is stored")
parser.add_option("-z", "--geopotential", dest="GEOPOTENTIAL_DATA",
                  help="geopotential at surface (z) on pressure level as *.nc file")
parser.add_option("-m", "--modelLevel", dest="ML_DATA",
                  help="variables temperature (t) and humidity (q) on model levels as *.nc file")
parser.add_option("-s", "--surfaceLevel", dest="SURFACE_DATA",
                  help="variable surface pressure (sp) or its logarithm (lnsp)"
                       "on surface level as *.nc file")
parser.add_option("-o", "--outfile_name", dest="OUTFILE",
                  help="outfile_name where new variables are stored in (*.nc)."
                       "If file is existing variables are added, if file does not"
                       "exist a new nc-file is created containing only geopotential height"
                       "and pressure. A new file is recommended for large data.")
(options, args) = parser.parse_args()

LEVEL_DEFINITION = options.LEVEL_DEFINITION
ML_DATA = options.ML_DATA
SURFACE_DATA = options.SURFACE_DATA
GEOPOTENTIAL_DATA = options.GEOPOTENTIAL_DATA
OUTFILE = options.OUTFILE

# ---------------------------------------------------------------------------------
# Some informative prints
# ---------------------------------------------------------------------------------
time_begin = tim.time()

print("   ")
print("-------------")
print("compute_vert_coords.py Copyright (C) 2020  Deborah Morgenstern, Johannes Horak.")
print("This program comes with ABSOLUTELY NO WARRANTY.")
print("This is free software, and you are welcome to redistribute it")
print("under certain conditions; see <https://www.gnu.org/licenses/> for details.")
print("-------------")
print("Output for " + OUTFILE)
print("   ")
print("Script started at:            " + tim.strftime("%Y-%m-%d %H:%M:%S", tim.localtime()))


# ---------------------------------------------------------------------------------
# constants and definitions
# ---------------------------------------------------------------------------------

# Gas constants for the equation of state are taken from:
# Paul Markowski and Yvette Richardson 2016: Mesoscale Meteorology in Mitlatitudes
# Wiley-Blackwell, Royal Meteorological Society, ISBN: 978-0-470-74213-6
R_DRY = 287.04  # dry air,     J K**-1 kg**-1
R_VAPOUR = 461.51  # water vapor, J K**-1 kg**-1

# gravity taken from: http://glossary.ametsoc.org/wiki/Geopotential_height
GRAVITY = 9.80665  # globally averaged acceleration of gravity at sea level,  m s**-2


# ----------------------------------------------
# functions regarding input files
# ----------------------------------------------
def getModelLevelDefinition(path_to_levDef):
    # Definition of model levels a and b is loaded.
    # path_to_levDef    : String. Path to file (*.csv) containing level definitions

    levDef = pd.read_csv(path_to_levDef, sep='\t')
    a = levDef['a [Pa]']
    b = levDef['b']

    return a, b


def loadAlignCombineERA5(path_model_level_t_q, path_surface_sp, path_surface_geopotential):
    # This function loads all three netCDF files,
    # calls functions to do temporal and spatial alignment,
    # combines data to one dataFrame
    #
    # path_model_level_t_q      : string. path, where netCDF containing t and q is stored
    # path_surface_ln           : string. path, where netCDF containing lnsp is stored
    # path_surface_geopotential : string. path, where netCDF containing z is stored
    # return                    : one dataFrame containing only needed data

    tq_data  = xa.open_dataset(path_model_level_t_q)
    sp_data  = xa.open_dataset(path_surface_sp)
    z_data   = xa.open_dataset(path_surface_geopotential)

    # calculate sp from lnsp if neccessary (added 2021-05-06)
    if "lnsp" in list(sp_data.keys()) :
        sp_data = sp_data.assign(sp = np.exp(sp_data.lnsp))
        sp_data = sp_data.drop_vars("lnsp")

    # need only one z timestep (added 2021-05-06)
    if len(z_data.z.shape) > 2 :
        z_data = z_data.isel(time = 1, drop = True)
        
    # cut data to same temporal resolution
    sp_data = alignData(data_in_desired_shape=tq_data, data_to_be_subsetted=sp_data, dimension="time")

    # cut data to same spatial resolution
    sp_data = alignData(data_in_desired_shape=tq_data, data_to_be_subsetted=sp_data, dimension="longitude")
    sp_data = alignData(data_in_desired_shape=tq_data, data_to_be_subsetted=sp_data, dimension="latitude")
    z_data = alignData(data_in_desired_shape=tq_data, data_to_be_subsetted=z_data, dimension="longitude")
    z_data = alignData(data_in_desired_shape=tq_data, data_to_be_subsetted=z_data, dimension="latitude")

    # check if all data is in desired shape
    checkShape(model_level_data=tq_data, pressure_data=sp_data, z_data=z_data)

    # combine data to one dataFrame
    era5_combined = xa.merge([tq_data, sp_data, z_data], join="exact")

    return era5_combined


def alignData(data_in_desired_shape, data_to_be_subsetted, dimension):
    # data_to_be_subsetted is subsetted to the temporal domain of
    # data_in_desired_shape.
    # data_in_desired_shape : loaded netcdf dataframe containing data with the wanted dimensions
    # data_to_be_subsetted  : loaded netcdf dataframe containing data with other dimensions

    # check if they are already the same
    if data_in_desired_shape[dimension].shape == data_to_be_subsetted[dimension].shape:
        aligned = data_to_be_subsetted
    else:
        first = data_in_desired_shape[dimension][0].values
        last = data_in_desired_shape[dimension][-1].values

        if dimension == "time":
            aligned = data_to_be_subsetted.sel(time=slice(*[first, last]))
        elif dimension == "longitude":
            aligned = data_to_be_subsetted.sel(longitude=slice(*[first, last]))
        elif dimension == "latitude":
            aligned = data_to_be_subsetted.sel(latitude=slice(*[first, last]))

    return aligned


def checkShape(model_level_data, pressure_data, z_data):
    # make sure that input data has correct shape.
    # if not, an exeption is issued

    shape_t = model_level_data.t.shape
    shape_q = model_level_data.q.shape
    same1 = True if (shape_t == shape_q) else False

    wanted_p_shape = (
        model_level_data.time.shape[0], model_level_data.latitude.shape[0], model_level_data.longitude.shape[0])
    shape_p = pressure_data.sp.shape
    same2 = True if (wanted_p_shape == shape_p) else False

    wanted_z_shape = (model_level_data.latitude.shape[0], model_level_data.longitude.shape[0])
    shape_z = z_data.z.shape
    same3 = True if (wanted_z_shape == shape_z) else False

    if not (same1 and same2 and same3):
        raise Exception("STOP: Input variables do not have the same shape. "
                        "Check your data.")
    return


# ----------------------------------------------
# functions regarding calculations
# ----------------------------------------------
def getGeopotential(min_level):
    # calculate final_geopotential at full model level k as given by equation 2.22
    # min_level     : integer, highest level for which final_geopotential is calculated.
    # return        : list containing final_geopotential on all levels, at all times, for each cell.

    # Index for saving 73...0 (74 levels, python starts counting at zero)
    idx = ERA5_MAX_LEVEL - ERA5_MIN_LEVEL

    # Calculate geopotential iteratively, starting at surface
    for k in range(ERA5_MAX_LEVEL, min_level - 1, -1):

        # ----------
        #  get quantities that are needed for formula

        # at surface, final_geopotential from ERA5 is used
        if k == ERA5_MAX_LEVEL:
            phi_k_plus_half = era5_ds.z.values

        # for all other levels, final_geopotential at half level below has to be calculated
        else:
            phi_k_plus_half = getPhiAtKPlusHalf(k, phi_k_plus_one_and_a_half)

        alpha_k, p_k = getAlphaAndPressure(k)
        t_virtual_k = getVirtualTemperature(k)

        # ----------
        # formula 2.22
        phi_k = phi_k_plus_half + alpha_k * R_DRY * t_virtual_k

        # ----------
        # safe result
        geopotential_height[:, idx, :, :] = phi_k / GRAVITY
        pressure[:, idx, :, :] = p_k

        # ----------
        # for next round
        idx = idx - 1
        phi_k_plus_one_and_a_half = phi_k_plus_half

    return geopotential_height, pressure


def getAlphaAndPressure(k):
    # calculate coefficient alpha_k as given by equation 2.23
    # calculate final_pressure p_k at full level (eq. 2.11 b)
    # k         : number of full level
    # return    : alpha for full level k
    #             final_pressure for full level k

    # final_pressure at half levels above and below
    p_k_minus_half = getPressureAtKMinusHalf(k)
    p_k_plus_half = getPressureAtKPlusHalf(k)

    if k == 1:
        alpha_k = np.log(2)
    else:
        delta_p_k = p_k_plus_half - p_k_minus_half
        # formula 2.23
        alpha_k = 1 - (p_k_minus_half / delta_p_k) * np.log(p_k_plus_half / p_k_minus_half)

    # formula 2.11 b
    p_k = 0.5 * (p_k_minus_half + p_k_plus_half)

    return alpha_k, p_k


def getPressureAtKMinusHalf(k):
    # calculate final_pressure at half level above full level
    # k_minus_half is the same as k_plus_half of the subsequent (higher) full level
    # k         : number of subsequent lower full level
    # return    : final_pressure at half level above k

    a = all_a[k - 1]
    b = all_b[k - 1]

    return a + b * era5_ds.sp.values


def getPressureAtKPlusHalf(k):
    # calculate final_pressure at half level below full level
    # k         : number of subsequent higher full level
    # return    : final_pressure at half level below k

    # for level 137 k_plus_half is just surface final_pressure
    if k == ERA5_MAX_LEVEL:
        return era5_ds.sp.values
    else:
        a = all_a[k]
        b = all_b[k]
        return a + b * era5_ds.sp.values


def getVirtualTemperature(k):
    # calculate virtual temperature at full level k.
    # temperature tht a sample of dry air must have so that its density is
    # equivalent to a sample of moist air at the same final_pressure.
    # k         : number of level
    # return    : virtual temperature at level k

    t = era5_ds.sel(level=k)['t'].values  # temperature at full level k
    q = era5_ds.sel(level=k)['q'].values  # specific humidity at full level k
    # formula 2.2 b
    t_virtual_k = t * (1 + (R_VAPOUR / R_DRY - 1.0) * q)
    return t_virtual_k


def getPhiAtKPlusHalf(k, phi_k_plus_one_and_a_half):
    # calculate final_geopotential (phi) at half level below.
    # based on formula 2.21
    # instead of summing all geopotentials up until current k (slow)
    # previous final_geopotential (at k_plus_one_and_a_half for current k)
    # is used so that only difference between last and current level
    # needs to be calculated (fast)
    # 2.21: phi_k_plus_half = phi_sfc + sum_{j = k+1}^{NLEV}
    #                         (R_DRY * t_virtual_j * ln(p_j_plus_half / p_j_minus_half))
    # here: phi_k_plus_half = phi_k_plus_one_and_a_half +
    #                          R_DRY * t_virtual_k_plus_one * ln(p_k_plus_one_and_a_half / p_k_plus_half)
    #
    # k         : level below which final_geopotential is calculated
    # return    : final_geopotential at half level below k

    # p_k_plus_half = getPressureAtKMinusHalf(k + 1)
    p_k_plus_half = getPressureAtKPlusHalf(k)
    p_k_plus_one_and_a_half = getPressureAtKPlusHalf(k + 1)
    t_virtual_k_plus_one = getVirtualTemperature(k + 1)
    phi_k_plus_half = phi_k_plus_one_and_a_half + R_DRY * t_virtual_k_plus_one * np.log(
        p_k_plus_one_and_a_half / p_k_plus_half)

    return phi_k_plus_half


# ----------------------------------------------
# function to store result as netCDF
# ----------------------------------------------
def saveToNetCDF(final_geopotential_height, final_pressure, outfile_name, ds):
    # store calculated variables to a netCDF file.
    # if outfile_name exists, variables are added (variables must not exist in that file)
    # else, a new netCDF file is created based on the dimensions of ds.
    # final_geopotential_height   : calculated final_geopotential_height
    # final_pressure              : calculated final_pressure
    # outfile_name                : path, to which netCDF file the new variables are added
    # ds                          : loaded netCDF file that giives basis for dimensions of new netCDF file


    # open existing netCDF file
    if os.path.isfile(outfile_name):
        print("Appending to netCDF-file at:   " + tim.strftime("%Y-%m-%d %H:%M:%S", tim.localtime()))
        # open netCDF file
        # mode = a means append (w = overwrite, r = read only)
        # f4 = 32-bit floating point, i4 = integer 32-bit, i2 = 16-bit signed integer, u2 = unsigned 16-bit integer
        outfile = netCDF4.Dataset(outfile_name, mode='a')

    # create new netCDF file
    else:
        print("Creating new netCDF-file at:  " + tim.strftime("%Y-%m-%d %H:%M:%S", tim.localtime()))
        outfile = netCDF4.Dataset(outfile_name, mode='w')

        # General Attributes
        outfile.description = "Created by compute_vert_coords.py to derive geopotential and pressure on model levels of ERA5 data."
        outfile.history = "Created" + tim.strftime("%Y-%m-%d %H:%M:%S", tim.localtime())

        # create Dimensions
        outfile.createDimension("time", ds.dims["time"])
        outfile.createDimension("level", ds.dims["level"])
        outfile.createDimension("latitude", ds.dims["latitude"])
        outfile.createDimension("longitude", ds.dims["longitude"])

        # create coordinate variables
        # arguments: name of variable, variable type, dimension (use previously defined names here)
        times = outfile.createVariable("time", "f4", ("time",))
        times.units = "hours since 1900-01-01 00:00:00.0"
        times.calendar = "gregorian"
        times.long_name = "time"
        times_pandas_timestamp = pd.to_datetime(ds.time.values)
        times_datetime = times_pandas_timestamp.to_pydatetime()
        times_numeric = date2num(times_datetime, units=times.units, calendar=times.calendar)
        times[:] = times_numeric

        levels = outfile.createVariable("level", "i2", ("level",))
        levels.long_name = "vertical model level"
        levels[:] = ds.level.values

        latitudes = outfile.createVariable("latitude", "f4", ("latitude",))
        latitudes.units = "degrees north"
        latitudes[:] = ds.latitude.values

        longitudes = outfile.createVariable("longitude", "f4", ("longitude",))
        longitudes.units = "degrees east"
        longitudes[:] = ds.longitude.values

    # -----------
    # make sure that data are scaled using scale and offset. (Is default anyway...)
    outfile.set_auto_scale(True)

    # ------------
    # add variable final_geopotential height
    print("-------------")

    # writing to netcdf
    print("Create geoh at:               " + tim.strftime("%Y-%m-%d %H:%M:%S", tim.localtime()))
    geohnc = outfile.createVariable('geoh', "i2", ('time', 'level', 'latitude', 'longitude'))

    print("Add descriptions to geoh at:  " + tim.strftime("%Y-%m-%d %H:%M:%S", tim.localtime()))
    geohnc.units = 'gpm'
    geohnc.long_name = 'Derived geopotential height on full model levels'

    print("Compute scale for geoh at:    " + tim.strftime("%Y-%m-%d %H:%M:%S", tim.localtime()))
    geohnc.scale_factor, geohnc.add_offset = compute_scale_and_offset(min_value=final_geopotential_height.min(),
                                                                      max_value=final_geopotential_height.max())

    print("Save results for geoh at:     " + tim.strftime("%Y-%m-%d %H:%M:%S", tim.localtime()))
    geohnc[:, :, :, :] = final_geopotential_height

    print("-------------")


    # ------------
    # add variable final_pressure
    print("Create pressure at:           " + tim.strftime("%Y-%m-%d %H:%M:%S", tim.localtime()))
    pnc = outfile.createVariable('pres', "i2", ('time', 'level', 'latitude', 'longitude'))

    print("Add descript. to pressure at: " + tim.strftime("%Y-%m-%d %H:%M:%S", tim.localtime()))
    pnc.units = 'Pa'
    pnc.long_name = 'Derived pressure on full model levels'

    print("Compute scale for pressure at:" + tim.strftime("%Y-%m-%d %H:%M:%S", tim.localtime()))
    pnc.scale_factor, pnc.add_offset = compute_scale_and_offset(min_value=final_pressure.min(),
                                                                max_value=final_pressure.max())

    print("Save results for pressure at: " + tim.strftime("%Y-%m-%d %H:%M:%S", tim.localtime()))
    pnc[:, :, :, :] = final_pressure

    print("-------------")

    # ------------
    # close netCDF file
    outfile.close()
    return


def compute_scale_and_offset(min_value, max_value, nbits=16):
    # calculate scale_factor and offset for compression
    # min_value : minimum value of your data
    # max value : maximum value of your data
    # nbits     : number of bits you have to save your data in.
    #             gives accuracy. if integer with 16 bits set nbits = 16.
    # returns
    # scale_factor : factor to scale your data with. if below 1, it is set to 1
    # add_offset   : offset = min_value

    # stretch/compress data to the available packed range
    scale_factor = (max_value - min_value) / (2 ** nbits - 1)
    add_offset = min_value + scale_factor * 32768

    return scale_factor, add_offset


# ----------------------------------------------
# Call functions and print the results
# ----------------------------------------------
print("Loading data at:              " + tim.strftime("%Y-%m-%d %H:%M:%S", tim.localtime()))

# ----------
# Data input

# Get model level definitions. Either load data or download it.
all_a, all_b = getModelLevelDefinition(path_to_levDef=LEVEL_DEFINITION)

# load era5 data
era5_ds = loadAlignCombineERA5(path_model_level_t_q=ML_DATA,
                               path_surface_sp=SURFACE_DATA,
                               path_surface_geopotential=GEOPOTENTIAL_DATA)

# # calculate sp from lnsp if neccessary (added 2021-05-06)
# if "lnsp" in list(era5_ds.keys()) :
#     era5_ds = era5_ds.assign(sp = np.exp(era5_ds.lnsp))

# ----------
# get dimensions from data
ERA5_MIN_LEVEL = int(era5_ds.level[0].values.item())  # Highest level to calculate final_geopotential for
ERA5_MAX_LEVEL = int(era5_ds.level[-1].values.item())  # Lowest level of era5. Don't change this!

# initialize result
geopotential_height = np.zeros(era5_ds.t.shape)
pressure = np.zeros(era5_ds.t.shape)

# ----------
# calculate geopotential
print("Calculating at:               " + tim.strftime("%Y-%m-%d %H:%M:%S", tim.localtime()))
res_geopotential_height, res_pressure = getGeopotential(ERA5_MIN_LEVEL)

# ----------------------------------------------
# safe to netCDF
# ----------------------------------------------

saveToNetCDF(final_geopotential_height=res_geopotential_height,
             final_pressure=res_pressure,
             outfile_name=OUTFILE,
             ds=era5_ds)


# print some time statistics
print("Now it is: " + tim.strftime("%Y-%m-%d %H:%M:%S", tim.localtime()))
time_end = tim.time()
time_diff_sec = time_end - time_begin
time_diff_min = time_diff_sec / 60
time_diff_hour = time_diff_min / 60
print("The whole script took: " + str(time_diff_sec) + " seconds or")
print("The whole script took: " + str(time_diff_min) + " minutes or")
print("The whole script took: " + str(time_diff_hour) + " hours")
print("   ")
print("Done with: " + OUTFILE)
print("End of compute_vert_coords.py")
print("-------------")
