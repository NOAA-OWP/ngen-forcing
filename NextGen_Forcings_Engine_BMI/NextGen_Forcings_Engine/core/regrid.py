"""Regridding module file for regridding input forcing files."""

import functools
import hashlib
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from time import monotonic, time
from pathlib import Path
from .. import os_utils

# import mpi4py.util.pool as mpi_pool
# For ESMF + shapely 2.x, shapely must be imported first, to avoid segfault "address not mapped to object" stemming from calls such as:
# /usr/local/esmf/lib/libO/Linux.gfortran.64.openmpi.default/libesmf_fullylinked.so(get_geom+0x36)
import shapely

# from mpi4py.futures import MPIPoolExecutor
from mpi4py.futures import MPICommExecutor

try:
    import esmpy as ESMF
except ImportError:
    import ESMF

import logging

from ..esmf_utils import esmf_field_retry, esmf_grid_retry, esmf_mesh_retry, esmf_regrid_retry, esmf_regridfromfile_retry, esmf_regridobj_call_retry

import dask
import dask.delayed
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

from . import err_handler, ioMod, timeInterpMod
from .config import ConfigOptions
from .geoMod import GeoMetaWrfHydro
from .parallel import MpiConfig

# Get log level string from environment variable (defaults to 'INFO' if unset)
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()

# Convert string ('DEBUG', 'INFO', 'WARNING', etc.) to logging level constant
log_level = logging.getLevelName(log_level_str)

# Fallback check in case an invalid string was passed in the environment variable
if not isinstance(log_level, int):
    log_level = logging.INFO

# Configure logging directly to stdout
logging.basicConfig(level=log_level, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')

LOG = logging.getLogger()

if "WGRIB2" not in os.environ:
    WGRIB2_env = False
else:
    WGRIB2_env = True

NETCDF = "NETCDF"
GRIB2 = "GRIB2"

next_file_number = 0


@contextmanager
def timing_block(step_str: str):
    """Context manager for timing code execution.

    Args:
        step_str: Description of the step being timed.

    """
    start = time()
    yield
    end = time()
    LOG.debug(f"  Execution time for {step_str}: {round(end - start, 2)} seconds")


def mkfilename():
    """Create a unique filename suffix."""
    global next_file_number
    next_file_number += 1
    return f"{next_file_number}"


def static_vars(**kwargs):
    """Add static variables to a function."""

    def decorate(func):
        """Add static variables to a function."""
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def create_link(name, input_file, tmpFile, config_options, mpi_config):
    """Create a symbolic link to the input file for processing."""
    if mpi_config.rank == 0:
        try:
            config_options.statusMsg = name + " file being used: " + input_file
            err_handler.log_msg(config_options, mpi_config, True)  # log at debug level

            os.symlink(input_file, tmpFile)
        except:
            config_options.errMsg = "Unable to create link: " + input_file + " to: " + tmpFile
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

def get_mrms_subhourly_avg(config_options,supplemental_precip,mpi_config):

    supplemental_precip.file_in1_grib2 = []
    # Loop through the MRMS sub-hourly gz files and extract grib2 files
    for mrms_gzfile in supplemental_precip.file_in1:
        mrms_grib2_name = mrms_gzfile.stem
        mrms_tmp_grib2 = str(Path(config_options.scratch_dir) / f"{mrms_grib2_name}")
        ioMod.unzip_file(mrms_gzfile, mrms_tmp_grib2, config_options, mpi_config)
        supplemental_precip.file_in1_grib2.append(mrms_tmp_grib2)

    # Extract the second time stamp of MRMS sub-hourly data if required for the given cycle
    if(len(supplemental_precip.tmpFile2_mrms_subhourly_timesteps) != 0):
        supplemental_precip.file_in2_grib2 = []
        # Loop through the MRMS sub-hourly gz files and extract grib2 files
        for mrms_gzfile in supplemental_precip.file_in2:
            mrms_grib2_name = mrms_gzfile.stem
            mrms_tmp_grib2 = str(Path(config_options.scratch_dir) / f"{mrms_grib2_name}")
            ioMod.unzip_file(mrms_gzfile, mrms_tmp_grib2, config_options, mpi_config)
            supplemental_precip.file_in2_grib2.append(mrms_tmp_grib2)

    # Extract the first and second time stamp of MRMS hourly data for the bias correction calculation
    mrms_grib2_name = supplemental_precip.file_in1_bc.stem
    mrms_hourly_tmp1_grib2 = str(Path(config_options.scratch_dir) / f"{mrms_grib2_name}")
    ioMod.unzip_file(supplemental_precip.file_in1_bc,mrms_hourly_tmp1_grib2, config_options, mpi_config)

    if(len(supplemental_precip.tmpFile2_mrms_subhourly_timesteps) != 0):
        mrms_grib2_name = supplemental_precip.file_in2_bc.stem
        mrms_hourly_tmp2_grib2 = str(Path(config_options.scratch_dir) / f"{mrms_grib2_name}")
        ioMod.unzip_file(supplemental_precip.file_in2_bc,mrms_hourly_tmp2_grib2, config_options, mpi_config)

    # Open grib2 files associated with current AnA hour
    ds_subhourly = xr.open_mfdataset(supplemental_precip.file_in1_grib2,engine="cfgrib",concat_dim="time",combine="nested")
    # Set missing values (< 0) to nan values for xarray computations
    ds_subhourly['precip_rate'] = ds_subhourly['unknown'].where(ds_subhourly['unknown'] >= 0, np.nan)
    # Resample 2-minute snapshots into 15-minute mean intensity blocks
    subhourly_15m = ds_subhourly.resample(time="15min", closed="left").mean(dim="time")
    # Calculate the precipitation accumulation over the 15 minute interval
    subhourly_15m_depth = subhourly_15m['precip_rate'] * 0.25
    # Calculate the hourly precipitation accumulation estimate from MRMS sub-hourly data fields
    subhourly_accumulation = subhourly_15m_depth.sum(dim="time", skipna=True)

    #Open MRMS hourly gauge corrected (truth) precipitation fields
    ds_hourly = xr.open_mfdataset(mrms_hourly_tmp1_grib2,engine="cfgrib")
    # Set missing values (< 0) to nan values for xarray computations
    mrms_hourly = ds_hourly["unknown"].where(ds_hourly["unknown"] >= 0, np.nan)

    # Calculate the MRMS sub-hourly bias residual
    bias_ratio = mrms_hourly / (subhourly_accumulation + 1e-05)

    # Apply Meteorological Guardrails to the Bias Ratio Grid
    # Condition: Only apply a correction if BOTH fields register meaningful rain (> 0.1 mm)
    # Otherwise, default the bias multiplier to 1.0 (no change/neutral)
    meaningful_rain_mask = (mrms_hourly > 0.1) & (subhourly_accumulation > 0.1)

    # Apply mask, clip the ratio between 0.1x and 10.0x, and fill dry pixels with 1.0
    bounded_bias_ratio = bias_ratio.where(meaningful_rain_mask, 1.0)
    bounded_bias_ratio = bounded_bias_ratio.clip(min=0.1, max=10.0)


    # Derive spatial predictions using a Gaussian spatial filter (sigma=5 grid cells)
    # This extracts the continuous error trends from terrain/radar alignment quirks
    bias_per_15min_depth = gaussian_filter(bounded_bias_ratio.compute().values, sigma=5)

    # Convert arrays back into Xarray coordinates for clean dataset mapping
    bias_depth_xr = xr.DataArray(bias_per_15min_depth, coords=[ds_subhourly.latitude.compute().values, ds_subhourly.longitude.compute().values], dims=["latitude", "longitude"])

    # Apply the correction in absolute depth space (mm)
    corrected_15m_depth = subhourly_15m_depth * bias_depth_xr

    # Apply physical constraints: Precipitation depth cannot drop below 0.0 mm
    subhourly_15m_tmp1 = corrected_15m_depth.where(corrected_15m_depth >= 0, 0.0)

    if(len(supplemental_precip.tmpFile2_mrms_subhourly_timesteps) != 0):
        # Open grib2 files associated with current AnA hour
        ds_subhourly = xr.open_mfdataset(supplemental_precip.file_in2_grib2,engine="cfgrib",concat_dim="time",combine="nested")
        # Set missing values (< 0) to nan values for xarray computations
        ds_subhourly['precip_rate'] = ds_subhourly['unknown'].where(ds_subhourly['unknown'] >= 0, np.nan)
        # Resample 2-minute snapshots into 15-minute mean intensity blocks
        subhourly_15m = ds_subhourly.resample(time="15min", closed="left").mean(dim="time")
        # Calculate the precipitation accumulation over the 15 minute interval
        subhourly_15m_depth = subhourly_15m['precip_rate'] * 0.25
        # Calculate the hourly precipitation accumulation estimate from MRMS sub-hourly data fields
        subhourly_accumulation = subhourly_15m_depth.sum(dim="time", skipna=True)

        #Open MRMS hourly gauge corrected (truth) precipitation fields
        ds_hourly = xr.open_mfdataset(mrms_hourly_tmp2_grib2,engine="cfgrib")
        # Set missing values (< 0) to nan values for xarray computations
        mrms_hourly = ds_hourly["unknown"].where(ds_hourly["unknown"] >= 0, np.nan)

        # Calculate the MRMS sub-hourly bias residual
        bias_ratio = mrms_hourly / (subhourly_accumulation + 1e-05)

        # Apply Meteorological Guardrails to the Bias Ratio Grid
        # Condition: Only apply a correction if BOTH fields register meaningful rain (> 0.1 mm)
        # Otherwise, default the bias multiplier to 1.0 (no change/neutral)
        meaningful_rain_mask = (mrms_hourly > 0.1) & (subhourly_accumulation > 0.1)

        # Apply mask, clip the ratio between 0.1x and 10.0x, and fill dry pixels with 1.0
        bounded_bias_ratio = bias_ratio.where(meaningful_rain_mask, 1.0)
        bounded_bias_ratio = bounded_bias_ratio.clip(min=0.1, max=10.0)

        # Derive spatial predictions using a Gaussian spatial filter (sigma=5 grid cells)
        # This extracts the continuous error trends from terrain/radar alignment quirks
        bias_per_15min_depth = gaussian_filter(bounded_bias_ratio.compute().values, sigma=5)

        # Convert arrays back into Xarray coordinates for clean dataset mapping
        bias_depth_xr = xr.DataArray(bias_per_15min_depth, coords=[ds_subhourly.latitude.compute().values, ds_subhourly.longitude.compute().values], dims=["latitude", "longitude"])

        # Apply the correction in absolute depth space (mm)
        corrected_15m_depth = subhourly_15m_depth * bias_depth_xr

        # Apply physical constraints: Precipitation depth cannot drop below 0.0 mm
        subhourly_15m_tmp2 = corrected_15m_depth.where(corrected_15m_depth >= 0, 0.0)

        # Slice the sub-hourly cycle timestamps from tmp1 and tmp2 MRMS sub-hourly data
        subhourly_15m_tmp1 = subhourly_15m_tmp1.isel(time=supplemental_precip.tmpFile1_mrms_subhourly_timesteps)

        subhourly_15m_tmp2 = subhourly_15m_tmp2.isel(time=supplemental_precip.tmpFile2_mrms_subhourly_timesteps)

        # Concatenate the xarray dataframes together based on sub-hourly cycle of interest
        combined_subhourly = xr.concat([subhourly_15m_tmp1, subhourly_15m_tmp2], dim="time")

        # Compute hourly average from sub-hourly data for 15, 30, or 45 minute cycle
        mrms_subhourly_final = combined_subhourly.mean(dim="time", skipna=True).compute().values

        # Remove the variables no longer needed for this specific method
        del subhourly_15m_tmp2
        del combined_subhourly

    else:
        # Compute hourly average from sub-hourly data for 00 minute cycle
        mrms_subhourly_final = subhourly_15m_tmp1.mean(dim="time",skipna=True).compute().values

    # Remove the Python variables that are no longer needed
    del ds_subhourly
    del subhourly_15m
    del subhourly_15m_depth
    del subhourly_accumulation
    del mrms_hourly
    del ds_hourly
    del bias_ratio
    del bounded_bias_ratio
    del meaningful_rain_mask
    del bias_per_15min_depth
    del bias_depth_xr
    del corrected_15m_depth
    del subhourly_15m_tmp1


    # Now remove the temporary MRMS grib files since calculations were completed
    for f in supplemental_precip.file_in1_grib2:
        if os.path.isfile(f):
            try:
                os_utils.os_remove_retry(f)
            except OSError:
                config_options.errMsg = f"Unable to remove scratch file: {f}"
                LOG.warning(config_options.errMsg)

    if os.path.isfile(mrms_hourly_tmp1_grib2):
        try:
            os_utils.os_remove_retry(mrms_hourly_tmp1_grib2)
        except OSError:
            config_options.errMsg = f"Unable to remove scratch file: {mrms_hourly_tmp1_grib2}"
            LOG.warning(config_options.errMsg)

    if(len(supplemental_precip.tmpFile2_mrms_subhourly_timesteps) != 0):
        for f in supplemental_precip.file_in2_grib2:
            if os.path.isfile(f):
                try:
                    os_utils.os_remove_retry(f)
                except OSError:
                    config_options.errMsg = f"Unable to remove scratch file: {f}"
                    LOG.warning(config_options.errMsg)

        if os.path.isfile(mrms_hourly_tmp2_grib2):
            try:
                os_utils.os_remove_retry(mrms_hourly_tmp2_grib2)
            except OSError:
                config_options.errMsg = f"Unable to remove scratch file: {mrms_hourly_tmp2_grib2}"
                LOG.warning(config_options.errMsg)

    return mrms_subhourly_final

def get_subhourly_avg(input_forcings,var,id_tmp,id_tmp2):

    # If sub-hourly cycling is 15, 30, or 45 minutes
    if(len(input_forcings.file_in2_indices) != 0):

        # Open the HRRR datasets between sub-hourly forecast hour
        ds1 = xr.open_dataset(id_tmp)
        ds2 = xr.open_dataset(id_tmp2)

        # Extract only the specific variable requested
        da1_var = ds1[var]
        da2_var = ds2[var]

        # Subset the data using integer-position indexing along the time dimension
        da1_subset = da1_var.isel(time=input_forcings.file_in1_indices)
        da2_subset = da2_var.isel(time=input_forcings.file_in2_indices)

        # Concatenate the two subsets together along the time axis
        combined_series = xr.concat([da1_subset, da2_subset], dim='time')

        if var != "APCP":
            # Calculate the temporal mean across the concatenated time index
            spatial_avg = combined_series.mean(dim='time', keep_attrs=True)
        else:
            # Calculate the sum for precipitation accumulation fields across the concatenated time index
            spatial_avg = combined_series.sum(dim='time', keep_attrs=True)

        # Compute the data into memory and extract the raw 2D array underlying it
        spatial_avg_computed = spatial_avg.compute().values

        # Close datasets to free up system file handles safely
        ds1.close()
        ds2.close()
        da1_subset.close()
        da2_subset.close()
        combined_series.close()
        spatial_avg.close()

    # We only need one HRRR-subhourly file for 00 minute sub-hourly cycle
    else:
        # Open single HRRR sub-hourly file for current forecast hour
        ds1 = xr.open_dataset(id_tmp)

        # Extract only the specific variable requested
        da1_var = ds1[var]

        # Subset the data using integer-position indexing along the time dimension
        da1_subset = da1_var.isel(time=input_forcings.file_in1_indices)

        if var != "APCP":
            # Calculate the temporal mean across the concatenated time index
            spatial_avg = da1_subset.mean(dim='time', keep_attrs=True)
        else:
            # Calculate the sum for precipitation accumulation fields across the concatenated time index
            spatial_avg = da1_subset.sum(dim='time', keep_attrs=True)

        # Compute the data into memory and extract the raw 2D array underlying it
        spatial_avg_computed = spatial_avg.compute().values

        # Close datasets to free up system file handles safely
        ds1.close()
        da1_subset.close()
        spatial_avg.close()

    return spatial_avg_computed

@dask.delayed
def compute(id_tmp, nc_var):
    """Compute masked array for a given NetCDF variable."""
    return id_tmp[nc_var].to_masked_array()


def regrid_ak_ext_ana(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Read in and regrid Alaska ExtAna data.

    Function for handling regridding of Alaska ExtAna data.
    Data was already regridded in the prior run of the AnA stage so just read data in.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    esmf_grid_retry_partial = functools.partial(esmf_grid_retry, mpi_config, config_options, err_handler)
    esmf_mesh_retry_partial = functools.partial(esmf_mesh_retry, mpi_config, config_options, err_handler)

    ds = None

    try:
        # If the expected file is missing, this means we are allowing missing files, simply
        # exit out of this routine as the regridded fields have already been set to NDV.
        if not os.path.isfile(input_forcings.file_in2):
            if mpi_config.rank == 0:
                config_options.statusMsg = "No AK AnA in_2 file found for this timestep."
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
            return

        # Check to see if the regrid complete flag for this
        # output time step is true. This entails the necessary
        # inputs have already been regridded and we can move on.
        if input_forcings.regridComplete:
            if mpi_config.rank == 0:
                config_options.statusMsg = "No AK AnA regridding required for this timestep."
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
            return

        # Only rank‐0 opens the file
        if mpi_config.rank == 0:
            from netCDF4 import Dataset

            try:
                ds = Dataset(input_forcings.file_in2, "r")
            except Exception as e:
                config_options.errMsg = f"Unable to open input NetCDF file: {input_forcings.file_in2} ({e})"
                err_handler.log_critical(config_options, mpi_config)

            # turn off automatic masking but keep scaling
            ds.set_auto_scale(True)
            ds.set_auto_mask(False)

        if input_forcings.nx_global is None or input_forcings.ny_global is None:
            # This is the first timestep.
            if mpi_config.rank == 0:
                input_forcings.ny_global = ds.dimensions["y"].size
                input_forcings.nx_global = ds.dimensions["x"].size

            input_forcings.ny_global = mpi_config.broadcast_parameter(input_forcings.ny_global, config_options, param_type=int)
            err_handler.check_program_status(config_options, mpi_config)
            input_forcings.nx_global = mpi_config.broadcast_parameter(input_forcings.nx_global, config_options, param_type=int)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                try:
                    # noinspection PyTypeChecker
                    input_forcings.esmf_grid_in = esmf_grid_retry_partial(np.array([input_forcings.ny_global, input_forcings.nx_global]), staggerloc=ESMF.StaggerLoc.CENTER, coord_sys=ESMF.CoordSys.SPH_DEG)
                except ESMF.ESMPyException as esmf_error:
                    config_options.errMsg = f"Unable to create source ESMF grid from netCDF file: {input_forcings.file_in} ({str(esmf_error)})"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.x_lower_bound = input_forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][1]
                    input_forcings.x_upper_bound = input_forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][1]
                    input_forcings.y_lower_bound = input_forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][0]
                    input_forcings.y_upper_bound = input_forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][0]
                    input_forcings.nx_local = input_forcings.x_upper_bound - input_forcings.x_lower_bound
                    input_forcings.ny_local = input_forcings.y_upper_bound - input_forcings.y_lower_bound
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = f"Unable to extract local X/Y boundaries from global grid from netCDF file: {input_forcings.file_in} ({str(err)})"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "unstructured":
                try:
                    input_forcings.esmf_grid_in = esmf_mesh_retry_partial(filename=config_options.geogrid, filetype=ESMF.FileFormat.ESMFMESH)
                    input_forcings.esmf_grid_in_elem = esmf_mesh_retry_partial(filename=config_options.geogrid, filetype=ESMF.FileFormat.ESMFMESH)
                except ESMF.ESMPyException as esmf_error:
                    config_options.errMsg = f"Unable to create source ESMF Mesh from netCDF file: {input_forcings.file_in} ({str(esmf_error)})"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.nx_local = len(input_forcings.esmf_grid_in.esmf_grid_in.coords[0][1])
                    input_forcings.ny_local = len(input_forcings.esmf_grid_in.esmf_grid_in.coords[0][1])
                    input_forcings.nx_local_elem = len(input_forcings.esmf_grid_in_elem.esmf_grid_in.coords[0][1])
                    input_forcings.ny_local_elem = len(input_forcings.esmf_grid_in_elem.esmf_grid_in.coords[0][1])
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = f"Unable to extract local X/Y boundaries from global mesh file: {input_forcings.file_in} ({str(err)})"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "hydrofabric":
                try:
                    input_forcings.esmf_grid_in = esmf_mesh_retry_partial(filename=config_options.geogrid, filetype=ESMF.FileFormat.ESMFMESH)
                except ESMF.ESMPyException as esmf_error:
                    config_options.errMsg = f"Unable to create source ESMF Mesh from netCDF file: {input_forcings.file_in} ({str(esmf_error)})"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.nx_local = len(input_forcings.esmf_grid_in.esmf_grid_in.coords[0][1])
                    input_forcings.ny_local = len(input_forcings.esmf_grid_in.esmf_grid_in.coords[0][1])
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = f"Unable to extract local X/Y boundaries from global mesh file: {input_forcings.file_in} ({str(err)})"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            # Create out regridded numpy arrays to hold the regridded data.
            force_count = 9 if config_options.include_lqfrac else 8

            if config_options.grid_type == "gridded":
                input_forcings.regridded_forcings1 = np.empty([force_count, wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32)
                input_forcings.regridded_forcings2 = np.empty([force_count, wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32)
            elif config_options.grid_type == "unstructured":
                input_forcings.regridded_forcings1 = np.empty([force_count, wrf_hydro_geo_meta.ny_local], np.float32)
                input_forcings.regridded_forcings2 = np.empty([force_count, wrf_hydro_geo_meta.ny_local], np.float32)
                input_forcings.regridded_forcings1_elem = np.empty([force_count, wrf_hydro_geo_meta.ny_local_elem], np.float32)
                input_forcings.regridded_forcings2_elem = np.empty([force_count, wrf_hydro_geo_meta.ny_local_elem], np.float32)
            elif config_options.grid_type == "unstructured":
                input_forcings.regridded_forcings1 = np.empty([force_count, wrf_hydro_geo_meta.ny_local], np.float32)
                input_forcings.regridded_forcings2 = np.empty([force_count, wrf_hydro_geo_meta.ny_local], np.float32)

        for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
            var_tmp = None
            var_tmp_elem = None
            if mpi_config.rank == 0:
                config_options.statusMsg = f"Processing input AK AnA variable: {nc_var} from {input_forcings.file_in2}"
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                LOG.debug(f"{config_options.statusMsg}")
                if config_options.grid_type == "gridded":
                    try:
                        var_tmp = ds.variables[nc_var][0, :, :]
                        var_tmp = np.float32(var_tmp)

                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = f"Unable to extract: {nc_var} from: {input_forcings.file_in2} ({str(err)})"
                        err_handler.log_critical(config_options, mpi_config)
                elif config_options.grid_type == "unstructured":
                    try:
                        var_tmp = ds.variables[nc_var][0, :]
                        var_tmp = np.float32(var_tmp)

                        var_tmp_elem = ds.variables[nc_var][0, :]
                        var_tmp_elem = np.float32(var_tmp_elem)
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = f"Unable to extract: {nc_var} from: {input_forcings.file_in2} ({str(err)})"
                        err_handler.log_critical(config_options, mpi_config)
                elif config_options.grid_type == "hydrofabric":
                    try:
                        var_tmp = ds.variables[nc_var][0, :]
                        var_tmp = np.float32(var_tmp)

                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = f"Unable to extract: {nc_var} from: {input_forcings.file_in2} ({str(err)})"
                        err_handler.log_critical(config_options, mpi_config)

            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)
                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ExtAnA forcing data from the AK AnA field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
            elif config_options.grid_type == "unstructured":
                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = var_tmp[wrf_hydro_geo_meta.mesh_inds]
                    input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :, :] = var_tmp[wrf_hydro_geo_meta.mesh_inds_elem]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ExtAnA forcing data from the AK AnA mesh field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
                    input_forcings.regridded_forcings1_elem[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :]
            elif config_options.grid_type == "hydrofabric":
                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = var_tmp[wrf_hydro_geo_meta.mesh_inds]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ExtAnA forcing data from the AK AnA mesh field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
    finally:
        if mpi_config.rank == 0 and ds is not None:
            try:
                ds.close()
            except OSError:
                config_options.errMsg = f"Unable to close input NetCDF file: {input_forcings.file_in2}"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

def _regrid_ak_ext_ana_pcp_stage4(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """Read in and regrid Alaska ExtAna supplemental Stage IV precip data."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if not os.path.exists(supplemental_precip.file_in1): return

    if supplemental_precip.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No StageIV regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)
        return

    file_name = f"STAGEIV_AK_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    stage4_tmp_nc = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    lat_var = "latitude"
    lon_var = "longitude"

    id_tmp = None
    try:
        if supplemental_precip.file_type != NETCDF:
            if mpi_config.rank == 0:
                if os.path.isfile(stage4_tmp_nc):
                    config_options.statusMsg = "Found old temporary file: " + stage4_tmp_nc + " - Removing....."
                    err_handler.log_warning(config_options, mpi_config)
                    try:
                        os_utils.os_remove_retry(stage4_tmp_nc)
                    except OSError:
                        config_options.errMsg = f"Unable to remove temporary file: {stage4_tmp_nc}"
                        err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if WGRIB2_env:
                cmd = f'$WGRIB2 -match "APCP:surface:0-6 hour acc fcst" {supplemental_precip.file_in2} -netcdf {stage4_tmp_nc}'
            else:
                cmd = "APCP:surface:0-6 hour acc fcst"

            if mpi_config.rank == 0:
                config_options.statusMsg = f"WGRIB2 command: {cmd}"
                err_handler.log_msg(config_options, mpi_config, True)
            id_tmp = ioMod.open_grib2(supplemental_precip.file_in2, stage4_tmp_nc, cmd, config_options, mpi_config, inputVar=None, special_case=False)
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link("STAGEIV-PCP", supplemental_precip.file_in2, stage4_tmp_nc, config_options, mpi_config)
            id_tmp = ioMod.open_netcdf_forcing(stage4_tmp_nc, config_options, mpi_config, False, lat_var, lon_var)

        calc_regrid_flag = check_supp_pcp_regrid_status(id_tmp, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Calculating STAGE IV regridding weights."
                err_handler.log_msg(config_options, mpi_config, True)
            calculate_supp_pcp_weights(supplemental_precip, id_tmp, stage4_tmp_nc, config_options, mpi_config, lat_var, lon_var)
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp = id_tmp.variables[supplemental_precip.netcdf_var_names[-1]][0, :, :]
                    var_tmp = np.where(var_tmp == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue, 0.0, var_tmp)
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract precipitation from STAGE IV file: " + supplemental_precip.file_in1 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)
        elif config_options.grid_type == "unstructured":
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp = id_tmp.variables[supplemental_precip.netcdf_var_names[-1]][0, :, :]
                    var_tmp = np.where(var_tmp == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue, 0.0, var_tmp)
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract precipitation from STAGE IV file: " + supplemental_precip.file_in1 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            var_tmp_elem = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp_elem = id_tmp.variables[supplemental_precip.netcdf_var_names[-1]][0, :, :]
                    var_tmp = np.where(var_tmp == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue, 0.0, var_tmp)
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract precipitation from STAGE IV file: " + supplemental_precip.file_in1 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(supplemental_precip, var_tmp_elem, config_options)
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp = id_tmp.variables[supplemental_precip.netcdf_var_names[-1]][0, :, :]
                    var_tmp = np.where(var_tmp == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue, 0.0, var_tmp)
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract precipitation from STAGE IV file: " + supplemental_precip.file_in1 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place STAGE IV precipitation into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:, :] = supplemental_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
                supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = "Unable to run NDV search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:, :] = supplemental_precip.regridded_precip2[:, :]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place STAGE IV precipitation into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = supplemental_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
                supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = "Unable to run NDV search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = supplemental_precip.regridded_precip2[:]
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place STAGE IV precipitation into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj_elem, supplemental_precip.esmf_field_in_elem, supplemental_precip.esmf_field_out_elem)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem.data[np.where(supplemental_precip.regridded_mask_elem == 0)] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2_elem[:] = supplemental_precip.esmf_field_out_elem.data
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(supplemental_precip.regridded_precip2_elem != config_options.globalNdv)
                supplemental_precip.regridded_precip2_elem[ind_valid] = supplemental_precip.regridded_precip2_elem[ind_valid] / 3600.0
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = "Unable to run NDV search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1_elem[:] = supplemental_precip.regridded_precip2_elem[:]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place STAGE IV precipitation into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = supplemental_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
                supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = "Unable to run NDV search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = supplemental_precip.regridded_precip2[:]
            err_handler.check_program_status(config_options, mpi_config)

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = f"Unable to close NetCDF file: {stage4_tmp_nc}: {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(stage4_tmp_nc)
            except FileNotFoundError:
                config_options.statusMsg = f"NetCDF file not found, continuing: {stage4_tmp_nc}"
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = f"Unable to remove NetCDF file: {stage4_tmp_nc}: {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)


def regrid_ak_ext_ana_pcp(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """Read in and regrid Alaska ExtAna supplemental precip data."""
    if supplemental_precip.ext_ana == "STAGE4":
        supplemental_precip.netcdf_var_names.append("APCP_surface")
        _regrid_ak_ext_ana_pcp_stage4(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config)
        supplemental_precip.netcdf_var_names.pop()
    else:
        supplemental_precip.netcdf_var_names.append("MultiSensorQPE01H_0mabovemeansealevel")
        regrid_mrms_hourly(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config)
        supplemental_precip.netcdf_var_names.pop()

def _regrid_conus_ext_ana_pcp_stage4(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """Read in and regrid Alaska ExtAna supplemental Stage IV precip data."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if not os.path.exists(supplemental_precip.file_in1): return

    if supplemental_precip.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No StageIV regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)
        return

    file_name = f"STAGEIV_CONUS_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    stage4_tmp_nc = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    lat_var = "latitude"
    lon_var = "longitude"

    id_tmp = None
    try:
        if supplemental_precip.file_type != NETCDF:
            if mpi_config.rank == 0:
                if os.path.isfile(stage4_tmp_nc):
                    config_options.statusMsg = "Found old temporary file: " + stage4_tmp_nc + " - Removing....."
                    err_handler.log_warning(config_options, mpi_config)
                    try:
                        os_utils.os_remove_retry(stage4_tmp_nc)
                    except OSError:
                        config_options.errMsg = f"Unable to remove temporary file: {stage4_tmp_nc}"
                        err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if WGRIB2_env:
                cmd = f'$WGRIB2 -match "APCP:surface:0-1 hour acc fcst" {supplemental_precip.file_in2} -netcdf {stage4_tmp_nc}'
            else:
                cmd = "APCP:surface:0-1 hour acc fcst"

            if mpi_config.rank == 0:
                config_options.statusMsg = f"WGRIB2 command: {cmd}"
                err_handler.log_msg(config_options, mpi_config, True)
            id_tmp = ioMod.open_grib2(supplemental_precip.file_in2, stage4_tmp_nc, cmd, config_options, mpi_config, inputVar=None, special_case=False)
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link("STAGEIV-PCP", supplemental_precip.file_in2, stage4_tmp_nc, config_options, mpi_config)
            id_tmp = ioMod.open_netcdf_forcing(stage4_tmp_nc, config_options, mpi_config, False, lat_var, lon_var)

        calc_regrid_flag = check_supp_pcp_regrid_status(id_tmp, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Calculating STAGE IV regridding weights."
                err_handler.log_msg(config_options, mpi_config, True)
            calculate_supp_pcp_weights(supplemental_precip, id_tmp, stage4_tmp_nc, config_options, mpi_config, lat_var, lon_var)
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp = id_tmp.variables[supplemental_precip.netcdf_var_names[-1]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract precipitation from STAGE IV file: " + supplemental_precip.file_in1 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)
        elif config_options.grid_type == "unstructured":
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp = id_tmp.variables[supplemental_precip.netcdf_var_names[-1]][0, :, :].data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract precipitation from STAGE IV file: " + supplemental_precip.file_in1 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            var_tmp_elem = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp_elem = id_tmp.variables[supplemental_precip.netcdf_var_names[-1]][0, :, :].data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract precipitation from STAGE IV file: " + supplemental_precip.file_in1 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(supplemental_precip, var_tmp_elem, config_options)
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp = id_tmp.variables[supplemental_precip.netcdf_var_names[-1]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract precipitation from STAGE IV file: " + supplemental_precip.file_in1 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place STAGE IV precipitation into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:, :] = supplemental_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(supplemental_precip.regridded_precip2 < 1e10)
                supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                invalid = np.where(supplemental_precip.regridded_precip2 > 1e10)
                supplemental_precip.regridded_precip2[invalid] = config_options.globalNdv
                del invalid
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = "Unable to run NDV search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:, :] = supplemental_precip.regridded_precip2[:, :]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place STAGE IV precipitation into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = supplemental_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(supplemental_precip.regridded_precip2 < 1e10)
                supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                invalid = np.where(supplemental_precip.regridded_precip2 > 1e10)
                supplemental_precip.regridded_precip2[invalid] = config_options.globalNdv
                del invalid
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = "Unable to run NDV search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = supplemental_precip.regridded_precip2[:]
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place STAGE IV precipitation into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj_elem, supplemental_precip.esmf_field_in_elem, supplemental_precip.esmf_field_out_elem)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem.data[np.where(supplemental_precip.regridded_mask_elem == 0)] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2_elem[:] = supplemental_precip.esmf_field_out_elem.data
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(supplemental_precip.regridded_precip2_elem < 1e10)
                supplemental_precip.regridded_precip2_elem[ind_valid] = supplemental_precip.regridded_precip2_elem[ind_valid] / 3600.0
                invalid = np.where(supplemental_precip.regridded_precip2_elem > 1e10)
                supplemental_precip.regridded_precip2_elem[invalid] = config_options.globalNdv
                del invalid
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = "Unable to run NDV search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1_elem[:] = supplemental_precip.regridded_precip2_elem[:]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place STAGE IV precipitation into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid STAGE IV supplemental precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = supplemental_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(supplemental_precip.regridded_precip2 < 1e10)
                supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                invalid = np.where(supplemental_precip.regridded_precip2 > 1e10)
                supplemental_precip.regridded_precip2[invalid] = config_options.globalNdv
                del invalid
                del ind_valid            
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = "Unable to run NDV search on STAGE IV supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = supplemental_precip.regridded_precip2[:]
            err_handler.check_program_status(config_options, mpi_config)

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = f"Unable to close NetCDF file: {stage4_tmp_nc}: {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(stage4_tmp_nc)
            except FileNotFoundError:
                config_options.statusMsg = f"NetCDF file not found, continuing: {stage4_tmp_nc}"
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = f"Unable to remove NetCDF file: {stage4_tmp_nc}: {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


def regrid_conus_ext_ana_pcp(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """Read in and regrid CONUS ExtAna supplemental precip data."""
    if supplemental_precip.ext_ana == "STAGE4":
        supplemental_precip.netcdf_var_names.append("APCP_surface")
        _regrid_conus_ext_ana_pcp_stage4(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config)
        supplemental_precip.netcdf_var_names.pop()
    else:
        supplemental_precip.netcdf_var_names.append("MultiSensorQPE01H_0mabovemeansealevel")
        regrid_mrms_hourly(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config)
        supplemental_precip.netcdf_var_names.pop()

def regrid_conus_hrrr(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid CONUS HRRR data."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if not os.path.isfile(input_forcings.file_in1) and input_forcings.product_name == "HRRR_15min":
        if mpi_config.rank == 0:
            config_options.statusMsg = "No HRRR file_in1 file found for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)
        return

    if not os.path.isfile(input_forcings.file_in2):
        if mpi_config.rank == 0:
            config_options.statusMsg = "No HRRR file_in2 file found for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)
        return

    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No HRRR regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)
        return

    file_name = f"HRRR_CONUS_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    input_forcings.tmpFile = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    if input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
        file_name = f"HRRR_CONUS_TMP2-{mkfilename()}.nc"
        file_uuid = str(mpi_config.uid64)
        input_forcings.tmpFile2 = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    id_tmp = None
    id_tmp2 = None
    try:
        config_options.statusMsg = "Regrid CONUS HRRR"
        err_handler.log_msg(config_options, mpi_config)

        if input_forcings.file_type != NETCDF:
            if mpi_config.rank == 0 and os.path.isfile(input_forcings.tmpFile):
                config_options.statusMsg = f"Found old temporary file: {input_forcings.tmpFile} - Removing..."
                err_handler.log_warning(config_options, mpi_config)
                try:
                    os_utils.os_remove_retry(input_forcings.tmpFile)
                except OSError:
                    config_options.errMsg = f"Unable to remove temporary file: {input_forcings.tmpFile}"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if mpi_config.rank == 0 and input_forcings.tmpFile2 is not None and os.path.isfile(input_forcings.tmpFile2) and input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                config_options.statusMsg = f"Found old temporary file: {input_forcings.tmpFile2} - Removing..."
                err_handler.log_warning(config_options, mpi_config)
                try:
                    os_utils.os_remove_retry(input_forcings.tmpFile2)
                except OSError:
                    config_options.errMsg = f"Unable to remove temporary file: {input_forcings.tmpFile2}"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Converting HRRR Variable: {grib_var}"
                    err_handler.log_msg(config_options, mpi_config, True)
                if 0 < input_forcings.cycle_freq < 60:
                    time_str = f"{input_forcings.fcst_min1}-{input_forcings.fcst_min2} min acc fcst" if grib_var == "APCP" else f"{input_forcings.fcst_min2} min fcst"
                    sub_rem = int(input_forcings.fcst_min1) % 60
                    sub_id = int(sub_rem / input_forcings.cycle_freq)
                else:
                    time_str = f"{input_forcings.fcst_hour1}-{input_forcings.fcst_hour2} hour acc fcst" if grib_var == "APCP" else f"{input_forcings.fcst_hour2} hour fcst"
                if input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                    fields.append(":(" + grib_var + "):(" + input_forcings.grib_levels[force_count] + "):")
                else:
                    fields.append(":" + grib_var + ":" + input_forcings.grib_levels[force_count] + ":" + time_str + ":")
            fields.append(":(HGT):(surface):")

            if WGRIB2_env:
                pattern = "|".join(fields)
                cmd = f'$WGRIB2 -match "({pattern})" {input_forcings.file_in2} -netcdf {input_forcings.tmpFile}'
            else:
                cmd = "(" + "|".join(fields) + ")"

            id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd, config_options, mpi_config, inputVar=None, special_case=False)
            err_handler.check_program_status(config_options, mpi_config)

            if input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                if WGRIB2_env:
                    pattern = "|".join(fields)
                    cmd = f'$WGRIB2 -match "({pattern})" {input_forcings.file_in1} -netcdf {input_forcings.tmpFile2}'
                else:
                    cmd = "(" + "|".join(fields) + ")"

                id_tmp2 = ioMod.open_grib2(input_forcings.file_in1, input_forcings.tmpFile2, cmd, config_options, mpi_config, inputVar=None, special_case=False)
                err_handler.check_program_status(config_options, mpi_config)

        else:
            create_link("HRRR", input_forcings.file_in2, input_forcings.tmpFile, config_options, mpi_config)
            id_tmp = ioMod.open_netcdf_forcing(input_forcings.tmpFile, config_options, mpi_config)

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Processing HRRR Variable: " + grib_var
                err_handler.log_msg(config_options, mpi_config, True)

            calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Calculating HRRR regridding weights."
                    err_handler.log_msg(config_options, mpi_config, True)
                calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.grid_type == "gridded":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            if 0 < input_forcings.cycle_freq < 60:
                                var_tmp = id_tmp.variables["HGT_surface"][sub_id]
                            elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                                var_tmp = get_subhourly_avg(input_forcings, "HGT_surface", input_forcings.tmpFile2, input_forcings.tmpFile)
                            else:
                                var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract HRRR elevation from " + input_forcings.tmpFile + ": " + str(err)

                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place input NetCDF HRRR data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding HRRR surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid HRRR surface elevation using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to perform HRRR mask search on elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract regridded HRRR elevation data from ESMF: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "unstructured":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            if 0 < input_forcings.cycle_freq < 60:
                                var_tmp = id_tmp.variables["HGT_surface"][sub_id]
                            elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                                var_tmp = get_subhourly_avg(input_forcings, "HGT_surface", input_forcings.tmpFile2, input_forcings.tmpFile)
                            else:
                                var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract HRRR elevation from " + input_forcings.tmpFile + ": " + str(err)

                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place input NetCDF HRRR data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding HRRR surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid HRRR surface elevation using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to perform HRRR mask search on elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract regridded HRRR elevation data from ESMF: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_tmp_elem = None
                    if mpi_config.rank == 0:
                        try:
                            if 0 < input_forcings.cycle_freq < 60:
                                var_tmp_elem = id_tmp.variables["HGT_surface"][sub_id]
                            elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                                var_tmp_elem = get_subhourly_avg(input_forcings, "HGT_surface", input_forcings.tmpFile2, input_forcings.tmpFile)
                            else:
                                var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract HRRR elevation from " + input_forcings.tmpFile + ": " + str(err)

                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place input NetCDF HRRR data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding HRRR surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid HRRR surface elevation using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to perform HRRR mask search on elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = input_forcings.esmf_field_out_elem.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract regridded HRRR elevation data from ESMF: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            if 0 < input_forcings.cycle_freq < 60:
                                var_tmp = id_tmp.variables["HGT_surface"][sub_id]
                            elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                                var_tmp = get_subhourly_avg(input_forcings, "HGT_surface", input_forcings.tmpFile2, input_forcings.tmpFile)
                            else:
                                var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract HRRR elevation from " + input_forcings.tmpFile + ": " + str(err)

                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place input NetCDF HRRR data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding HRRR surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid HRRR surface elevation using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to perform HRRR mask search on elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract regridded HRRR elevation data from ESMF: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                var_tmp = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Processing input HRRR variable: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        if 0 < input_forcings.cycle_freq < 60:
                            var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][sub_id, :, :]
                        elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                            var_tmp = get_subhourly_avg(input_forcings, input_forcings.netcdf_var_names[force_count], input_forcings.tmpFile2, input_forcings.tmpFile)
                        else:
                            var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                        if grib_var == "APCP":
                            var_tmp /= 3600
                        if grib_var == "CPOFP":
                            var_tmp[var_tmp >= 0] = (100 - var_tmp[var_tmp >= 0]) / 100
                            var_tmp[var_tmp < 0] = np.nan
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'CPOFP':
                    mask[np.isnan(var_sub_tmp)] = 0

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place input HRRR data into ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding Input HRRR Field: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input HRRR forcing data: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    input_forcings.esmf_field_out.data[np.isnan(input_forcings.esmf_field_out.data)] = -50
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to perform mask test on regridded HRRR forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract regridded HRRR forcing data from the ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]

                mask[:] = prev_mask

            elif config_options.grid_type == "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Processing input HRRR variable: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        if 0 < input_forcings.cycle_freq < 60:
                            var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][sub_id, :, :]
                        elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                            var_tmp = get_subhourly_avg(input_forcings, input_forcings.netcdf_var_names[force_count], input_forcings.tmpFile2, input_forcings.tmpFile)
                        else:
                            var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                        if grib_var == "APCP":
                            var_tmp /= 3600
                        if grib_var == "CPOFP":
                            var_tmp[var_tmp >= 0] = (100 - var_tmp[var_tmp >= 0]) / 100
                            var_tmp[var_tmp < 0] = np.nan
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'CPOFP':
                    mask[np.isnan(var_sub_tmp)] = 0

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place input HRRR data into ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding Input HRRR Field: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input HRRR forcing data: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    input_forcings.esmf_field_out.data[np.isnan(input_forcings.esmf_field_out.data)] = -50
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to perform mask test on regridded HRRR forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract regridded HRRR forcing data from the ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]

                var_tmp_elem = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Processing input HRRR variable: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        if 0 < input_forcings.cycle_freq < 60:
                            var_tmp_elem = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][sub_id, :, :]
                        elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                            var_tmp_elem = get_subhourly_avg(input_forcings, input_forcings.netcdf_var_names[force_count], input_forcings.tmpFile2, input_forcings.tmpFile)
                        else:
                            var_tmp_elem = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                        if grib_var == "APCP":
                            var_tmp_elem /= 3600
                        if grib_var == "CPOFP":
                            var_tmp_elem[var_tmp_elem >= 0] = (100 - var_tmp_elem[var_tmp_elem >= 0]) / 100
                            var_tmp_elem[var_tmp_elem < 0] = np.nan
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                mask_elem = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask_elem = np.copy(mask_elem)
                if grib_var == 'CPOFP':
                    mask_elem[np.isnan(var_sub_tmp_elem)] = 0

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place input HRRR data into ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding Input HRRR Field: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input HRRR forcing data: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = config_options.globalNdv
                    input_forcings.esmf_field_out_elem.data[np.isnan(input_forcings.esmf_field_out_elem.data)] = -50
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to perform mask test on regridded HRRR forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract regridded HRRR forcing data from the ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :]

                mask[:] = prev_mask
                mask_elem[:] = prev_mask_elem

            elif config_options.grid_type == "hydrofabric":
                var_tmp = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Processing input HRRR variable: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        if 0 < input_forcings.cycle_freq < 60:
                            var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][sub_id, :, :]
                        elif input_forcings.product_name == "HRRR_CONUS_15min_Cycling":
                            var_tmp = get_subhourly_avg(input_forcings, input_forcings.netcdf_var_names[force_count], input_forcings.tmpFile2, input_forcings.tmpFile)
                        else:
                            var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                        if grib_var == "APCP":
                            var_tmp /= 3600
                        if grib_var == "CPOFP":
                            var_tmp[var_tmp >= 0] = (100 - var_tmp[var_tmp >= 0]) / 100
                            var_tmp[var_tmp < 0] = np.nan
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'CPOFP':
                    mask[np.isnan(var_sub_tmp)] = 0

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place input HRRR data into ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding Input HRRR Field: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input HRRR forcing data: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    input_forcings.esmf_field_out.data[np.isnan(input_forcings.esmf_field_out.data)] = -50
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to perform mask test on regridded HRRR forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract regridded HRRR forcing data from the ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]

                mask[:] = prev_mask

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = f"Unable to close NetCDF file: {input_forcings.tmpFile} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile)
            except FileNotFoundError:
                config_options.statusMsg = f"NetCDF file not found, continuing: {input_forcings.tmpFile}"
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = f"Unable to remove NetCDF file: {input_forcings.tmpFile} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if mpi_config.rank == 0 and id_tmp2 is not None:
            try:
                id_tmp2.close()
            except Exception as e:
                config_options.errMsg = f"Unable to close NetCDF file: {input_forcings.tmpFile2} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile2)
            except FileNotFoundError:
                config_options.statusMsg = f"NetCDF file not found, continuing: {input_forcings.tmpFile2}"
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = f"Unable to remove NetCDF file: {input_forcings.tmpFile2} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

def regrid_conus_rap(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid CONUS RAP 13km data."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if not os.path.isfile(input_forcings.file_in2):
        return

    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No RAP regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)
        return

    file_name = f"RAP_CONUS_BGRB-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    input_forcings.tmpFile = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    file_name = f"RAP_CONUS_PGRB-{mkfilename()}.nc"
    input_forcings.tmpFile2 = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    err_handler.check_program_status(config_options, mpi_config)

    id_tmp = None
    try:
        config_options.statusMsg = "Regrid CONUS RAP"
        err_handler.log_msg(config_options, mpi_config)
        if input_forcings.file_type != NETCDF:
            if mpi_config.rank == 0:
                if os.path.isfile(input_forcings.tmpFile):
                    config_options.statusMsg = "Found old temporary file: " + input_forcings.tmpFile + " - Removing....."
                    err_handler.log_warning(config_options, mpi_config)
                    try:
                        os_utils.os_remove_retry(input_forcings.tmpFile)
                    except OSError:
                        config_options.errMsg = "Unable to remove file: " + input_forcings.tmpFile
                        err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Converting CONUS RAP Variable: " + grib_var
                    err_handler.log_msg(config_options, mpi_config, True)
                time_str = "{}-{} hour acc fcst".format(input_forcings.fcst_hour1, input_forcings.fcst_hour2) if grib_var in ("APCP") else str(input_forcings.fcst_hour2) + " hour fcst"
                fields.append(":" + grib_var + ":" + input_forcings.grib_levels[force_count] + ":" + time_str + ":")
            fields.append(":(HGT):(surface):")

            fields.append(":(CFRZR):(surface):")
            fields.append(":(CICEP):(surface):")
            fields.append(":(CSNOW):(surface):")
            fields.append(":(CRAIN):(surface):")

            if input_forcings.t2dDownscaleOpt == 3:
                fields.append(":(HGT):(12 hybrid level):")
                fields.append(":(TMP):(12 hybrid level):")

            if WGRIB2_env:
                cmd = '$WGRIB2 -match "(' + "|".join(fields) + ')" ' + input_forcings.file_in2 + " -netcdf " + input_forcings.tmpFile
                cmd2 = '$WGRIB2 -match "(' + "|".join(fields) + ')" ' + input_forcings.file_in2.replace("bgrb", "pgrb") + " -netcdf " + input_forcings.tmpFile2
            else:
                cmd = "(" + "|".join(fields) + ")"
                cmd2 = "(" + "|".join(fields) + ")"

            id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd, config_options, mpi_config, inputVar=None, special_case=False)
            err_handler.check_program_status(config_options, mpi_config)

            id_tmp2 = ioMod.open_grib2(input_forcings.file_in2.replace("bgrb", "pgrb"), input_forcings.tmpFile2, cmd2, config_options, mpi_config, inputVar=None, special_case=False)
            err_handler.check_program_status(config_options, mpi_config)

        else:
            create_link("RAP", input_forcings.file_in2, input_forcings.tmpFile, config_options, mpi_config)
            id_tmp = ioMod.open_netcdf_forcing(input_forcings.tmpFile, config_options, mpi_config)

            create_link("RAP", input_forcings.file_in2.replace("bgrb", "pgrb"), input_forcings.tmpFile2, config_options, mpi_config)
            id_tmp2 = ioMod.open_netcdf_forcing(input_forcings.tmpFile2, config_options, mpi_config)

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Processing Conus RAP Variable: " + grib_var
                err_handler.log_msg(config_options, mpi_config, True)

            if grib_var != "LQFRAC":
                calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
            else:
                calc_regrid_flag = False

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Calculating RAP regridding weights."
                    err_handler.log_msg(config_options, mpi_config, True)
                calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.grid_type == "gridded":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract HGT_surface from : " + id_tmp + " (" + str(err) + ")"
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place temporary RAP elevation variable into ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding RAP surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid RAP elevation data using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to perform mask search on RAP elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place RAP ESMF elevation field into local array: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "unstructured":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract HGT_surface from : " + id_tmp + " (" + str(err) + ")"
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place temporary RAP elevation variable into ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding RAP surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid RAP elevation data using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to perform mask search on RAP elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place RAP ESMF elevation field into local array: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_tmp_elem = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract HGT_surface from : " + id_tmp + " (" + str(err) + ")"
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place temporary RAP elevation variable into ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding RAP surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid RAP elevation data using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to perform mask search on RAP elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = input_forcings.esmf_field_out_elem.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place RAP ESMF elevation field into local array: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract HGT_surface from : " + id_tmp + " (" + str(err) + ")"
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place temporary RAP elevation variable into ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding RAP surface elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid RAP elevation data using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to perform mask search on RAP elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place RAP ESMF elevation field into local array: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        if grib_var == "LQFRAC":
                            var_tmp_CFRZR = id_tmp2.variables['CFRZR_surface'][0, :, :]
                            var_tmp_CICEP = id_tmp2.variables['CICEP_surface'][0, :, :]
                            var_tmp_CSNOW = id_tmp2.variables['CSNOW_surface'][0, :, :]
                            var_tmp_CRAIN = id_tmp2.variables['CRAIN_surface'][0, :, :]

                            var_tmp = var_tmp_CRAIN / (var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+1)
                            var_tmp = np.where(var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+var_tmp_CRAIN == 0, np.nan, var_tmp)
                        else:
                            var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                            if grib_var in ("APCP",):
                                var_tmp /= 3600
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'LQFRAC':
                    mask[np.isnan(var_sub_tmp)] = 0

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local RAP array into ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding Input RAP Field: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid RAP variable: " + input_forcings.netcdf_var_names[force_count] + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    input_forcings.esmf_field_out.data[np.isnan(input_forcings.esmf_field_out.data)] = -50
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to run mask calculation on RAP variable: " + input_forcings.netcdf_var_names[force_count] + " (" + str(npe) + ")"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place RAP ESMF data into local gridded array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        if grib_var == "LQFRAC":
                            var_tmp_CFRZR = id_tmp2.variables['CFRZR_surface'][0, :, :]
                            var_tmp_CICEP = id_tmp2.variables['CICEP_surface'][0, :, :]
                            var_tmp_CSNOW = id_tmp2.variables['CSNOW_surface'][0, :, :]
                            var_tmp_CRAIN = id_tmp2.variables['CRAIN_surface'][0, :, :]

                            var_tmp = var_tmp_CRAIN / (var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+1)
                            var_tmp = np.where(var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+var_tmp_CRAIN == 0, np.nan, var_tmp)
                        else:
                            var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                            if grib_var in ("APCP",):
                                var_tmp /= 3600
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'LQFRAC':
                    mask[np.isnan(var_sub_tmp)] = 0

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local RAP array into ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding Input RAP Field: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid RAP variable: " + input_forcings.netcdf_var_names[force_count] + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    input_forcings.esmf_field_out.data[np.isnan(input_forcings.esmf_field_out.data)] = -50
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to run mask calculation on RAP variable: " + input_forcings.netcdf_var_names[force_count] + " (" + str(npe) + ")"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place RAP ESMF data into local unstructured node array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)

                var_tmp_elem = None
                if mpi_config.rank == 0:
                    try:
                        if grib_var == "LQFRAC":
                            var_tmp_CFRZR = id_tmp2.variables['CFRZR_surface'][0, :, :]
                            var_tmp_CICEP = id_tmp2.variables['CICEP_surface'][0, :, :]
                            var_tmp_CSNOW = id_tmp2.variables['CSNOW_surface'][0, :, :]
                            var_tmp_CRAIN = id_tmp2.variables['CRAIN_surface'][0, :, :]

                            var_tmp_elem = var_tmp_CRAIN / (var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+1)
                            var_tmp_elem = np.where(var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+var_tmp_CRAIN == 0, np.nan, var_tmp_elem)
                        else:
                            var_tmp_elem = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                            if grib_var in ("APCP",):
                                var_tmp_elem /= 3600
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                mask_elem = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask_elem = np.copy(mask_elem)
                if grib_var == 'LQFRAC':
                    mask_elem[np.isnan(var_sub_tmp_elem)] = 0

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local RAP array into ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding Input RAP Field: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid RAP variable: " + input_forcings.netcdf_var_names[force_count] + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = config_options.globalNdv
                    input_forcings.esmf_field_out_elem.data[np.isnan(input_forcings.esmf_field_out_elem.data)] = -50
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to run mask calculation on RAP variable: " + input_forcings.netcdf_var_names[force_count] + " (" + str(npe) + ")"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place RAP ESMF data into local unstructured elem array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)

                mask[:] = prev_mask
                mask_elem[:] = prev_mask_elem

            elif config_options.grid_type == "hydrofabric":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        if grib_var == "LQFRAC":
                            var_tmp_CFRZR = id_tmp2.variables['CFRZR_surface'][0, :, :]
                            var_tmp_CICEP = id_tmp2.variables['CICEP_surface'][0, :, :]
                            var_tmp_CSNOW = id_tmp2.variables['CSNOW_surface'][0, :, :]
                            var_tmp_CRAIN = id_tmp2.variables['CRAIN_surface'][0, :, :]

                            var_tmp = var_tmp_CRAIN / (var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+1)
                            var_tmp = np.where(var_tmp_CFRZR+var_tmp_CSNOW+var_tmp_CICEP+var_tmp_CRAIN == 0, np.nan, var_tmp)
                        else:
                            var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                            if grib_var in ("APCP",):
                                var_tmp /= 3600
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'LQFRAC':
                    mask[np.isnan(var_sub_tmp)] = 0

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local RAP array into ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding Input RAP Field: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid RAP variable: " + input_forcings.netcdf_var_names[force_count] + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    input_forcings.esmf_field_out.data[np.isnan(input_forcings.esmf_field_out.data)] = -50
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to run mask calculation on RAP variable: " + input_forcings.netcdf_var_names[force_count] + " (" + str(npe) + ")"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place RAP ESMF data into local hydrofaborc elem array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)

                mask[:] = prev_mask

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = f"Unable to close NetCDF file: {input_forcings.tmpFile} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile)
            except FileNotFoundError:
                config_options.statusMsg = f"NetCDF file not found, continuing: {input_forcings.tmpFile}"
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = f"Unable to remove NetCDF file: {input_forcings.tmpFile} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if mpi_config.rank == 0 and id_tmp2 is not None:
            try:
                id_tmp2.close()
            except Exception as e:
                config_options.errMsg = f"Unable to close NetCDF file: {input_forcings.tmpFile2} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile2)
            except FileNotFoundError:
                config_options.statusMsg = f"NetCDF file not found, continuing: {input_forcings.tmpFile2}"
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = f"Unable to remove NetCDF file: {input_forcings.tmpFile2} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

def regrid_cfsv2(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid global CFSv2 forecast data."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if not os.path.isfile(input_forcings.file_in2):
        return

    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No need to read in new CFSv2 data at this time."
            err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
        return

    file_name = f"CFSv2_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    input_forcings.tmpFile = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    err_handler.check_program_status(config_options, mpi_config)

    id_tmp = None
    try:
        config_options.statusMsg = "Regrid CFSv2"
        err_handler.log_msg(config_options, mpi_config)
        if input_forcings.file_type != NETCDF:
            if mpi_config.rank == 0:
                if os.path.isfile(input_forcings.tmpFile):
                    config_options.statusMsg = "Found old temporary file: " + input_forcings.tmpFile + " - Removing....."
                    err_handler.log_warning(config_options, mpi_config)
                    try:
                        os_utils.os_remove_retry(input_forcings.tmpFile)
                    except OSError as err:
                        config_options.errMsg = "Unable to remove previous temporary file: " + input_forcings.tmpFile + str(err)
                        err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Converting CFSv2 Variable: " + grib_var
                    err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                fields.append(":" + grib_var + ":" + input_forcings.grib_levels[force_count] + ":" + str(input_forcings.fcst_hour2) + " hour fcst:")
            fields.append(":(HGT):(surface):")

            if WGRIB2_env:
                cmd = '$WGRIB2 -match "(' + "|".join(fields) + ')" ' + input_forcings.file_in2 + " -netcdf " + input_forcings.tmpFile
            else:
                cmd = "(" + "|".join(fields) + ")"

            id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd, config_options, mpi_config, inputVar=None, special_case=False)
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link("CFSv2", input_forcings.file_in2, input_forcings.tmpFile, config_options, mpi_config)
            id_tmp = ioMod.open_netcdf_forcing(input_forcings.tmpFile, config_options, mpi_config)

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Processing CFSv2 Variable: " + grib_var
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level

            calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Calculate CFSv2 regridding weights."
                    err_handler.log_msg(config_options, mpi_config, True)  # log at debug level

                calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.grid_type == "gridded":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract HGT_surface from file: " + input_forcings.file_in2 + " (" + str(err) + ")"
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place CFSv2 elevation data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding CFSv2 elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)  # log at debug level

                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid CFSv2 elevation data to the WRF-Hydro domain: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to run mask calculation on CFSv2 elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract CFSv2 regridded elevation data from ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "unstructured":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract HGT_surface from file: " + input_forcings.file_in2 + " (" + str(err) + ")"
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place CFSv2 elevation data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding CFSv2 elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)  # log at debug level

                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid CFSv2 elevation data to the WRF-Hydro domain: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to run mask calculation on CFSv2 elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract CFSv2 regridded elevation data from ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_tmp_elem = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract HGT_surface from file: " + input_forcings.file_in2 + " (" + str(err) + ")"
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place CFSv2 elevation data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding CFSv2 elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)  # log at debug level

                    try:
                        input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid CFSv2 elevation data to the WRF-Hydro domain: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to run mask calculation on CFSv2 elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = input_forcings.esmf_field_out_elem.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract CFSv2 regridded elevation data from ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract HGT_surface from file: " + input_forcings.file_in2 + " (" + str(err) + ")"
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place CFSv2 elevation data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding CFSv2 elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)  # log at debug level

                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid CFSv2 elevation data to the WRF-Hydro domain: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to run mask calculation on CFSv2 elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract CFSv2 regridded elevation data from ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                var_tmp = None
                if mpi_config.rank == 0:
                    if not config_options.runCfsNldasBiasCorrect:
                        config_options.statusMsg = "Regridding CFSv2 variable: " + input_forcings.netcdf_var_names[force_count]
                        err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                    try:
                        var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + " from file: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.runCfsNldasBiasCorrect:
                    if input_forcings.coarse_input_forcings1 is None:
                        input_forcings.coarse_input_forcings1 = np.empty([9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64)

                    if input_forcings.coarse_input_forcings2 is None:
                        input_forcings.coarse_input_forcings2 = np.empty([9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64)

                    try:
                        input_forcings.coarse_input_forcings2[input_forcings.input_map_output[force_count], :, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place local CFSv2 input variable: " + input_forcings.netcdf_var_names[force_count] + " into local numpy array. (" + str(err) + ")"

                    if config_options.current_output_step == 1:
                        input_forcings.coarse_input_forcings1[input_forcings.input_map_output[force_count], :, :] = input_forcings.coarse_input_forcings2[input_forcings.input_map_output[force_count], :, :]
                else:
                    input_forcings.coarse_input_forcings2 = None
                    input_forcings.coarse_input_forcings1 = None
                err_handler.check_program_status(config_options, mpi_config)

                if not config_options.runCfsNldasBiasCorrect:
                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place CFSv2 forcing data into temporary ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid CFSv2 variable: " + input_forcings.netcdf_var_names[force_count] + " (" + str(ve) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to run mask calculation on CFSv2 variable: " + input_forcings.netcdf_var_names[force_count] + " (" + str(npe) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract ESMF field data for CFSv2: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if config_options.current_output_step == 1:
                        input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = config_options.globalNdv
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = config_options.globalNdv

            elif config_options.grid_type == "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    if not config_options.runCfsNldasBiasCorrect:
                        config_options.statusMsg = "Regridding CFSv2 variable: " + input_forcings.netcdf_var_names[force_count]
                        err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                    try:
                        var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + " from file: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.runCfsNldasBiasCorrect:
                    if input_forcings.coarse_input_forcings1 is None:
                        input_forcings.coarse_input_forcings1 = np.empty([9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64)

                    if input_forcings.coarse_input_forcings2 is None:
                        input_forcings.coarse_input_forcings2 = np.empty([9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64)

                    try:
                        input_forcings.coarse_input_forcings2[input_forcings.input_map_output[force_count], :, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place local CFSv2 input variable: " + input_forcings.netcdf_var_names[force_count] + " into local numpy array. (" + str(err) + ")"

                    if config_options.current_output_step == 1:
                        input_forcings.coarse_input_forcings1[input_forcings.input_map_output[force_count], :, :] = input_forcings.coarse_input_forcings2[input_forcings.input_map_output[force_count], :, :]
                else:
                    input_forcings.coarse_input_forcings2 = None
                    input_forcings.coarse_input_forcings1 = None
                err_handler.check_program_status(config_options, mpi_config)

                if not config_options.runCfsNldasBiasCorrect:
                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place CFSv2 forcing data into temporary ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid CFSv2 variable: " + input_forcings.netcdf_var_names[force_count] + " (" + str(ve) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to run mask calculation on CFSv2 variable: " + input_forcings.netcdf_var_names[force_count] + " (" + str(npe) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract ESMF field data for CFSv2: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if config_options.current_output_step == 1:
                        input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = config_options.globalNdv
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = config_options.globalNdv

                var_tmp_elem = None
                if mpi_config.rank == 0:
                    if not config_options.runCfsNldasBiasCorrect:
                        config_options.statusMsg = "Regridding CFSv2 variable: " + input_forcings.netcdf_var_names[force_count]
                        err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                    try:
                        var_tmp_elem = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + " from file: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.runCfsNldasBiasCorrect:
                    if input_forcings.coarse_input_forcings1_elem is None:
                        input_forcings.coarse_input_forcings1_elem = np.empty([9, var_sub_tmp_elem.shape[0], var_sub_tmp_elem.shape[1]], np.float64)

                    if input_forcings.coarse_input_forcings2_elem is None:
                        input_forcings.coarse_input_forcings2_elem = np.empty([9, var_sub_tmp_elem.shape[0], var_sub_tmp_elem.shape[1]], np.float64)

                    try:
                        input_forcings.coarse_input_forcings2_elem[input_forcings.input_map_output[force_count], :, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place local CFSv2 input variable: " + input_forcings.netcdf_var_names[force_count] + " into local numpy array. (" + str(err) + ")"

                    if config_options.current_output_step == 1:
                        input_forcings.coarse_input_forcings1_elem[input_forcings.input_map_output[force_count], :, :] = input_forcings.coarse_input_forcings2_elem[input_forcings.input_map_output[force_count], :, :]
                else:
                    input_forcings.coarse_input_forcings2_elem = None
                    input_forcings.coarse_input_forcings1_elem = None
                err_handler.check_program_status(config_options, mpi_config)

                if not config_options.runCfsNldasBiasCorrect:
                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place CFSv2 forcing data into temporary ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid CFSv2 variable: " + input_forcings.netcdf_var_names[force_count] + " (" + str(ve) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to run mask calculation on CFSv2 variable: " + input_forcings.netcdf_var_names[force_count] + " (" + str(npe) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out_elem.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract ESMF field data for CFSv2: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if config_options.current_output_step == 1:
                        input_forcings.regridded_forcings1_elem[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :]
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    input_forcings.regridded_forcings1_elem[input_forcings.input_map_output[force_count], :] = config_options.globalNdv
                    input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :] = config_options.globalNdv

            elif config_options.grid_type == "hydrofabric":
                var_tmp = None
                if mpi_config.rank == 0:
                    if not config_options.runCfsNldasBiasCorrect:
                        config_options.statusMsg = "Regridding CFSv2 variable: " + input_forcings.netcdf_var_names[force_count]
                        err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                    try:
                        var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + " from file: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.runCfsNldasBiasCorrect:
                    if input_forcings.coarse_input_forcings1 is None:
                        input_forcings.coarse_input_forcings1 = np.empty([9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64)

                    if input_forcings.coarse_input_forcings2 is None:
                        input_forcings.coarse_input_forcings2 = np.empty([9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64)

                    try:
                        input_forcings.coarse_input_forcings2[input_forcings.input_map_output[force_count], :, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place local CFSv2 input variable: " + input_forcings.netcdf_var_names[force_count] + " into local numpy array. (" + str(err) + ")"

                    if config_options.current_output_step == 1:
                        input_forcings.coarse_input_forcings1[input_forcings.input_map_output[force_count], :, :] = input_forcings.coarse_input_forcings2[input_forcings.input_map_output[force_count], :, :]
                else:
                    input_forcings.coarse_input_forcings2 = None
                    input_forcings.coarse_input_forcings1 = None
                err_handler.check_program_status(config_options, mpi_config)

                if not config_options.runCfsNldasBiasCorrect:
                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place CFSv2 forcing data into temporary ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid CFSv2 variable: " + input_forcings.netcdf_var_names[force_count] + " (" + str(ve) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to run mask calculation on CFSv2 variable: " + input_forcings.netcdf_var_names[force_count] + " (" + str(npe) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract ESMF field data for CFSv2: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if config_options.current_output_step == 1:
                        input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = config_options.globalNdv
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = config_options.globalNdv

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = f"Unable to close NetCDF file: {input_forcings.tmpFile} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile)
            except FileNotFoundError:
                config_options.statusMsg = f"NetCDF file not found, continuing: {input_forcings.tmpFile}"
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = f"Unable to remove NetCDF file: {input_forcings.tmpFile} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)


def regrid_hourly_nbm(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid NBM 2.5km data."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if not os.path.isfile(input_forcings.file_in2) or input_forcings.regridComplete:
        return

    file_uuid = str(mpi_config.uid64)
    input_forcings.tmpFile = str(Path(config_options.scratch_dir) / f"{file_uuid}_NBM_TMP-{mkfilename()}.nc")
    err_handler.check_program_status(config_options, mpi_config)
    id_tmp = None

    try:
        config_options.statusMsg = "Regrid NBM"
        err_handler.log_msg(config_options, mpi_config)
        if input_forcings.file_type != NETCDF:
            if mpi_config.rank == 0 and os.path.isfile(input_forcings.tmpFile):
                config_options.statusMsg = f"Found old temporary file: {input_forcings.tmpFile} - Removing....."
                err_handler.log_warning(config_options, mpi_config)
                try:
                    os_utils.os_remove_retry(input_forcings.tmpFile)
                except OSError:
                    config_options.errMsg = f"Unable to remove file: {input_forcings.tmpFile}"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            fields = [f":{gv}:{input_forcings.grib_levels[i]}:{(f'{input_forcings.fcst_hour1}-{input_forcings.fcst_hour2} hour acc fcst' if gv == 'APCP' else f'{input_forcings.fcst_hour2} hour fcst')}:" for i, gv in enumerate(input_forcings.grib_vars)]
            fields.append(":(HGT):(surface):")

            cmd = f'$WGRIB2 -match "({"|".join(fields)})" {input_forcings.file_in2} -netcdf {input_forcings.tmpFile}' if WGRIB2_env else f'({"|".join(fields)})'
            id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd, config_options, mpi_config, inputVar=None, special_case=False)
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link("NBM", input_forcings.file_in2, input_forcings.tmpFile, config_options, mpi_config)
            id_tmp = ioMod.open_netcdf_forcing(input_forcings.tmpFile, config_options, mpi_config)

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if check_regrid_status(id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
                calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta)

                var_tmp = id_tmp.variables["HGT_surface"][0, :, :] if mpi_config.rank == 0 else None
                err_handler.check_program_status(config_options, mpi_config)

                input_forcings.esmf_field_in.data[:, :] = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                input_forcings.height = input_forcings.esmf_field_out.data

            var_tmp = None
            if mpi_config.rank == 0:
                try:
                    var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    if grib_var == "APCP":
                        var_tmp /= 3600.0
                except Exception as err:
                    config_options.errMsg = f"Unable to extract {input_forcings.netcdf_var_names[force_count]}: {err}"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            input_forcings.esmf_field_in.data[:, :] = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv

            idx = input_forcings.input_map_output[force_count]
            input_forcings.regridded_forcings2[idx, :, :] = input_forcings.esmf_field_out.data
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[idx, :, :] = input_forcings.regridded_forcings2[idx, :, :]

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception:
                pass
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile)
            except FileNotFoundError:
                pass
        err_handler.check_program_status(config_options, mpi_config)


def regrid_ndfd(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid NDFD Data."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if not os.path.isfile(input_forcings.file_in2) or input_forcings.regridComplete:
        return

    file_uuid = str(mpi_config.uid64)
    input_forcings.tmpFile = str(Path(config_options.scratch_dir) / f"{file_uuid}_NDFD_TMP-{mkfilename()}.nc")
    err_handler.check_program_status(config_options, mpi_config)
    id_tmp = None

    try:
        config_options.statusMsg = "Regrid NDFD"
        err_handler.log_msg(config_options, mpi_config)
        if input_forcings.file_type != NETCDF:
            if mpi_config.rank == 0 and os.path.isfile(input_forcings.tmpFile):
                config_options.statusMsg = f"Found old temporary file: {input_forcings.tmpFile} - Removing....."
                err_handler.log_warning(config_options, mpi_config)
                try:
                    os_utils.os_remove_retry(input_forcings.tmpFile)
                except OSError:
                    config_options.errMsg = f"Unable to remove file: {input_forcings.tmpFile}"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            fields = [f":{gv}:{input_forcings.grib_levels[i]}:{(f'{input_forcings.fcst_hour1}-{input_forcings.fcst_hour2} hour acc fcst' if gv == 'APCP' else f'{input_forcings.fcst_hour2} hour fcst')}:" for i, gv in enumerate(input_forcings.grib_vars)]
            fields.append(":(HGT):(surface):")

            cmd = f'$WGRIB2 -match "({"|".join(fields)})" {input_forcings.file_in2} -netcdf {input_forcings.tmpFile}' if WGRIB2_env else f'({"|".join(fields)})'
            id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd, config_options, mpi_config, inputVar=None, special_case=False)
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link("NDFD", input_forcings.file_in2, input_forcings.tmpFile, config_options, mpi_config)
            id_tmp = ioMod.open_netcdf_forcing(input_forcings.tmpFile, config_options, mpi_config)

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if check_regrid_status(id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
                calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta)

                var_tmp = id_tmp.variables["HGT_surface"][0, :, :] if mpi_config.rank == 0 else None
                err_handler.check_program_status(config_options, mpi_config)

                input_forcings.esmf_field_in.data[:, :] = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                input_forcings.height = input_forcings.esmf_field_out.data

            var_tmp = None
            if mpi_config.rank == 0:
                try:
                    var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    if grib_var == "APCP":
                        var_tmp /= 3600.0
                except Exception as err:
                    config_options.errMsg = f"Unable to extract {input_forcings.netcdf_var_names[force_count]}: {err}"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            input_forcings.esmf_field_in.data[:, :] = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv

            idx = input_forcings.input_map_output[force_count]
            input_forcings.regridded_forcings2[idx, :, :] = input_forcings.esmf_field_out.data
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[idx, :, :] = input_forcings.regridded_forcings2[idx, :, :]

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception:
                pass
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile)
            except FileNotFoundError:
                pass
        err_handler.check_program_status(config_options, mpi_config)

def regrid_nwm(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid custom input NetCDF hourly forcing files.

    Function for handling regridding of custom input NetCDF hourly forcing files.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    ## Flag to jump to different regridding module for AWS NWM Forcing data
    if config_options.aws:
        regrid_nwm_aws(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config)
        return

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No NWM NetCDF regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
        return

    # Open the input NetCDF file containing necessary data.
    id_tmp = ioMod.open_netcdf_forcing(input_forcings.file_in2, config_options, mpi_config, open_on_all_procs=True)

    config_options.statusMsg = "Regrid NWM Custom NetCDF Forcing Variables"
    err_handler.log_msg(config_options, mpi_config)

    for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Processing Custom NetCDF Forcing Variable: " + nc_var
            err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
        calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config)

        if calc_regrid_flag:
            calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta)

        input_forcings.height = None
        if mpi_config.rank == 0:
            config_options.statusMsg = f"Unable to locate HGT_surface in: {input_forcings.file_in2}. Downscaling will not be available."
            err_handler.log_msg(config_options, mpi_config, True)  # log at debug level

        if config_options.grid_type == "gridded":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding Custom netCDF input variable: " + nc_var
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                try:
                    var_tmp = id_tmp.variables[nc_var][:][0, :, :]
                except Exception as err:
                    config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input Custom netCDF forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding Custom netCDF input variable: " + nc_var
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                try:
                    var_tmp = id_tmp.variables[nc_var][:][0, :, :]
                except Exception as err:
                    config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input Custom netCDF forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the input variables.
            var_tmp_elem = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding Custom netCDF input variable: " + nc_var
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                try:
                    var_tmp_elem = id_tmp.variables[nc_var][:][0, :, :]
                except Exception as err:
                    config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input Custom netCDF forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out_elem.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1_elem[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding Custom netCDF input variable: " + nc_var
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                try:
                    var_tmp = id_tmp.variables[nc_var][:][0, :, :]
                except Exception as err:
                    config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input Custom netCDF forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
            err_handler.check_program_status(config_options, mpi_config)
    # Close the NetCDF file
    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
            err_handler.err_out(config_options)


def regrid_nwm_aws(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid AWS NWM Forcing Data Downloaded from Server.

    Function for handling regridding of AWS NWM Forcing Data Downloaded from Server.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No NWM NetCDF regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
        return
    mpi_config.comm.barrier()
    with MPICommExecutor(comm=mpi_config.comm, root=0) as executor:
        with dask.config.set(scheduler=executor):
            if mpi_config.rank == 0:
                id_tmp = config_options.aws_obj
            else:
                id_tmp = None
    mpi_config.comm.barrier()

    # Convert projected coordinates to geographic
    if mpi_config.rank == 0 and id_tmp is not None:
        if "x" in id_tmp.coords and "y" in id_tmp.coords:
            nwm_crs = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
            transformer = Transformer.from_crs(nwm_crs, "EPSG:4326", always_xy=True)

            x_coords, y_coords = np.meshgrid(id_tmp.x.values, id_tmp.y.values)
            lon_coords, lat_coords = transformer.transform(x_coords, y_coords)

            id_tmp = id_tmp.assign_coords(longitude=(["y", "x"], lon_coords), latitude=(["y", "x"], lat_coords))

    config_options.statusMsg = "Regrid NWM Custom zarr Forcing Variables"
    err_handler.log_msg(config_options, mpi_config)

    for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Processing Custom zarr Forcing Variable: " + nc_var
            err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
        calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config)
        if calc_regrid_flag:
            calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta)

        input_forcings.height = None
        if mpi_config.rank == 0:
            config_options.statusMsg = f"Unable to locate HGT_surface in: {input_forcings.file_in2}. Downscaling will not be available."
            err_handler.log_msg(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding Custom zarr input variable: " + nc_var
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                try:
                    var_tmp = id_tmp[nc_var].to_masked_array()
                except Exception as err:
                    config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input Custom zarr forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding Custom zarr input variable: " + nc_var
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                try:
                    var_tmp = id_tmp[nc_var].to_masked_array()
                except Exception as err:
                    config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input Custom zarr forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the input variables.
            var_tmp_elem = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding Custom zarr input variable: " + nc_var
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                try:
                    var_tmp_elem = id_tmp[nc_var].to_masked_array()
                except Exception as err:
                    config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input Custom zarr forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out_elem.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1_elem[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding Custom zarr input variable: " + nc_var
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                try:
                    var_tmp = id_tmp[nc_var].to_masked_array()
                except Exception as err:
                    config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # convert to array for NWM
            if input_forcings.product_name == "NWM":
                var_tmp = np.asarray(var_tmp, dtype=np.float64)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input Custom zarr forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
            err_handler.check_program_status(config_options, mpi_config)
    # Close the NetCDF file
    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
            err_handler.err_out(config_options)

def regrid_custom_hourly_netcdf(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid Custom Hourly NetCDF Forcing Data.

    Function for handling regridding of custom input NetCDF hourly forcing files.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    with timing_block("Regrid AORC AWS"):
        # Flag to jump to different regridding function soley for AORC AWS data
        if config_options.aws:
            regrid_aorc_aws(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config)
            return

    with timing_block("Regrid Custom Hourly NetCDF Forcing Data"):
        # If the expected file is missing, this means we are allowing missing files, simply
        # exit out of this routine as the regridded fields have already been set to NDV.
        if not os.path.isfile(input_forcings.file_in2):
            return

        # Check to see if the regrid complete flag for this
        # output time step is true. This entails the necessary
        # inputs have already been regridded and we can move on.
        if input_forcings.regridComplete:
            if mpi_config.rank == 0:
                config_options.statusMsg = "No Custom Hourly NetCDF regridding required for this timestep."
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
            return

        # Open the input NetCDF file containing necessary data.
        id_tmp = ioMod.open_netcdf_forcing(input_forcings.file_in2, config_options, mpi_config, open_on_all_procs=True)

        fill_values = {"TMP": 288.0, "SPFH": 0.005, "PRES": 101300.0, "APCP": 0, "UGRD": 1.0, "VGRD": 1.0, "DSWRF": 80.0, "DLWRF": 310.0}

        config_options.statusMsg = "Regrid Custom Hourly NetCDF Forcing Variables"
        err_handler.log_msg(config_options, mpi_config)

        for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Processing Custom NetCDF Forcing Variable: " + nc_var
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
            calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config)

            if calc_regrid_flag:
                calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta)

                # Flag to set regridded mask for AORC to overlay with ERA5-Interim blend
                if 23 in config_options.input_forcings:
                    input_forcings.regridded_mask_AORC = input_forcings.regridded_mask
                    if config_options.grid_type == "unstructured":
                        input_forcings.regridded_mask_elem_AORC = input_forcings.regridded_mask_elem

                # Read in the RAP height field, which is used for downscaling purposes.
                if "HGT_surface" in id_tmp.variables.keys():
                    if config_options.grid_type == "gridded":
                        # Regrid the height variable.
                        if mpi_config.rank == 0:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        else:
                            var_tmp = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to place NetCDF elevation data into the ESMF field object: " + str(err)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            config_options.statusMsg = "Regridding elevation data to the WRF-Hydro domain."
                            err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                        try:
                            input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                        except ValueError as ve:
                            config_options.errMsg = "Unable to regrid elevation data to the WRF-Hydro domain using ESMF: " + str(ve)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        # Set any pixel cells outside the input domain to the global missing value.
                        try:
                            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            config_options.errMsg = "Unable to compute mask on elevation data: " + str(npe)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract ESMF regridded elevation data to a local array: " + str(err)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                    elif config_options.grid_type == "unstructured":
                        # Regrid the height variable.
                        if mpi_config.rank == 0:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        else:
                            var_tmp = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to place NetCDF elevation data into the ESMF field object: " + str(err)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            config_options.statusMsg = "Regridding elevation data to the WRF-Hydro domain."
                            err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                        try:
                            input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                        except ValueError as ve:
                            config_options.errMsg = "Unable to regrid elevation data to the WRF-Hydro domain using ESMF: " + str(ve)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        # Set any pixel cells outside the input domain to the global missing value.
                        try:
                            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            config_options.errMsg = "Unable to compute mask on elevation data: " + str(npe)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.height[:] = input_forcings.esmf_field_out.data
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract ESMF regridded elevation data to a local array: " + str(err)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        # Regrid the height variable.
                        if mpi_config.rank == 0:
                            var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                        else:
                            var_tmp_elem = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to place NetCDF elevation data into the ESMF field object: " + str(err)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            config_options.statusMsg = "Regridding elevation data to the WRF-Hydro domain."
                            err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                        try:
                            input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
                        except ValueError as ve:
                            config_options.errMsg = "Unable to regrid elevation data to the WRF-Hydro domain using ESMF: " + str(ve)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        # Set any pixel cells outside the input domain to the global missing value.
                        try:
                            input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            config_options.errMsg = "Unable to compute mask on elevation data: " + str(npe)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.height_elem[:] = input_forcings.esmf_field_out_elem.data
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract ESMF regridded elevation data to a local array: " + str(err)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                    elif config_options.grid_type == "hydrofabric":
                        # Regrid the height variable.
                        if mpi_config.rank == 0:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        else:
                            var_tmp = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to place NetCDF elevation data into the ESMF field object: " + str(err)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            config_options.statusMsg = "Regridding elevation data to the WRF-Hydro domain."
                            err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                        try:
                            input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                        except ValueError as ve:
                            config_options.errMsg = "Unable to regrid elevation data to the WRF-Hydro domain using ESMF: " + str(ve)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        # Set any pixel cells outside the input domain to the global missing value.
                        try:
                            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            config_options.errMsg = "Unable to compute mask on elevation data: " + str(npe)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.height[:] = input_forcings.esmf_field_out.data
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract ESMF regridded elevation data to a local array: " + str(err)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                else:
                    input_forcings.height = None
                    if mpi_config.rank == 0:
                        config_options.statusMsg = f"Unable to locate HGT_surface in: {input_forcings.file_in2}. Downscaling will not be available."
                        err_handler.log_msg(config_options, mpi_config)

                # close netCDF file on non-root ranks
                if mpi_config.rank != 0:
                    id_tmp.close()

            if config_options.grid_type == "gridded":
                # Regrid the input variables.
                var_tmp = None
                fill = fill_values.get(input_forcings.grib_vars[force_count], config_options.globalNdv)
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding Custom netCDF input variable: " + nc_var
                    err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                    try:
                        config_options.statusMsg = f"Using {fill} to replace missing values in input"
                        err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                        var_tmp = id_tmp.variables[nc_var][:].filled(fill)[0, :, :]
                    except Exception as err:
                        config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input Custom netCDF forcing variables using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = fill
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to calculate mask from input Custom netCDF regridded forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                # Convert the hourly precipitation total to a rate of mm/s
                if nc_var == "APCP_surface":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != fill)
                        # Flag to set regridded mask for AORC to overlay with ERA5-Interim blend
                        input_forcings.esmf_field_out.data[ind_valid] = input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        del ind_valid
                    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                        config_options.errMsg = "Unable to run NDV search on Custom netCDF precipitation: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                # Regrid the input variables.
                var_tmp = None
                fill = fill_values.get(input_forcings.grib_vars[force_count], config_options.globalNdv)
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding Custom netCDF input variable: " + nc_var
                    err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                    try:
                        config_options.statusMsg = f"Using {fill} to replace missing values in input"
                        err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                        var_tmp = id_tmp.variables[nc_var][:].filled(fill)[0, :, :]
                    except Exception as err:
                        config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input Custom netCDF forcing variables using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = fill
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to calculate mask from input Custom netCDF regridded forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                # Convert the hourly precipitation total to a rate of mm/s
                if nc_var == "APCP_surface":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != fill)
                        input_forcings.esmf_field_out.data[ind_valid] = input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        del ind_valid
                    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                        config_options.errMsg = "Unable to run NDV search on Custom netCDF precipitation: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)

                # Regrid the input variables.
                var_tmp_elem = None
                fill = fill_values.get(input_forcings.grib_vars[force_count], config_options.globalNdv)
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding Custom netCDF input variable: " + nc_var
                    err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                    try:
                        config_options.statusMsg = f"Using {fill} to replace missing values in input"
                        err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                        var_tmp_elem = id_tmp.variables[nc_var][:].filled(fill)[0, :, :]
                    except Exception as err:
                        config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input Custom netCDF forcing variables using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = fill
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to calculate mask from input Custom netCDF regridded forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                # Convert the hourly precipitation total to a rate of mm/s
                if nc_var == "APCP_surface":
                    try:
                        ind_valid_elem = np.where(input_forcings.esmf_field_out_elem.data != fill)
                        input_forcings.esmf_field_out_elem.data[ind_valid_elem] = input_forcings.esmf_field_out_elem.data[ind_valid_elem] / 3600.0
                        del ind_valid_elem
                    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                        config_options.errMsg = "Unable to run NDV search on Custom netCDF precipitation: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "hydrofabric":
                # Regrid the input variables.
                var_tmp = None
                fill = fill_values.get(input_forcings.grib_vars[force_count], config_options.globalNdv)
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding Custom netCDF input variable: " + nc_var
                    err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                    try:
                        config_options.statusMsg = f"Using {fill} to replace missing values in input"
                        err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                        var_tmp = id_tmp.variables[nc_var][:].filled(fill)[0, :, :]
                    except Exception as err:
                        config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)
                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input Custom netCDF forcing variables using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = fill
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to calculate mask from input Custom netCDF regridded forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if nc_var == "DSWRF_surface":
                    ind_valid = np.where(input_forcings.esmf_field_out.data < 0.0)
                    input_forcings.esmf_field_out.data[ind_valid] = 0.0
                # Convert the hourly precipitation total to a rate of mm/s
                if nc_var == "APCP_surface":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != fill)
                        input_forcings.esmf_field_out.data[ind_valid] = input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        ind_valid = np.where(input_forcings.esmf_field_out.data < 0.0)
                        input_forcings.esmf_field_out.data[ind_valid] = 0.0
                        del ind_valid
                    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                        config_options.errMsg = "Unable to run NDV search on Custom netCDF precipitation: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)
        # Close the NetCDF file
        if mpi_config.rank == 0:
            try:
                id_tmp.close()
            except OSError:
                config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
                err_handler.err_out(config_options)


def regrid_era5(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Rebgrid ERA5-Interim Forcing Variables.

    Function for handling regridding of a single ERA5 netcdf custom file.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in1):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No ERA5-Interim regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
        return

    # Open the input NetCDF file containing necessary data.
    id_tmp = ioMod.open_netcdf_forcing(input_forcings.file_in2, config_options, mpi_config, open_on_all_procs=True)

    # create netcdf time stamp
    time = nc.num2date(id_tmp.variables["time"][:].data, units=id_tmp.variables["time"].units, only_use_cftime_datetimes=False)
    # Find the timestamp index based on latest timestamp requested
    # by the forcings engine
    seconds_index = np.abs((pd.to_datetime(time) - pd.to_datetime(config_options.current_time)).total_seconds())
    ind = np.where(seconds_index == np.min(seconds_index))[0][0]

    config_options.statusMsg = "Regrid Custom Hourly NetCDF Forcing Variables"
    err_handler.log_msg(config_options, mpi_config)

    for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Processing ERA-5 Interim Forcing Variable: " + nc_var
            err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
        calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config)
        if calc_regrid_flag:
            calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta)

            # Read in the ERA5-Interim height field, which is used for downscaling purposes.
            if "Geopotential" in id_tmp.variables.keys():
                LOG.info("Found geopotential height in ERA5-Interim data")
            # To Do, see if NCAR downscaling methods are applicable
            # for reanalysis datasets with coarser resolution
            else:
                input_forcings.height = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Unable to locate Geopoential height in: {input_forcings.file_in2}. Downscaling will not be available."
                    err_handler.log_msg(config_options, mpi_config)

            # close netCDF file on non-root ranks
            if mpi_config.rank != 0:
                id_tmp.close()

        if config_options.grid_type == "gridded":
            # Regrid the input variables.
            var_tmp = None

            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding ERA5-Interim input variable: " + nc_var
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                try:
                    config_options.statusMsg = "Using -9999. to replace missing values in input"
                    err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                    var_tmp = id_tmp.variables[nc_var][:].filled(-9999.0)[ind, :, :]
                    # Since ERA5-Interim only supplies dew points
                    # we will need to calculate the specific humidity
                    if nc_var == "d2m":
                        var_tmp = var_tmp - 273.15
                        e = 6.112 * np.exp((17.67 * var_tmp) / (var_tmp + 243.5))
                        pres = id_tmp.variables["sp"][:].filled(-9999.0)[ind, :, :] / 100
                        var_tmp = (0.622 * e) / (pres - (0.378 * e))
                        del e
                        del pres
                except Exception as err:
                    config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input ERA5-Interim forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = -9999.0
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to calculate mask from input ERA5-Interim regridded forcings: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total to a rate of mm/s
            if nc_var == "mtpr":
                try:
                    ind_valid = np.where(input_forcings.esmf_field_out.data != -9999.0)
                    input_forcings.esmf_field_out.data[ind_valid] = input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                    del ind_valid
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    config_options.errMsg = "Unable to run NDV search on ERA5-Interim precipitation: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding ERA5-Interim input variable: " + nc_var
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                try:
                    config_options.statusMsg = "Using -9999. to replace missing values in input"
                    err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                    var_tmp = id_tmp.variables[nc_var][:].filled(-9999.0)[ind, :, :]
                    # Since ERA5-Interim only supplies dew points
                    # we will need to calculate the specific humidity
                    if nc_var == "d2m":
                        var_tmp = var_tmp - 273.15
                        e = 6.112 * np.exp((17.67 * var_tmp) / (var_tmp + 243.5))
                        pres = id_tmp.variables["sp"][:].filled(-9999.0)[ind, :, :] / 100
                        var_tmp = (0.622 * e) / (pres - (0.378 * e))
                        del e
                        del pres
                except Exception as err:
                    config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input ERA5-Interim forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = -9999.0
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to calculate mask from input ERA5-Interim regridded forcings: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total to a rate of mm/s
            if nc_var == "mtpr":
                try:
                    ind_valid = np.where(input_forcings.esmf_field_out.data != -9999.0)
                    input_forcings.esmf_field_out.data[ind_valid] = input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                    del ind_valid
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    config_options.errMsg = "Unable to run NDV search on ERA5-Interim precipitation: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the input variables.
            var_tmp_elem = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding ERA5-Interim input variable: " + nc_var
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                try:
                    config_options.statusMsg = "Using -9999. to replace missing values in input"
                    err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                    var_tmp_elem = id_tmp.variables[nc_var][:].filled(-9999.0)[ind, :, :]
                    # Since ERA5-Interim only supplies dew points
                    # we will need to calculate the specific humidity
                    if nc_var == "d2m":
                        var_tmp_elem = var_tmp_elem - 273.15
                        e = 6.112 * np.exp((17.67 * var_tmp_elem) / (var_tmp_elem + 243.5))
                        pres = id_tmp.variables["sp"][:].filled(-9999.0)[ind, :, :] / 100
                        var_tmp = (0.622 * e) / (pres - (0.378 * e))
                        del e
                        del pres
                except Exception as err:
                    config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input ERA5-Interim forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = -9999.0
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to calculate mask from input Custom netCDF regridded forcings: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total to a rate of mm/s
            if nc_var == "mtpr":
                try:
                    ind_valid_elem = np.where(input_forcings.esmf_field_out_elem.data != -9999.0)
                    input_forcings.esmf_field_out_elem.data[ind_valid_elem] = input_forcings.esmf_field_out_elem.data[ind_valid_elem] / 3600.0
                    del ind_valid_elem
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    config_options.errMsg = "Unable to run NDV search on ERA5-Interim precipitation: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out_elem.data

            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1_elem[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding ERA5-Interim input variable: " + nc_var
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                try:
                    config_options.statusMsg = "Using -9999. to replace missing values in input"
                    err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                    var_tmp = id_tmp.variables[nc_var][:].filled(-9999.0)[ind, :, :]
                    # Since ERA5-Interim only supplies dew points
                    # we will need to calculate the specific humidity
                    if nc_var == "d2m":
                        var_tmp = var_tmp - 273.15
                        e = 6.112 * np.exp((17.67 * var_tmp) / (var_tmp + 243.5))
                        pres = id_tmp.variables["sp"][:].filled(-9999.0)[ind, :, :] / 100
                        var_tmp = (0.622 * e) / (pres - (0.378 * e))
                        del e
                        del pres
                except Exception as err:
                    config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid input ERA5-Interim forcing variables using ESMF: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = -9999.0
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to calculate mask from input ERA5-Interim regridded forcings: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total to a rate of mm/s
            if nc_var == "mtpr":
                try:
                    ind_valid = np.where(input_forcings.esmf_field_out.data != -9999.0)
                    input_forcings.esmf_field_out.data[ind_valid] = input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                    del ind_valid
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    config_options.errMsg = "Unable to run NDV search on ERA5-Interim precipitation: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
            err_handler.check_program_status(config_options, mpi_config)

    # Close the NetCDF file
    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
            err_handler.err_out(config_options)

@static_vars(last_file=None)
def regrid_gfs(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Rebgrid GFS Forcing Variables.

    Function for handing regridding of input GFS data
    from GRIB2 files.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No 13km GFS regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
        return

    file_uuid = str(mpi_config.uid64)
    file_name = f"GFS_TMP-{mkfilename()}.nc"
    input_forcings.tmpFile = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    id_tmp = None
    try:
        config_options.statusMsg = "Regridding 13km GFS Variables."
        err_handler.log_msg(config_options, mpi_config)

        if input_forcings.file_type != NETCDF:
            if mpi_config.rank == 0 and os.path.isfile(input_forcings.tmpFile):
                config_options.statusMsg = "Found old temporary file: " + input_forcings.tmpFile + " - Removing....."
                err_handler.log_warning(config_options, mpi_config)
                try:
                    os_utils.os_remove_retry(input_forcings.tmpFile)
                except OSError:
                    config_options.errMsg = "Unable to remove file: " + input_forcings.tmpFile
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Converting 13km GFS Variable: " + grib_var
                    err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                if grib_var == "PRATE":
                    if input_forcings.fcst_hour2 <= 384:
                        tmp_hr_current = input_forcings.fcst_hour2

                        diff_tmp = tmp_hr_current % 6 if tmp_hr_current % 6 > 0 else 6
                        tmp_hr_previous = tmp_hr_current - diff_tmp

                    else:
                        tmp_hr_previous = input_forcings.fcst_hour1

                    fields.append(":" + grib_var + ":" + input_forcings.grib_levels[force_count] + ":" + str(tmp_hr_previous) + "-" + str(input_forcings.fcst_hour2) + " hour ave fcst:")
                else:
                    fields.append(":" + grib_var + ":" + input_forcings.grib_levels[force_count] + ":" + str(input_forcings.fcst_hour2) + " hour fcst:")

            fields.append(":(HGT):(surface):")
            if WGRIB2_env:
                cmd = '$WGRIB2 -match "(' + "|".join(fields) + ')" ' + input_forcings.file_in2 + " -netcdf " + input_forcings.tmpFile
            else:
                cmd = "(" + "|".join(fields) + ")"

            id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd, config_options, mpi_config, inputVar=None, special_case=False)
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link("GFS", input_forcings.file_in2, input_forcings.tmpFile, config_options, mpi_config)
            id_tmp = ioMod.open_netcdf_forcing(input_forcings.tmpFile, config_options, mpi_config)

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Processing 13km GFS Variable: " + grib_var
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level

            calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Calculating 13km GFS regridding weights."
                    err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.grid_type == "gridded":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract GFS elevation from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)
                elif config_options.grid_type == "unstructured":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract GFS elevation from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_tmp_elem = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract GFS elevation from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                    err_handler.check_program_status(config_options, mpi_config)
                elif config_options.grid_type == "hydrofabric":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = "Unable to extract GFS elevation from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                            err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local GFS array into an ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding 13km GFS surface elevation data to the WRF-Hydro domain."
                    err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
                if config_options.grid_type == "gridded":
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid GFS elevation data: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to perform mask search on GFS elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "unstructured":
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid GFS elevation data with ESMF mesh nodes: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place local GFS array into an ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid GFS elevation data with ESMF mesh elements: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to perform mask search on GFS elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to perform mask search on GFS elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)
                elif config_options.grid_type == "hydrofabric":
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid GFS elevation data: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to perform mask search on GFS elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                if config_options.grid_type == "gridded":
                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract GFS elevation array from ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)
                elif config_options.grid_type == "unstructured":
                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract GFS elevation array from ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = input_forcings.esmf_field_out_elem.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract GFS elevation array from ESMF field with mesh elements: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract GFS elevation array from ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type in ("gridded", "hydrofabric"):
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_tmp_elem = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp_elem = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if input_forcings.product_name == "GFS_Production_GRIB2":
                if grib_var == "PRATE":
                    if mpi_config.rank == 0:
                        if config_options.grid_type == "gridded":
                            input_forcings.globalPcpRate2 = var_tmp
                            var_tmp = timeInterpMod.gfs_pcp_time_interp(input_forcings, config_options, mpi_config)
                        elif config_options.grid_type == "unstructured":
                            input_forcings.globalPcpRate2 = var_tmp
                            input_forcings.globalPcpRate2_elem = var_tmp_elem
                            var_tmp, var_tmp_elem = timeInterpMod.gfs_pcp_time_interp(input_forcings, config_options, mpi_config)
                        elif config_options.grid_type == "hydrofabric":
                            input_forcings.globalPcpRate2 = var_tmp
                            var_tmp = timeInterpMod.gfs_pcp_time_interp(input_forcings, config_options, mpi_config)

            if grib_var == "CPOFP":
                if mpi_config.rank == 0:
                    var_tmp[var_tmp >= 0] = (100 - var_tmp[var_tmp >= 0]) / 100
                    var_tmp[var_tmp < 0] = np.nan
                    if config_options.grid_type == "unstructured":
                        var_tmp_elem[var_tmp_elem >= 0] = (100 - var_tmp_elem[var_tmp_elem >= 0]) / 100
                        var_tmp_elem[var_tmp_elem < 0] = np.nan

            if config_options.grid_type == "gridded":
                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                mpi_config.comm.barrier()
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "unstructured":
                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                mpi_config.comm.barrier()
                err_handler.check_program_status(config_options, mpi_config)
                var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                mpi_config.comm.barrier()
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "hydrofabric":
                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                mpi_config.comm.barrier()
                err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type in ("gridded", "hydrofabric"):
                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'CPOFP':
                    mask[np.isnan(var_sub_tmp)] = 0
                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place GFS local array into ESMF element field object: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                mask = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask = np.copy(mask)
                if grib_var == 'CPOFP':
                    mask[np.isnan(var_sub_tmp)] = 0
                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place GFS local array into ESMF element field object: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                mask_elem = input_forcings.esmf_grid_in.get_item(ESMF.GridItem.MASK)
                prev_mask_elem = np.copy(mask_elem)
                if grib_var == 'CPOFP':
                    mask_elem[np.isnan(var_sub_tmp_elem)] = 0
                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place GFS local array into ESMF element field object: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding 13km GFS Variable: " + input_forcings.netcdf_var_names[force_count]
                err_handler.log_msg(config_options, mpi_config, True)  # log at debug level
            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid GFS variable: " + input_forcings.netcdf_var_names[force_count] + " (" + str(ve) + ")"
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "unstructured":
                try:
                    input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid GFS variable with ESMF mesh elements: " + input_forcings.netcdf_var_names[force_count] + " (" + str(ve) + ")"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                input_forcings.esmf_field_out.data[np.isnan(input_forcings.esmf_field_out.data)] = -50
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to perform mask calculation on GFS variable: " + input_forcings.netcdf_var_names[force_count] + " (" + str(npe) + ")"
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "unstructured":
                try:
                    input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = config_options.globalNdv
                    input_forcings.esmf_field_out_elem.data[np.isnan(input_forcings.esmf_field_out_elem.data)] = -50
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to perform mask calculation on GFS variable: " + input_forcings.netcdf_var_names[force_count] + " (" + str(npe) + ")"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ESMF field data for GFS: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "unstructured":
                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ESMF field data for GFS: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ESMF field data for GFS: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
                    input_forcings.regridded_forcings1_elem[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "hydrofabric":
                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ESMF field data for GFS: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)

            mask[:] = prev_mask
            if config_options.grid_type == "unstructured":
                mask_elem[:] = prev_mask_elem
    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = f"Unable to close NetCDF file: {input_forcings.tmpFile} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile)
            except FileNotFoundError:
                config_options.statusMsg = f"NetCDF file not found, continuing: {input_forcings.tmpFile}"
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = f"Unable to remove NetCDF file: {input_forcings.tmpFile} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

def regrid_nam_nest(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Rebgrid NAM Nest Data."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if not os.path.isfile(input_forcings.file_in2):
        return

    if input_forcings.regridComplete:
        config_options.statusMsg = "No regridding of NAM nest data necessary for this timestep - already completed."
        err_handler.log_msg(config_options, mpi_config, True)
        return

    file_name = f"NAM_CONUS_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    input_forcings.tmpFile = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    err_handler.check_program_status(config_options, mpi_config)

    id_tmp = None
    try:
        config_options.statusMsg = "Regridding NAM nest data"
        err_handler.log_msg(config_options, mpi_config)
        if input_forcings.file_type != NETCDF:
            if mpi_config.rank == 0:
                if os.path.isfile(input_forcings.tmpFile):
                    config_options.statusMsg = "Found old temporary file: " + input_forcings.tmpFile + " - Removing....."
                    err_handler.log_warning(config_options, mpi_config)
                    try:
                        os_utils.os_remove_retry(input_forcings.tmpFile)
                    except OSError:
                        err_handler.err_out(config_options)
            err_handler.check_program_status(config_options, mpi_config)

            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Converting NAM-Nest Variable: " + grib_var
                    err_handler.log_msg(config_options, mpi_config, True)
                fields.append(":" + grib_var + ":" + input_forcings.grib_levels[force_count] + ":" + str(input_forcings.fcst_hour2) + " hour fcst:")
            fields.append(":(HGT):(surface):")

            if WGRIB2_env:
                cmd = '$WGRIB2 -match "(' + "|".join(fields) + ')" ' + input_forcings.file_in2 + " -netcdf " + input_forcings.tmpFile
            else:
                cmd = "(" + "|".join(fields) + ")"

            id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd, config_options, mpi_config, inputVar=None, special_case=False)
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link("NAM-Nest", input_forcings.file_in2, input_forcings.tmpFile, config_options, mpi_config)
            id_tmp = ioMod.open_netcdf_forcing(input_forcings.tmpFile, config_options, mpi_config)

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Processing NAM Nest Variable: " + grib_var
                err_handler.log_msg(config_options, mpi_config, True)

            calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Calculating NAM nest regridding weights...."
                    err_handler.log_msg(config_options, mpi_config, True)
                calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.grid_type == "gridded":
                    if mpi_config.rank == 0:
                        var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place NetCDF NAM nest elevation data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding NAM nest elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid NAM nest elevation data to the WRF-Hydro domain using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to compute mask on NAM nest elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract ESMF regridded NAM nest elevation data to a local array: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "unstructured":
                    if mpi_config.rank == 0:
                        var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place NetCDF NAM nest elevation data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding NAM nest elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid NAM nest elevation data to the WRF-Hydro domain using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to compute mask on NAM nest elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract ESMF regridded NAM nest elevation data to a local array: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp_elem = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place NetCDF NAM nest elevation data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding NAM nest elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid NAM nest elevation data to the WRF-Hydro domain using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to compute mask on NAM nest elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = input_forcings.esmf_field_out_elem.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract ESMF regridded NAM nest elevation data to a local array: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    if mpi_config.rank == 0:
                        var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place NetCDF NAM nest elevation data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding NAM nest elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid NAM nest elevation data to the WRF-Hydro domain using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to compute mask on NAM nest elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract ESMF regridded NAM nest elevation data to a local array: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                var_tmp = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding NAM nest input variable: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input NAM nest forcing variables using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to calculate mask from input NAM nest regridded forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding NAM nest input variable: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input NAM nest forcing variables using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to calculate mask from input NAM nest regridded forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)

                var_tmp_elem = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding NAM nest input variable: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        var_tmp_elem = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input NAM nest forcing variables using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to calculate mask from input NAM nest regridded forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "hydrofabric":
                var_tmp = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding NAM nest input variable: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input NAM nest forcing variables using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to calculate mask from input NAM nest regridded forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = f"Unable to close NetCDF file: {input_forcings.tmpFile} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile)
            except FileNotFoundError:
                config_options.statusMsg = f"NetCDF file not found, continuing: {input_forcings.tmpFile}"
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = f"Unable to remove NetCDF file: {input_forcings.tmpFile} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)


def regrid_mrms_hourly(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """Rebgrid MRMS hourly precipitation."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if not config_options.use_data_at_current_time:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Exceeded max hours for MRMS precipitation"
            err_handler.log_msg(config_options, mpi_config)
        return

    if not os.path.isfile(supplemental_precip.file_in2):
        if os.path.isfile(supplemental_precip.file_in1):
            supplemental_precip.file_in2 = supplemental_precip.file_in1
        else:
            return

    if supplemental_precip.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No MRMS regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)
        return

    file_uuid = str(mpi_config.uid64)
    mrms_tmp_grib2 = str(Path(config_options.scratch_dir) / f"{file_uuid}_MRMS_PCP_TMP-{mkfilename()}.grib2")
    mrms_tmp_nc = str(Path(config_options.scratch_dir) / f"{file_uuid}_MRMS_PCP_TMP-{mkfilename()}.nc")
    mrms_tmp_rqi_grib2 = str(Path(config_options.scratch_dir) / f"{file_uuid}_MRMS_RQI_TMP-{mkfilename()}.grib2")
    mrms_tmp_rqi_nc = str(Path(config_options.scratch_dir) / f"{file_uuid}_MRMS_RQI_TMP-{mkfilename()}.nc")

    if not supplemental_precip.file_in1 or not supplemental_precip.file_in2:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No MRMS Precipitation available. Supplemental precipitation will not be used."
            err_handler.log_msg(config_options, mpi_config)
        supplemental_precip.regridded_precip2 = None
        supplemental_precip.regridded_precip1 = None
        if config_options.grid_type == "unstructured":
            supplemental_precip.regridded_precip2_elem = None
            supplemental_precip.regridded_precip1_elem = None
        return

    err_handler.check_program_status(config_options, mpi_config)

    id_mrms = None
    id_mrms_rqi = None
    try:
        config_options.statusMsg = "Rrgrid MRMS"
        err_handler.log_msg(config_options, mpi_config)

        if supplemental_precip.file_type != NETCDF:
            ioMod.unzip_file(supplemental_precip.file_in2, mrms_tmp_grib2, config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.rqiMethod == 1:
                ioMod.unzip_file(supplemental_precip.rqi_file_in2, mrms_tmp_rqi_grib2, config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if WGRIB2_env:
                cmd1 = "$WGRIB2 " + mrms_tmp_grib2 + " -netcdf " + mrms_tmp_nc
            else:
                cmd1 = mrms_tmp_grib2

            id_mrms = ioMod.open_grib2(mrms_tmp_grib2, mrms_tmp_nc, cmd1, config_options, mpi_config, supplemental_precip.netcdf_var_names[0], special_case=True)
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.rqiMethod == 1:
                if WGRIB2_env:
                    cmd2 = "$WGRIB2 " + mrms_tmp_rqi_grib2 + " -netcdf " + mrms_tmp_rqi_nc
                else:
                    cmd2 = mrms_tmp_rqi_grib2

                id_mrms_rqi = ioMod.open_grib2(mrms_tmp_rqi_grib2, mrms_tmp_rqi_nc, cmd2, config_options, mpi_config, supplemental_precip.rqi_netcdf_var_names[0], special_case=True)
                err_handler.check_program_status(config_options, mpi_config)
            else:
                id_mrms_rqi = None

        else:
            create_link("MRMS", supplemental_precip.file_in2, mrms_tmp_nc, config_options, mpi_config)
            id_mrms = ioMod.open_netcdf_forcing(mrms_tmp_nc, config_options, mpi_config)
            if supplemental_precip.rqiMethod == 1:
                create_link("RQI", supplemental_precip.rqi_file_in2, mrms_tmp_rqi_nc, config_options, mpi_config)
                id_mrms_rqi = ioMod.open_netcdf_forcing(mrms_tmp_rqi_nc, config_options, mpi_config)
            else:
                id_mrms_rqi = None

        calc_regrid_flag = check_supp_pcp_regrid_status(id_mrms, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Calculating MRMS regridding weights."
                err_handler.log_msg(config_options, mpi_config, True)
            calculate_supp_pcp_weights(supplemental_precip, id_mrms, mrms_tmp_nc, config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        if supplemental_precip.rqiMethod == 1:
            if config_options.grid_type != "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp = id_mrms_rqi.variables[supplemental_precip.rqi_netcdf_var_names[0]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + supplemental_precip.rqi_netcdf_var_names[0] + " from: " + mrms_tmp_rqi_grib2 + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place MRMS data into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding MRMS RQI Field."
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid MRMS RQI field: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    n_masked = len((supplemental_precip.regridded_mask == 0))
                    if n_masked > 0:
                        if mpi_config.rank == 0:
                            config_options.statusMsg = f"{n_masked} masked cells in RQI field, will remove"
                            err_handler.log_msg(config_options, mpi_config, True)

                    supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to run mask calculation for MRMS RQI data: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp = id_mrms_rqi.variables[supplemental_precip.rqi_netcdf_var_names[0]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + supplemental_precip.rqi_netcdf_var_names[0] + " from: " + mrms_tmp_rqi_grib2 + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place MRMS data into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding MRMS RQI Field."
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid MRMS RQI field: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    n_masked = len((supplemental_precip.regridded_mask == 0))
                    if n_masked > 0:
                        if mpi_config.rank == 0:
                            config_options.statusMsg = f"{n_masked} masked cells in RQI field, will remove"
                            err_handler.log_msg(config_options, mpi_config, True)

                    supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to run mask calculation for MRMS RQI data: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_tmp_elem = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp_elem = id_mrms_rqi.variables[supplemental_precip.rqi_netcdf_var_names[0]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract: " + supplemental_precip.rqi_netcdf_var_names[0] + " from: " + mrms_tmp_rqi_grib2 + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(supplemental_precip, var_tmp_elem, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place MRMS data into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding MRMS RQI Field."
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    supplemental_precip.esmf_field_out_elem = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj_elem, supplemental_precip.esmf_field_in_elem, supplemental_precip.esmf_field_out_elem)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid MRMS RQI field: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    n_masked = len((supplemental_precip.regridded_mask_elem == 0))
                    if n_masked > 0:
                        if mpi_config.rank == 0:
                            config_options.statusMsg = f"{n_masked} masked cells in RQI field, will remove"
                            err_handler.log_msg(config_options, mpi_config, True)

                    supplemental_precip.esmf_field_out_elem.data[np.where(supplemental_precip.regridded_mask_elem == 0)] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to run mask calculation for MRMS RQI data: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

        if not supplemental_precip.rqiMethod:
            if config_options.grid_type == "gridded":
                supplemental_precip.regridded_rqi2[:, :] = 1.0
            elif config_options.grid_type == "unstructured":
                supplemental_precip.regridded_rqi2[:] = 1.0
                supplemental_precip.regridded_rqi2_elem[:] = 1.0
            elif config_options.grid_type == "hydrofabric":
                supplemental_precip.regridded_rqi2[:] = 1.0

            if mpi_config.rank == 0:
                config_options.statusMsg = "MRMS Will not be filtered using RQI values."
                err_handler.log_msg(config_options, mpi_config, True)

        elif supplemental_precip.rqiMethod == 2:
            ioMod.read_rqi_monthly_climo(config_options, mpi_config, supplemental_precip, wrf_hydro_geo_meta)
        elif supplemental_precip.rqiMethod == 1:
            if config_options.grid_type == "gridded":
                supplemental_precip.regridded_rqi2[:, :] = supplemental_precip.esmf_field_out.data
            elif config_options.grid_type == "unstructured":
                supplemental_precip.regridded_rqi2[:] = supplemental_precip.esmf_field_out.data
                supplemental_precip.regridded_rqi2_elem[:] = supplemental_precip.esmf_field_out_elem.data
            elif config_options.grid_type == "hydrofabric":
                supplemental_precip.regridded_rqi2[:] = supplemental_precip.esmf_field_out.data

        err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding: " + supplemental_precip.netcdf_var_names[0]
                err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp = id_mrms.variables[supplemental_precip.netcdf_var_names[0]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract: " + supplemental_precip.netcdf_var_names[0] + " from: " + mrms_tmp_nc + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid MRMS precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                if len(np.argwhere(supplemental_precip.esmf_field_out.data < 0)) > 0:
                    supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.esmf_field_out.data < 0)] = config_options.globalNdv

                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv

            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on MRMS supplemental precip: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:, :] = supplemental_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.rqiMethod > 0:
                try:
                    ind_filter = np.where(supplemental_precip.regridded_rqi2 < supplemental_precip.rqiThresh)
                    if len(ind_filter) > 0:
                        if mpi_config.rank == 0:
                            config_options.statusMsg = f"Removing {len(ind_filter)} MRMS cells below RQI threshold of {supplemental_precip.rqiThresh}"
                            err_handler.log_msg(config_options, mpi_config, True)
                    supplemental_precip.regridded_precip2[ind_filter] = config_options.globalNdv
                    del ind_filter
                except (ValueError, AttributeError, KeyError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to run MRMS RQI threshold search: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.keyValue != 14:
                try:
                    ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
                    supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                    del ind_valid
                except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
                    config_options.errMsg = "Unable to run global NDV search on MRMS regridded precip: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:, :] = supplemental_precip.regridded_precip2[:, :]
                supplemental_precip.regridded_rqi1[:, :] = supplemental_precip.regridded_rqi2[:, :]

        elif config_options.grid_type == "unstructured":
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding: " + supplemental_precip.netcdf_var_names[0]
                err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp = id_mrms.variables[supplemental_precip.netcdf_var_names[0]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract: " + supplemental_precip.netcdf_var_names[0] + " from: " + mrms_tmp_nc + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid MRMS precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                if len(np.argwhere(supplemental_precip.esmf_field_out.data < 0)) > 0:
                    supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.esmf_field_out.data < 0)] = config_options.globalNdv

                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv

            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on MRMS supplemental precip: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = supplemental_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.rqiMethod > 0:
                try:
                    ind_filter = np.where(supplemental_precip.regridded_rqi2 < supplemental_precip.rqiThresh)
                    if len(ind_filter) > 0:
                        if mpi_config.rank == 0:
                            config_options.statusMsg = f"Removing {len(ind_filter)} MRMS cells below RQI threshold of {supplemental_precip.rqiThresh}"
                            err_handler.log_msg(config_options, mpi_config, True)
                    supplemental_precip.regridded_precip2[ind_filter] = config_options.globalNdv
                    del ind_filter
                except (ValueError, AttributeError, KeyError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to run MRMS RQI threshold search: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.keyValue != 14:
                try:
                    ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
                    supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                    del ind_valid
                except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
                    config_options.errMsg = "Unable to run global NDV search on MRMS regridded precip: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = supplemental_precip.regridded_precip2[:]
                supplemental_precip.regridded_rqi1[:] = supplemental_precip.regridded_rqi2[:]

            var_tmp_elem = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding: " + supplemental_precip.netcdf_var_names[0]
                err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp_elem = id_mrms.variables[supplemental_precip.netcdf_var_names[0]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract: " + supplemental_precip.netcdf_var_names[0] + " from: " + mrms_tmp_nc + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(supplemental_precip, var_tmp_elem, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj_elem, supplemental_precip.esmf_field_in_elem, supplemental_precip.esmf_field_out_elem)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid MRMS precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem.data[np.where(supplemental_precip.regridded_mask_elem == 0)] = config_options.globalNdv

                if len(np.argwhere(supplemental_precip.esmf_field_out_elem.data < 0)) > 0:
                    supplemental_precip.esmf_field_out_elem.data[np.where(supplemental_precip.esmf_field_out_elem.data < 0)] = config_options.globalNdv

            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on MRMS supplemental precip: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2_elem[:] = supplemental_precip.esmf_field_out_elem.data
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.rqiMethod > 0:
                try:
                    ind_filter_elem = np.where(supplemental_precip.regridded_rqi2_elem < supplemental_precip.rqiThresh)
                    if len(ind_filter_elem) > 0:
                        if mpi_config.rank == 0:
                            config_options.statusMsg = f"Removing {len(ind_filter)} MRMS cells below RQI threshold of {supplemental_precip.rqiThresh}"
                            err_handler.log_msg(config_options, mpi_config, True)
                    supplemental_precip.regridded_precip2_elem[ind_filter_elem] = config_options.globalNdv
                    del ind_filter_elem
                except (ValueError, AttributeError, KeyError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to run MRMS RQI threshold search: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid_elem = np.where(supplemental_precip.regridded_precip2_elem != config_options.globalNdv)
                supplemental_precip.regridded_precip2_elem[ind_valid_elem] = supplemental_precip.regridded_precip2_elem[ind_valid_elem] / 3600.0
                del ind_valid_elem
            except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
                config_options.errMsg = "Unable to run global NDV search on MRMS regridded precip: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1_elem[:] = supplemental_precip.regridded_precip2_elem[:]
                supplemental_precip.regridded_rqi1_elem[:] = supplemental_precip.regridded_rqi2_elem[:]

        elif config_options.grid_type == "hydrofabric":
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding: " + supplemental_precip.netcdf_var_names[0]
                err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp = id_mrms.variables[supplemental_precip.netcdf_var_names[0]][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract: " + supplemental_precip.netcdf_var_names[0] + " from: " + mrms_tmp_nc + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid MRMS precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                if len(np.argwhere(supplemental_precip.esmf_field_out.data < 0)) > 0:
                    supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.esmf_field_out.data < 0)] = config_options.globalNdv

                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv

            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on MRMS supplemental precip: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = supplemental_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.rqiMethod > 0:
                try:
                    ind_filter = np.where(supplemental_precip.regridded_rqi2 < supplemental_precip.rqiThresh)
                    if len(ind_filter) > 0:
                        if mpi_config.rank == 0:
                            config_options.statusMsg = f"Removing {len(ind_filter)} MRMS cells below RQI threshold of {supplemental_precip.rqiThresh}"
                            err_handler.log_msg(config_options, mpi_config, True)
                    supplemental_precip.regridded_precip2[ind_filter] = config_options.globalNdv
                    del ind_filter
                except (ValueError, AttributeError, KeyError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to run MRMS RQI threshold search: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.keyValue != 14:
                try:
                    ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
                    supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                    del ind_valid
                except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
                    config_options.errMsg = "Unable to run global NDV search on MRMS regridded precip: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = supplemental_precip.regridded_precip2[:]
                supplemental_precip.regridded_rqi1[:] = supplemental_precip.regridded_rqi2[:]

    finally:
        if id_mrms is not None:
            try:
                id_mrms.close()
            except OSError:
                config_options.errMsg = f"Unable to close NetCDF file: {mrms_tmp_nc}"
                err_handler.log_critical(config_options, mpi_config)

        if id_mrms_rqi is not None:
            try:
                id_mrms_rqi.close()
            except OSError:
                config_options.errMsg = f"Unable to close NetCDF file: {mrms_tmp_rqi_nc}"
                err_handler.log_critical(config_options, mpi_config)

        if mpi_config.rank == 0:
            for f in (mrms_tmp_grib2, mrms_tmp_nc, mrms_tmp_rqi_grib2, mrms_tmp_rqi_nc):
                if os.path.isfile(f):
                    try:
                        os_utils.os_remove_retry(f)
                    except OSError:
                        config_options.errMsg = f"Unable to remove scratch file: {f}"
                        err_handler.log_critical(config_options, mpi_config)
        mpi_config.comm.barrier()
        err_handler.check_program_status(config_options, mpi_config)

def regrid_mrms_subhourly(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """Rebgrid MRMS hourly precipitation."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if not config_options.use_data_at_current_time:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Exceeded max hours for MRMS precipitation"
            err_handler.log_msg(config_options, mpi_config)
        return

    if supplemental_precip.product_name == "MRMS_Subhourly_Cycling":
        if not os.path.isfile(supplemental_precip.file_in1_bc) and not os.path.isfile(supplemental_precip.file_in2_bc):
            if mpi_config.rank == 0:
                config_options.statusMsg = "No MRMS hourly Precipitation available. Supplemental precipitation will not be used."
                err_handler.log_msg(config_options, mpi_config)
            return
        elif any(not f.is_file() for f in supplemental_precip.file_in1) or any(not f.is_file() for f in supplemental_precip.file_in2):
            if mpi_config.rank == 0:
                config_options.statusMsg = "No MRMS sub-hourly Precipitation available. Supplemental precipitation will not be used."
                err_handler.log_msg(config_options, mpi_config)
            return
    else:
        if not os.path.isfile(supplemental_precip.file_in1_bc):
            return

    if supplemental_precip.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No MRMS sub-hourly regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)
        return

    if not supplemental_precip.file_in1_bc:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No MRMS Precipitation available. Supplemental precipitation will not be used."
            err_handler.log_msg(config_options, mpi_config)
        supplemental_precip.regridded_precip2 = None
        supplemental_precip.regridded_precip1 = None
        if config_options.grid_type == "unstructured":
            supplemental_precip.regridded_precip2_elem = None
            supplemental_precip.regridded_precip1_elem = None
        return

    err_handler.check_program_status(config_options, mpi_config)

    file_uuid = str(mpi_config.uid64)
    mrms_tmp_grib2 = str(Path(config_options.scratch_dir) / f"{file_uuid}_MRMS_PCP_TMP-{mkfilename()}.grib2")
    mrms_tmp_nc = str(Path(config_options.scratch_dir) / f"{file_uuid}_MRMS_PCP_TMP-{mkfilename()}.nc")

    id_mrms = None
    try:
        config_options.statusMsg = "Regrid MRMS Sub-hourly"
        err_handler.log_msg(config_options, mpi_config)

        ioMod.unzip_file(supplemental_precip.file_in1_bc, mrms_tmp_grib2, config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        cmd1 = f"$WGRIB2 {mrms_tmp_grib2} -netcdf {mrms_tmp_nc}" if WGRIB2_env else mrms_tmp_grib2

        id_mrms = ioMod.open_grib2(mrms_tmp_grib2, mrms_tmp_nc, cmd1, config_options, mpi_config, supplemental_precip.netcdf_var_names[0], special_case=True)
        err_handler.check_program_status(config_options, mpi_config)

        calc_regrid_flag = check_supp_pcp_regrid_status(id_mrms, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Calculating MRMS sub-hourly regridding weights."
                err_handler.log_msg(config_options, mpi_config, True)
            calculate_supp_pcp_weights(supplemental_precip, id_mrms, mrms_tmp_nc, config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            supplemental_precip.regridded_rqi2[:, :] = 1.0
        elif config_options.grid_type in ("unstructured", "hydrofabric"):
            supplemental_precip.regridded_rqi2[:] = 1.0
            if config_options.grid_type == "unstructured":
                supplemental_precip.regridded_rqi2_elem[:] = 1.0

        if mpi_config.rank == 0:
            config_options.statusMsg = "MRMS sub-hourly will not be filtered using RQI values."
            err_handler.log_msg(config_options, mpi_config, True)

        err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding: " + supplemental_precip.netcdf_var_names[0]
                err_handler.log_msg(config_options, mpi_config, True)
                try:
                    if supplemental_precip.product_name == "MRMS_Subhourly_Cycling":
                        var_tmp = get_mrms_subhourly_avg(config_options, supplemental_precip, mpi_config)
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to perform MRMS sub-hourly bias correction and aggregation"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid MRMS sub-hourly precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                if len(np.argwhere(supplemental_precip.esmf_field_out.data < 0)) > 0:
                    supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.esmf_field_out.data < 0)] = config_options.globalNdv

                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv

            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on MRMS sub-hourly supplemental precip: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:, :] = supplemental_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.keyValue != 14:
                try:
                    ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
                    supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                    del ind_valid
                except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
                    config_options.errMsg = "Unable to run global NDV search on MRMS sub-hourly regridded precip: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:, :] = supplemental_precip.regridded_precip2[:, :]
                supplemental_precip.regridded_rqi1[:, :] = supplemental_precip.regridded_rqi2[:, :]

        elif config_options.grid_type == "unstructured":
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding: " + supplemental_precip.netcdf_var_names[0]
                err_handler.log_msg(config_options, mpi_config, True)
                try:
                    if supplemental_precip.product_name == "MRMS_Subhourly_Cycling":
                        var_tmp = get_mrms_subhourly_avg(config_options, supplemental_precip, mpi_config)
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to perform MRMS sub-hourly bias correction and aggregation"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid MRMS sub-hourly precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                if len(np.argwhere(supplemental_precip.esmf_field_out.data < 0)) > 0:
                    supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.esmf_field_out.data < 0)] = config_options.globalNdv

                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv

            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on MRMS sub-hourly supplemental precip: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = supplemental_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.keyValue != 14:
                try:
                    ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
                    supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                    del ind_valid
                except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
                    config_options.errMsg = "Unable to run global NDV search on MRMS sub-hourly regridded precip: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = supplemental_precip.regridded_precip2[:]
                supplemental_precip.regridded_rqi1[:] = supplemental_precip.regridded_rqi2[:]

            var_tmp_elem = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding: " + supplemental_precip.netcdf_var_names[0]
                err_handler.log_msg(config_options, mpi_config, True)
                try:
                    if supplemental_precip.product_name == "MRMS_Subhourly_Cycling":
                        var_tmp_elem = get_mrms_subhourly_avg(config_options, supplemental_precip, mpi_config)
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to perform MRMS sub-hourly bias correction and aggregation"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(supplemental_precip, var_tmp_elem, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj_elem, supplemental_precip.esmf_field_in_elem, supplemental_precip.esmf_field_out_elem)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid MRMS sub-hourly precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem.data[np.where(supplemental_precip.regridded_mask_elem == 0)] = config_options.globalNdv

                if len(np.argwhere(supplemental_precip.esmf_field_out_elem.data < 0)) > 0:
                    supplemental_precip.esmf_field_out_elem.data[np.where(supplemental_precip.esmf_field_out_elem.data < 0)] = config_options.globalNdv

            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on MRMS sub-hourly supplemental precip: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2_elem[:] = supplemental_precip.esmf_field_out_elem.data
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.keyValue != 14:
                try:
                    ind_valid_elem = np.where(supplemental_precip.regridded_precip2_elem != config_options.globalNdv)
                    supplemental_precip.regridded_precip2_elem[ind_valid_elem] = supplemental_precip.regridded_precip2_elem[ind_valid_elem] / 3600.0
                    del ind_valid_elem
                except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
                    config_options.errMsg = "Unable to run global NDV search on MRMS sub-hourly regridded precip: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1_elem[:] = supplemental_precip.regridded_precip2_elem[:]
                supplemental_precip.regridded_rqi1_elem[:] = supplemental_precip.regridded_rqi2_elem[:]

        elif config_options.grid_type == "hydrofabric":
            var_tmp = None
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding: " + supplemental_precip.netcdf_var_names[0]
                err_handler.log_msg(config_options, mpi_config, True)
                try:
                    if supplemental_precip.product_name == "MRMS_Subhourly_Cycling":
                        var_tmp = get_mrms_subhourly_avg(config_options, supplemental_precip, mpi_config)
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to perform MRMS sub-hourly bias correction and aggregation"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid MRMS sub-hourly precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                if len(np.argwhere(supplemental_precip.esmf_field_out.data < 0)) > 0:
                    supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.esmf_field_out.data < 0)] = config_options.globalNdv

                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv

            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on MRMS sub-hourly supplemental precip: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = supplemental_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.keyValue != 14:
                try:
                    ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
                    supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                    del ind_valid
                except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
                    config_options.errMsg = "Unable to run global NDV search on MRMS sub-hourly regridded precip: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = supplemental_precip.regridded_precip2[:]
                supplemental_precip.regridded_rqi1[:] = supplemental_precip.regridded_rqi2[:]

    finally:
        if id_mrms is not None:
            try:
                id_mrms.close()
            except OSError:
                config_options.errMsg = f"Unable to close NetCDF file: {mrms_tmp_nc}"
                err_handler.log_critical(config_options, mpi_config)

        if mpi_config.rank == 0:
            for f in (mrms_tmp_grib2, mrms_tmp_nc):
                if os.path.isfile(f):
                    try:
                        os_utils.os_remove_retry(f)
                    except OSError:
                        config_options.errMsg = f"Unable to remove scratch file: {f}"
                        err_handler.log_critical(config_options, mpi_config)
        mpi_config.comm.barrier()
        err_handler.check_program_status(config_options, mpi_config)


def regrid_mrms_precip_flag(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """Rebgrid MRMS hourly precipitation flag."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if not os.path.exists(supplemental_precip.file_in2):
        return

    if supplemental_precip.regridComplete:
        return

    file_name = f"_MRMS_PCP_FLAG_TMP_{mkfilename()}"
    file_uuid = str(mpi_config.uid64)
    mrms_tmp_grib2 = str(Path(config_options.scratch_dir) / f"{file_uuid}{file_name}.grib2")
    mrms_tmp_nc = str(Path(config_options.scratch_dir) / f"{file_uuid}{file_name}.nc")

    ioMod.unzip_file(supplemental_precip.file_in2, mrms_tmp_grib2, config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    cmd1 = f"$WGRIB2 {mrms_tmp_grib2} -netcdf {mrms_tmp_nc}"
    id_tmp = ioMod.open_grib2(mrms_tmp_grib2, mrms_tmp_nc, cmd1, config_options, mpi_config, supplemental_precip.netcdf_var_names[0])
    err_handler.check_program_status(config_options, mpi_config)

    calc_regrid_flag = check_supp_pcp_regrid_status(id_tmp, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if calc_regrid_flag:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Calculating MRMS PrecipFlag regridding weights."
            err_handler.log_msg(config_options, mpi_config)
        calculate_supp_pcp_weights(supplemental_precip, id_tmp, supplemental_precip.file_in2, config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    var_tmp = None
    if mpi_config.rank == 0:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding MRMS PrecipFlag Fraction."
            err_handler.log_msg(config_options, mpi_config)
        try:
            var_tmp = id_tmp.variables[supplemental_precip.netcdf_var_names[0]][0, :, :]
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to extract PrecipFlag from file: " + supplemental_precip.file_in2 + " (" + str(err) + ")"
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    mask = supplemental_precip.esmf_grid_in.get_item(ESMF.GridItem.MASK)
    prev_mask = np.copy(mask)
    mask[np.where(var_sub_tmp < 0)] = 0

    try:
        var_sub_tmp[var_sub_tmp <= 0] = np.nan
        var_sub_tmp[var_sub_tmp == 3] = 0.0
        var_sub_tmp[var_sub_tmp == 7] = 0.0
        var_sub_tmp[var_sub_tmp > 0] = 1.0

        supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
    except (ValueError, KeyError, AttributeError) as err:
        config_options.errMsg = "Unable to place MRMS PrecipFlag into local ESMF field: " + str(err)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
    except ValueError as ve:
        config_options.errMsg = "Unable to regrid MRMS PrecipFlag: " + str(ve)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv
        supplemental_precip.esmf_field_out.data[np.isnan(supplemental_precip.esmf_field_out.data)] = config_options.globalNdv
    except (ValueError, ArithmeticError) as npe:
        config_options.errMsg = "Unable to run mask search on MRMS PrecipFlag: " + str(npe)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    supplemental_precip.regridded_precip2[:] = supplemental_precip.esmf_field_out.data
    err_handler.check_program_status(config_options, mpi_config)

    if config_options.current_output_step == 1:
        supplemental_precip.regridded_precip1[:] = supplemental_precip.regridded_precip2[:]
    err_handler.check_program_status(config_options, mpi_config)

    if config_options.grid_type == "unstructured":
        var_tmp_elem = None
        if mpi_config.rank == 0:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding MRMS PrecipFlag Fraction."
                err_handler.log_msg(config_options, mpi_config)
            try:
                var_tmp_elem = id_tmp.variables[supplemental_precip.netcdf_var_names[0]][0, :, :]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract PrecipFlag from file: " + supplemental_precip.file_in2 + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp_elem = mpi_config.scatter_array(supplemental_precip, var_tmp_elem, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        mask_elem = supplemental_precip.esmf_grid_in.get_item(ESMF.GridItem.MASK)
        prev_mask_elem = np.copy(mask_elem)
        mask_elem[np.where(var_sub_tmp_elem < 0)] = 0

        try:
            var_sub_tmp_elem[var_sub_tmp_elem <= 0] = np.nan
            var_sub_tmp_elem[var_sub_tmp_elem == 3] = 0.0
            var_sub_tmp_elem[var_sub_tmp_elem == 7] = 0.0
            var_sub_tmp_elem[var_sub_tmp_elem > 0] = 1.0

            supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place MRMS PrecipFlag into local ESMF field: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_precip.esmf_field_out_elem = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj_elem, supplemental_precip.esmf_field_in_elem, supplemental_precip.esmf_field_out_elem)
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid MRMS PrecipFlag: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_precip.esmf_field_out_elem.data[np.where(supplemental_precip.regridded_mask_elem == 0)] = config_options.globalNdv
            supplemental_precip.esmf_field_out_elem.data[np.isnan(supplemental_precip.esmf_field_out_elem.data)] = config_options.globalNdv
        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to run mask search on MRMS PrecipFlag: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_precip.regridded_precip2_elem[:] = supplemental_precip.esmf_field_out_elem.data
        err_handler.check_program_status(config_options, mpi_config)

        if config_options.current_output_step == 1:
            supplemental_precip.regridded_precip1_elem[:] = supplemental_precip.regridded_precip2_elem[:]
        err_handler.check_program_status(config_options, mpi_config)

        mask_elem[:] = prev_mask_elem

    mask[:] = prev_mask

    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except Exception as e:
            config_options.errMsg = f"Unable to close NetCDF file: {mrms_tmp_nc} - {e}\n{traceback.format_exc()}"
            err_handler.log_critical(config_options, mpi_config)
        try:
            os_utils.os_remove_retry(mrms_tmp_nc)
        except FileNotFoundError:
            config_options.statusMsg = f"NetCDF file not found, continuing: {mrms_tmp_nc}"
            err_handler.log_warning(config_options, mpi_config)
        except Exception as e:
            config_options.errMsg = f"Unable to remove NetCDF file: {mrms_tmp_nc} - {e}\n{traceback.format_exc()}"
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

def regrid_hourly_wrf_arw(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid NAM Nest Data."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if not os.path.isfile(input_forcings.file_in2):
        return

    if input_forcings.regridComplete:
        config_options.statusMsg = "No regridding of WRF-ARW nest data necessary for this timestep - already completed."
        err_handler.log_msg(config_options, mpi_config, True)
        return

    file_name = f"ARW_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    input_forcings.tmpFile = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    err_handler.check_program_status(config_options, mpi_config)

    id_tmp = None
    try:
        config_options.statusMsg = "Regrid WRF-ARW nest data"
        err_handler.log_msg(config_options, mpi_config)

        if input_forcings.file_type != NETCDF:
            if mpi_config.rank == 0:
                if os.path.isfile(input_forcings.tmpFile):
                    config_options.statusMsg = "Found old temporary file: " + input_forcings.tmpFile + " - Removing....."
                    err_handler.log_warning(config_options, mpi_config)
                    try:
                        os_utils.os_remove_retry(input_forcings.tmpFile)
                    except OSError:
                        err_handler.err_out(config_options)
            err_handler.check_program_status(config_options, mpi_config)

            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Converting WRF-ARW Variable: " + grib_var
                    err_handler.log_msg(config_options, mpi_config, True)
                time_str = "{}-{} hour acc fcst".format(input_forcings.fcst_hour1, input_forcings.fcst_hour2) if grib_var == "APCP" else str(input_forcings.fcst_hour2) + " hour fcst"
                fields.append(":" + grib_var + ":" + input_forcings.grib_levels[force_count] + ":" + time_str + ":")
            fields.append(":(HGT):(surface):")

            if WGRIB2_env:
                cmd = '$WGRIB2 -match "(' + "|".join(fields) + ')" ' + input_forcings.file_in2 + " -netcdf " + input_forcings.tmpFile
            else:
                cmd = "(" + "|".join(fields) + ")"

            id_tmp = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd, config_options, mpi_config, inputVar=None, special_case=False)
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link("WRF-ARW", input_forcings.file_in2, input_forcings.tmpFile, config_options, mpi_config)
            id_tmp = ioMod.open_netcdf_forcing(input_forcings.tmpFile, config_options, mpi_config)

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Processing WRF-ARW Variable: " + grib_var
                err_handler.log_msg(config_options, mpi_config, True)

            calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Calculating WRF-ARW regridding weights...."
                    err_handler.log_msg(config_options, mpi_config, True)
                calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.grid_type == "gridded":
                    if mpi_config.rank == 0:
                        var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place NetCDF WRF-ARW elevation data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding WRF-ARW elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid WRF-ARW elevation data to the WRF-Hydro domain using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to compute mask on WRF-ARW elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract ESMF regridded WRF-ARW elevation data to a local array: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)
                elif config_options.grid_type == "unstructured":
                    if mpi_config.rank == 0:
                        var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place NetCDF WRF-ARW elevation data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding WRF-ARW elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid WRF-ARW elevation data to the WRF-Hydro domain using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to compute mask on WRF-ARW elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract ESMF regridded WRF-ARW elevation data to a local array: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp_elem = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place NetCDF WRF-ARW elevation data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding WRF-ARW elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid WRF-ARW elevation data to the WRF-Hydro domain using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to compute mask on WRF-ARW elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = input_forcings.esmf_field_out_elem.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract ESMF regridded WRF-ARW elevation data to a local array: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    if mpi_config.rank == 0:
                        var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place NetCDF WRF-ARW elevation data into the ESMF field object: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        config_options.statusMsg = "Regridding WRF-ARW elevation data to the WRF-Hydro domain."
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid WRF-ARW elevation data to the WRF-Hydro domain using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to compute mask on WRF-ARW elevation data: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract ESMF regridded WRF-ARW elevation data to a local array: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                var_tmp = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding WRF-ARW input variable: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input WRF-ARW forcing variables using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to calculate mask from input WRF-ARW regridded forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if grib_var == "APCP":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != config_options.globalNdv)
                        input_forcings.esmf_field_out.data[ind_valid] = input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        del ind_valid
                    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                        config_options.errMsg = "Unable to run NDV search on WRF ARW precipitation: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding WRF-ARW input variable: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input WRF-ARW forcing variables using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to calculate mask from input WRF-ARW regridded forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if grib_var == "APCP":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != config_options.globalNdv)
                        input_forcings.esmf_field_out.data[ind_valid] = input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        del ind_valid
                    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                        config_options.errMsg = "Unable to run NDV search on WRF ARW precipitation: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)

                var_tmp_elem = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding WRF-ARW input variable: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        var_tmp_elem = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input WRF-ARW forcing variables using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to calculate mask from input WRF-ARW regridded forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if grib_var == "APCP":
                    try:
                        ind_valid_elem = np.where(input_forcings.esmf_field_out_elem.data != config_options.globalNdv)
                        input_forcings.esmf_field_out_elem.data[ind_valid_elem] = input_forcings.esmf_field_out_elem.data[ind_valid_elem] / 3600.0
                        del ind_valid_elem
                    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                        config_options.errMsg = "Unable to run NDV search on WRF ARW precipitation: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "hydrofabric":
                var_tmp = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding WRF-ARW input variable: " + input_forcings.netcdf_var_names[force_count]
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input WRF-ARW forcing variables using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to calculate mask from input WRF-ARW regridded forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if grib_var == "APCP":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != config_options.globalNdv)
                        input_forcings.esmf_field_out.data[ind_valid] = input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        del ind_valid
                    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                        config_options.errMsg = "Unable to run NDV search on WRF ARW precipitation: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = f"Unable to close NetCDF file: {input_forcings.tmpFile} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(input_forcings.tmpFile)
            except FileNotFoundError:
                config_options.statusMsg = f"NetCDF file not found, continuing: {input_forcings.tmpFile}"
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = f"Unable to remove NetCDF file: {input_forcings.tmpFile} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


def regrid_hourly_wrf_arw_hi_res_pcp(supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """Rebgrid hourly WRF-ARW hi-res nest precipitation."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if not os.path.exists(supplemental_precip.file_in1):
        return

    if supplemental_precip.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No ARW regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)
        return

    file_name = f"ARW_PCP_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    arw_tmp_nc = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    id_tmp = None
    try:
        config_options.statusMsg = "Regrid ARW"
        err_handler.log_msg(config_options, mpi_config)

        if supplemental_precip.file_type != NETCDF:
            if mpi_config.rank == 0:
                if os.path.isfile(arw_tmp_nc):
                    config_options.statusMsg = "Found old temporary file: " + arw_tmp_nc + " - Removing....."
                    err_handler.log_warning(config_options, mpi_config)
                    try:
                        os_utils.os_remove_retry(arw_tmp_nc)
                    except IOError:
                        err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if WGRIB2_env:
                cmd = "$WGRIB2 " + supplemental_precip.file_in1 + ' -match ":(APCP):(surface):(' + str(supplemental_precip.fcst_hour1 - 1) + "-" + str(supplemental_precip.fcst_hour1) + ' hour acc fcst):"' + " -netcdf " + arw_tmp_nc
            else:
                cmd = "(APCP):(surface):(" + str(supplemental_precip.fcst_hour1 - 1) + "-" + str(supplemental_precip.fcst_hour1) + " hour acc fcst)"

            id_tmp = ioMod.open_grib2(supplemental_precip.file_in1, arw_tmp_nc, cmd, config_options, mpi_config, "APCP_surface", special_case=False)
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link("ARW-PCP", supplemental_precip.file_in1, arw_tmp_nc, config_options, mpi_config)
            id_tmp = ioMod.open_netcdf_forcing(arw_tmp_nc, config_options, mpi_config)

        calc_regrid_flag = check_supp_pcp_regrid_status(id_tmp, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Calculating WRF ARW regridding weights."
                err_handler.log_msg(config_options, mpi_config, True)
            calculate_supp_pcp_weights(supplemental_precip, id_tmp, arw_tmp_nc, config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding WRF ARW APCP Precipitation."
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp = id_tmp.variables["APCP_surface"][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract precipitation from WRF ARW file: " + supplemental_precip.file_in1 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place WRF ARW precipitation into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid WRF ARW supplemental precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on WRF ARW supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:, :] = supplemental_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
                supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = "Unable to run NDV search on WRF ARW supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:, :] = supplemental_precip.regridded_precip2[:, :]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding WRF ARW APCP Precipitation."
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp = id_tmp.variables["APCP_surface"][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract precipitation from WRF ARW file: " + supplemental_precip.file_in1 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place WRF ARW precipitation into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid WRF ARW supplemental precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on WRF ARW supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = supplemental_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
                supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = "Unable to run NDV search on WRF ARW supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = supplemental_precip.regridded_precip2[:]
            err_handler.check_program_status(config_options, mpi_config)

            var_tmp_elem = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding WRF ARW APCP Precipitation."
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp_elem = id_tmp.variables["APCP_surface"][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract precipitation from WRF ARW file: " + supplemental_precip.file_in1 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(supplemental_precip, var_tmp_elem, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place WRF ARW precipitation into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj_elem, supplemental_precip.esmf_field_in_elem, supplemental_precip.esmf_field_out_elem)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid WRF ARW supplemental precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem.data[np.where(supplemental_precip.regridded_mask_elem == 0)] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on WRF ARW supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2_elem[:] = supplemental_precip.esmf_field_out_elem.data
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid_elem = np.where(supplemental_precip.regridded_precip2_elem != config_options.globalNdv)
                supplemental_precip.regridded_precip2_elem[ind_valid_elem] = supplemental_precip.regridded_precip2_elem[ind_valid_elem] / 3600.0
                del ind_valid_elem
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = "Unable to run NDV search on WRF ARW supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1_elem[:] = supplemental_precip.regridded_precip2_elem[:]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding WRF ARW APCP Precipitation."
                    err_handler.log_msg(config_options, mpi_config, True)
                try:
                    var_tmp = id_tmp.variables["APCP_surface"][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract precipitation from WRF ARW file: " + supplemental_precip.file_in1 + " (" + str(err) + ")"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place WRF ARW precipitation into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
            except ValueError as ve:
                config_options.errMsg = "Unable to regrid WRF ARW supplemental precipitation: " + str(ve)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out.data[np.where(supplemental_precip.regridded_mask == 0)] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = "Unable to run mask search on WRF ARW supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = supplemental_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            try:
                ind_valid = np.where(supplemental_precip.regridded_precip2 != config_options.globalNdv)
                supplemental_precip.regridded_precip2[ind_valid] = supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = "Unable to run NDV search on WRF ARW supplemental precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = supplemental_precip.regridded_precip2[:]
            err_handler.check_program_status(config_options, mpi_config)

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = f"Unable to close NetCDF file: {arw_tmp_nc} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(arw_tmp_nc)
            except FileNotFoundError:
                config_options.statusMsg = f"NetCDF file not found, continuing: {arw_tmp_nc}"
                err_handler.log_warning(config_options, mpi_config)
            except Exception as e:
                config_options.errMsg = f"Unable to remove NetCDF file: {arw_tmp_nc}: {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)


def regrid_sbcv2_liquid_water_fraction(supplemental_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Rebgrid SBCv2 Liquid Water Fraction."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if not os.path.exists(supplemental_forcings.file_in1):
        return

    if supplemental_forcings.regridComplete:
        return

    id_tmp = ioMod.open_netcdf_forcing(supplemental_forcings.file_in1, config_options, mpi_config)

    calc_regrid_flag = check_supp_pcp_regrid_status(id_tmp, supplemental_forcings, config_options, wrf_hydro_geo_meta, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if calc_regrid_flag:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Calculating SBCv2 Liquid Water Fraction regridding weights."
            err_handler.log_msg(config_options, mpi_config, True)
        calculate_supp_pcp_weights(supplemental_forcings, id_tmp, supplemental_forcings.file_in1, config_options, mpi_config, lat_var="Lat", lon_var="Lon")
        err_handler.check_program_status(config_options, mpi_config)

    if config_options.grid_type == "gridded":
        var_tmp = None
        if mpi_config.rank == 0:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding SBCv2 Liquid Water Fraction."
                err_handler.log_msg(config_options, mpi_config)
            try:
                var_tmp = id_tmp.variables[supplemental_forcings.netcdf_var_names[0]][:]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract Liquid Water Fraction from SBCv2 file: " + supplemental_forcings.file_in1 + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(supplemental_forcings, var_tmp, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_in.data[:, :] = var_sub_tmp / 100.0
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place SBCv2 Liquid Water Fraction into local ESMF field: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_forcings.regridObj, supplemental_forcings.esmf_field_in, supplemental_forcings.esmf_field_out)
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid SBCv2 Liquid Water Fraction: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_out.data[np.where(supplemental_forcings.regridded_mask == 0)] = 1.0
            supplemental_forcings.esmf_field_out.data[np.where(supplemental_forcings.esmf_field_out.data < 0)] = 1.0
        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to run mask search on SBCv2 Liquid Water Fraction: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_forcings.regridded_precip2[:] = supplemental_forcings.esmf_field_out.data
        err_handler.check_program_status(config_options, mpi_config)

        if config_options.current_output_step == 1:
            supplemental_forcings.regridded_precip1[:] = supplemental_forcings.regridded_precip2[:]
        err_handler.check_program_status(config_options, mpi_config)

    elif config_options.grid_type == "unstructured":
        var_tmp = None
        if mpi_config.rank == 0:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding SBCv2 Liquid Water Fraction - unstructured"
                err_handler.log_msg(config_options, mpi_config, True)
            try:
                var_tmp = id_tmp.variables[supplemental_forcings.netcdf_var_names[0]][:]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract Liquid Water Fraction from SBCv2 file: " + supplemental_forcings.file_in1 + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(supplemental_forcings, var_tmp, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_in.data[:, :] = var_sub_tmp / 100.0
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place SBCv2 Liquid Water Fraction into local ESMF field: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_forcings.regridObj, supplemental_forcings.esmf_field_in, supplemental_forcings.esmf_field_out)
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid SBCv2 Liquid Water Fraction: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_out.data[np.where(supplemental_forcings.regridded_mask == 0)] = 1.0
            supplemental_forcings.esmf_field_out.data[np.where(supplemental_forcings.esmf_field_out.data < 0)] = 1.0
        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to run mask search on SBCv2 Liquid Water Fraction: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_forcings.regridded_precip2[:] = supplemental_forcings.esmf_field_out.data
        err_handler.check_program_status(config_options, mpi_config)

        if config_options.current_output_step == 1:
            supplemental_forcings.regridded_precip1[:] = supplemental_forcings.regridded_precip2[:]
        err_handler.check_program_status(config_options, mpi_config)

        var_tmp_elem = None
        if mpi_config.rank == 0:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding SBCv2 Liquid Water Fraction input variable."
                err_handler.log_msg(config_options, mpi_config, True)
            try:
                var_tmp_elem = id_tmp.variables[supplemental_forcings.netcdf_var_names[0]][:]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract Liquid Water Fraction from SBCv2 file: " + supplemental_forcings.file_in1 + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp_elem = mpi_config.scatter_array(supplemental_forcings, var_tmp_elem, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem / 100.0
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place SBCv2 Liquid Water Fraction into local ESMF field: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(supplemental_forcings.regridObj_elem, supplemental_forcings.esmf_field_in_elem, supplemental_forcings.esmf_field_out_elem)
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid SBCv2 Liquid Water Fraction: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_out_elem.data[np.where(supplemental_forcings.regridded_mask_elem == 0)] = 1.0
            supplemental_forcings.esmf_field_out_elem.data[np.where(supplemental_forcings.esmf_field_out_elem.data < 0)] = 1.0
        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to run mask search on SBCv2 Liquid Water Fraction: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_forcings.regridded_precip2_elem[:] = supplemental_forcings.esmf_field_out_elem.data
        err_handler.check_program_status(config_options, mpi_config)

        if config_options.current_output_step == 1:
            supplemental_forcings.regridded_precip1_elem[:] = supplemental_forcings.regridded_precip2_elem[:]
        err_handler.check_program_status(config_options, mpi_config)

    elif config_options.grid_type == "hydrofabric":
        var_tmp = None
        if mpi_config.rank == 0:
            if mpi_config.rank == 0:
                config_options.statusMsg = "Regridding SBCv2 Liquid Water Fraction - hydrofabric"
                err_handler.log_msg(config_options, mpi_config)
            try:
                var_tmp = id_tmp.variables[supplemental_forcings.netcdf_var_names[0]][:]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract Liquid Water Fraction from SBCv2 file: " + supplemental_forcings.file_in1 + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(supplemental_forcings, var_tmp, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_in.data[:, :] = var_sub_tmp / 100.0
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place SBCv2 Liquid Water Fraction into local ESMF field: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_forcings.regridObj, supplemental_forcings.esmf_field_in, supplemental_forcings.esmf_field_out)
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid SBCv2 Liquid Water Fraction: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_out.data[np.where(supplemental_forcings.regridded_mask == 0)] = 1.0
            supplemental_forcings.esmf_field_out.data[np.where(supplemental_forcings.esmf_field_out.data < 0)] = 1.0
        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to run mask search on SBCv2 Liquid Water Fraction: " + str(npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_forcings.regridded_precip2[:] = supplemental_forcings.esmf_field_out.data
        err_handler.check_program_status(config_options, mpi_config)

        if config_options.current_output_step == 1:
            supplemental_forcings.regridded_precip1[:] = supplemental_forcings.regridded_precip2[:]
        err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + supplemental_forcings.file_in1
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

@static_vars(last_file=None)
def regrid_hourly_nbm(forcings_or_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid hourly NBM precipitation."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if not config_options.use_data_at_current_time:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Exceeded max hours for NBM data, will not use NBM in final layering."
            err_handler.log_msg(config_options, mpi_config)
        return

    if not os.path.exists(forcings_or_precip.file_in1) or forcings_or_precip.regridComplete:
        return

    file_name = f"NBM_PCP_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    nbm_tmp_nc = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    if mpi_config.rank == 0 and os.path.isfile(nbm_tmp_nc):
        config_options.statusMsg = "Found old temporary file: " + nbm_tmp_nc + " - Removing....."
        err_handler.log_warning(config_options, mpi_config)
        try:
            os_utils.os_remove_retry(nbm_tmp_nc)
        except OSError:
            config_options.errMsg = "Unable to remove file: " + nbm_tmp_nc
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    id_tmp = None
    try:
        if forcings_or_precip.grib_vars is not None:
            fields = []
            for force_count, grib_var in enumerate(forcings_or_precip.grib_vars):
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Converting NBM Variable: " + grib_var
                    err_handler.log_msg(config_options, mpi_config, True)
                time_str = "{}-{} hour acc fcst".format(forcings_or_precip.fcst_hour1, forcings_or_precip.fcst_hour2) if grib_var == "APCP" else str(forcings_or_precip.fcst_hour2) + " hour fcst"
                fields.append(":" + grib_var + ":" + forcings_or_precip.grib_levels[force_count] + ":" + time_str + ":")
            cmd = '$WGRIB2 -match "(' + "|".join(fields) + ')" -not "prob" -not "ens" ' + forcings_or_precip.file_in1 + " -netcdf " + nbm_tmp_nc
        else:
            fieldnbm_match1 = '":APCP:"'
            fieldnbm_match2 = '"' + str(forcings_or_precip.fcst_hour1) + "-" + str(forcings_or_precip.fcst_hour2) + '"'
            fieldnbm_notmatch1 = '"prob"'
            cmd = "$WGRIB2 " + forcings_or_precip.file_in1 + " -match " + fieldnbm_match1 + " -match " + fieldnbm_match2 + " -not " + fieldnbm_notmatch1 + " -netcdf " + nbm_tmp_nc

        id_tmp = ioMod.open_grib2(forcings_or_precip.file_in1, nbm_tmp_nc, cmd, config_options, mpi_config, forcings_or_precip.netcdf_var_names[0], special_case=True)
        err_handler.check_program_status(config_options, mpi_config)

        config_options.statusMsg = forcings_or_precip.file_in1
        err_handler.log_msg(config_options, mpi_config)

        config_options.statusMsg = "Processing NBM Variables"
        err_handler.log_msg(config_options, mpi_config)

        for force_count, nc_var in enumerate(forcings_or_precip.netcdf_var_names):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Processing NBM Variable: " + nc_var
                err_handler.log_msg(config_options, mpi_config, True)

            is_supp = forcings_or_precip.grib_vars is None
            if is_supp:
                tag = "supplemental precip"
                calc_regrid_flag = check_supp_pcp_regrid_status(id_tmp, forcings_or_precip, config_options, wrf_hydro_geo_meta, mpi_config)
            else:
                tag = "input"
                calc_regrid_flag = check_regrid_status(id_tmp, force_count, forcings_or_precip, config_options, wrf_hydro_geo_meta, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if is_supp:
                    if mpi_config.rank == 0:
                        config_options.statusMsg = f"Calculating NBM {tag} regridding weights."
                        err_handler.log_msg(config_options, mpi_config, True)
                    calculate_supp_pcp_weights(forcings_or_precip, id_tmp, forcings_or_precip.file_in1, config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    if mpi_config.rank == 0:
                        config_options.statusMsg = f"Calculating NBM {tag} regridding weights."
                        err_handler.log_msg(config_options, mpi_config, True)
                    calculate_weights(id_tmp, force_count, forcings_or_precip, config_options, mpi_config, fill=True)
                    err_handler.check_program_status(config_options, mpi_config)

                    if config_options.grid_meta is None:
                        config_options.errMsg = "No NBM height file supplied, downscaling will not be available"
                        err_handler.log_warning(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)
                    else:
                        if not os.path.exists(config_options.grid_meta):
                            config_options.errMsg = f'NBM height file "{config_options.grid_meta}" does not exist'
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                        terrain_tmp = os.path.join(config_options.scratch_dir, "nbm_terrain_temp.nc")
                        cmd = f"$WGRIB2 {config_options.grid_meta} -netcdf {terrain_tmp}"
                        hgt_tmp = ioMod.open_grib2(config_options.grid_meta, terrain_tmp, cmd, config_options, mpi_config, "DIST_surface")
                        if config_options.grid_type == "gridded":
                            var_tmp = hgt_tmp.variables["DIST_surface"][0, :, :] if mpi_config.rank == 0 else None
                            err_handler.check_program_status(config_options, mpi_config)

                            var_sub_tmp = mpi_config.scatter_array(forcings_or_precip, var_tmp, config_options)
                            err_handler.check_program_status(config_options, mpi_config)

                            try:
                                forcings_or_precip.esmf_field_in.data[:, :] = var_sub_tmp
                            except (ValueError, KeyError, AttributeError) as err:
                                config_options.errMsg = "Unable to place NBM elevation data into the ESMF field object: " + str(err)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            if mpi_config.rank == 0:
                                config_options.statusMsg = "Regridding NBM elevation data to the WRF-Hydro domain."
                                err_handler.log_msg(config_options, mpi_config, True)
                            try:
                                forcings_or_precip.esmf_field_out = esmf_regridobj_call_retry_partial(forcings_or_precip.regridObj, forcings_or_precip.esmf_field_in, forcings_or_precip.esmf_field_out)
                            except ValueError as ve:
                                config_options.errMsg = "Unable to regrid NBM elevation data to the WRF-Hydro domain using ESMF: " + str(ve)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            try:
                                forcings_or_precip.esmf_field_out.data[np.where(forcings_or_precip.regridded_mask == 0)] = config_options.globalNdv
                            except (ValueError, ArithmeticError) as npe:
                                config_options.errMsg = "Unable to compute mask on NBM elevation data: " + str(npe)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            try:
                                forcings_or_precip.height[:, :] = forcings_or_precip.esmf_field_out.data
                            except (ValueError, KeyError, AttributeError) as err:
                                config_options.errMsg = "Unable to extract ESMF regridded NBM elevation data to a local array: " + str(err)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            if mpi_config.rank == 0:
                                hgt_tmp.close()
                        elif config_options.grid_type == "hydrofabric":
                            var_tmp = hgt_tmp.variables["DIST_surface"][0, :, :] if mpi_config.rank == 0 else None
                            err_handler.check_program_status(config_options, mpi_config)

                            var_sub_tmp = mpi_config.scatter_array(forcings_or_precip, var_tmp, config_options)
                            err_handler.check_program_status(config_options, mpi_config)

                            try:
                                forcings_or_precip.esmf_field_in.data[:, :] = var_sub_tmp
                            except (ValueError, KeyError, AttributeError) as err:
                                config_options.errMsg = "Unable to place NBM elevation data into the ESMF field object: " + str(err)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            if mpi_config.rank == 0:
                                config_options.statusMsg = "Regridding NBM elevation data to the WRF-Hydro domain."
                                err_handler.log_msg(config_options, mpi_config, True)
                            try:
                                forcings_or_precip.esmf_field_out = esmf_regridobj_call_retry_partial(forcings_or_precip.regridObj, forcings_or_precip.esmf_field_in, forcings_or_precip.esmf_field_out)
                            except ValueError as ve:
                                config_options.errMsg = "Unable to regrid NBM elevation data to the WRF-Hydro domain using ESMF: " + str(ve)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            try:
                                forcings_or_precip.esmf_field_out.data[np.where(forcings_or_precip.regridded_mask == 0)] = config_options.globalNdv
                            except (ValueError, ArithmeticError) as npe:
                                config_options.errMsg = "Unable to compute mask on NBM elevation data: " + str(npe)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            try:
                                forcings_or_precip.height[:] = forcings_or_precip.esmf_field_out.data
                            except (ValueError, KeyError, AttributeError) as err:
                                config_options.errMsg = "Unable to extract ESMF regridded NBM elevation data to a local array: " + str(err)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            if mpi_config.rank == 0:
                                hgt_tmp.close()
                        elif config_options.grid_type == "unstructured":
                            var_tmp = hgt_tmp.variables["DIST_surface"][0, :, :] if mpi_config.rank == 0 else None
                            err_handler.check_program_status(config_options, mpi_config)

                            var_sub_tmp = mpi_config.scatter_array(forcings_or_precip, var_tmp, config_options)
                            err_handler.check_program_status(config_options, mpi_config)

                            try:
                                forcings_or_precip.esmf_field_in.data[:, :] = var_sub_tmp
                            except (ValueError, KeyError, AttributeError) as err:
                                config_options.errMsg = "Unable to place NBM elevation data into the ESMF field object: " + str(err)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            if mpi_config.rank == 0:
                                config_options.statusMsg = "Regridding NBM elevation data to the WRF-Hydro domain."
                                err_handler.log_msg(config_options, mpi_config, True)
                            try:
                                forcings_or_precip.esmf_field_out = esmf_regridobj_call_retry_partial(forcings_or_precip.regridObj, forcings_or_precip.esmf_field_in, forcings_or_precip.esmf_field_out)
                            except ValueError as ve:
                                config_options.errMsg = "Unable to regrid NBM elevation data to the WRF-Hydro domain using ESMF: " + str(ve)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            try:
                                forcings_or_precip.esmf_field_out.data[np.where(forcings_or_precip.regridded_mask == 0)] = config_options.globalNdv
                            except (ValueError, ArithmeticError) as npe:
                                config_options.errMsg = "Unable to compute mask on NBM elevation data: " + str(npe)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            try:
                                forcings_or_precip.height[:] = forcings_or_precip.esmf_field_out.data
                            except (ValueError, KeyError, AttributeError) as err:
                                config_options.errMsg = "Unable to extract ESMF regridded NBM elevation data to a local array: " + str(err)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            var_tmp_elem = hgt_tmp.variables["DIST_surface"][0, :, :] if mpi_config.rank == 0 else None
                            err_handler.check_program_status(config_options, mpi_config)

                            var_sub_tmp_elem = mpi_config.scatter_array(forcings_or_precip, var_tmp_elem, config_options)
                            err_handler.check_program_status(config_options, mpi_config)

                            try:
                                forcings_or_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                            except (ValueError, KeyError, AttributeError) as err:
                                config_options.errMsg = "Unable to place NBM elevation data into the ESMF field object: " + str(err)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            if mpi_config.rank == 0:
                                config_options.statusMsg = "Regridding NBM elevation data to the unstructured domain."
                                err_handler.log_msg(config_options, mpi_config, True)
                            try:
                                forcings_or_precip.esmf_field_out_elem = esmf_regridobj_call_retry_partial(forcings_or_precip.regridObj_elem, forcings_or_precip.esmf_field_in_elem, forcings_or_precip.esmf_field_out_elem)
                            except ValueError as ve:
                                config_options.errMsg = "Unable to regrid NBM elevation data to the unstructured domain using ESMF: " + str(ve)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            try:
                                forcings_or_precip.esmf_field_out_elem.data[np.where(forcings_or_precip.regridded_mask_elem == 0)] = config_options.globalNdv
                            except (ValueError, ArithmeticError) as npe:
                                config_options.errMsg = "Unable to compute element mask on NBM elevation data: " + str(npe)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            try:
                                forcings_or_precip.height_elem[:] = forcings_or_precip.esmf_field_out_elem.data
                            except (ValueError, KeyError, AttributeError) as err:
                                config_options.errMsg = "Unable to extract ESMF regridded NBM elevation data to a local array: " + str(err)
                                err_handler.log_critical(config_options, mpi_config)
                            err_handler.check_program_status(config_options, mpi_config)

                            if mpi_config.rank == 0:
                                hgt_tmp.close()

            if config_options.grid_type == "gridded":
                var_tmp = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding NBM {nc_var}"
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        var_tmp = id_tmp.variables[nc_var][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract data from NBM file: " + forcings_or_precip.file_in1 + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(forcings_or_precip, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    forcings_or_precip.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = f"Unable to place NBM {tag} into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    forcings_or_precip.esmf_field_out = esmf_regridobj_call_retry_partial(forcings_or_precip.regridObj, forcings_or_precip.esmf_field_in, forcings_or_precip.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = f"Unable to regrid NBM {tag}: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    forcings_or_precip.esmf_field_out.data[np.where(forcings_or_precip.regridded_mask == 0)] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to run mask search on NBM supplemental precipitation: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                destination1 = forcings_or_precip.regridded_precip1 if is_supp else forcings_or_precip.regridded_forcings1[forcings_or_precip.input_map_output[force_count]]
                destination2 = forcings_or_precip.regridded_precip2 if is_supp else forcings_or_precip.regridded_forcings2[forcings_or_precip.input_map_output[force_count]]

                destination2[:, :] = forcings_or_precip.esmf_field_out.data
                err_handler.check_program_status(config_options, mpi_config)

                if "APCP" in nc_var:
                    try:
                        ind_valid = np.where(destination2 != config_options.globalNdv)
                        if forcings_or_precip.input_frequency == 60.0:
                            destination2[ind_valid] = destination2[ind_valid] / 3600.0
                        elif forcings_or_precip.input_frequency == 360.0:
                            destination2[ind_valid] = destination2[ind_valid] / 21600.0
                        del ind_valid
                    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                        config_options.errMsg = "Unable to run NDV search on NBM precipitation: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    destination1[:, :] = destination2[:, :]
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "hydrofabric":
                var_tmp = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding NBM {nc_var}"
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        var_tmp = id_tmp.variables[nc_var][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract data from NBM file: " + forcings_or_precip.file_in1 + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(forcings_or_precip, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    forcings_or_precip.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = f"Unable to place NBM {tag} into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    forcings_or_precip.esmf_field_out = esmf_regridobj_call_retry_partial(forcings_or_precip.regridObj, forcings_or_precip.esmf_field_in, forcings_or_precip.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = f"Unable to regrid NBM {tag}: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    forcings_or_precip.esmf_field_out.data[np.where(forcings_or_precip.regridded_mask == 0)] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to run mask search on NBM supplemental precipitation: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                destination1 = forcings_or_precip.regridded_precip1 if is_supp else forcings_or_precip.regridded_forcings1[forcings_or_precip.input_map_output[force_count]]
                destination2 = forcings_or_precip.regridded_precip2 if is_supp else forcings_or_precip.regridded_forcings2[forcings_or_precip.input_map_output[force_count]]

                destination2[:] = forcings_or_precip.esmf_field_out.data
                err_handler.check_program_status(config_options, mpi_config)

                if "APCP" in nc_var:
                    try:
                        ind_valid = np.where(destination2 != config_options.globalNdv)
                        if forcings_or_precip.input_frequency == 60.0:
                            destination2[ind_valid] = destination2[ind_valid] / 3600.0
                        elif forcings_or_precip.input_frequency == 360.0:
                            destination2[ind_valid] = destination2[ind_valid] / 21600.0
                        del ind_valid
                    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                        config_options.errMsg = "Unable to run NDV search on NBM precipitation: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    destination1[:] = destination2[:]
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding NBM {nc_var}"
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        var_tmp = id_tmp.variables[nc_var][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract data from NBM file: " + forcings_or_precip.file_in1 + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(forcings_or_precip, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    forcings_or_precip.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = f"Unable to place NBM {tag} into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    forcings_or_precip.esmf_field_out = esmf_regridobj_call_retry_partial(forcings_or_precip.regridObj, forcings_or_precip.esmf_field_in, forcings_or_precip.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = f"Unable to regrid NBM {tag}: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    forcings_or_precip.esmf_field_out.data[np.where(forcings_or_precip.regridded_mask == 0)] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to run mask search on NBM supplemental precipitation: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                destination1 = forcings_or_precip.regridded_precip1 if is_supp else forcings_or_precip.regridded_forcings1[forcings_or_precip.input_map_output[force_count]]
                destination2 = forcings_or_precip.regridded_precip2 if is_supp else forcings_or_precip.regridded_forcings2[forcings_or_precip.input_map_output[force_count]]

                destination2[:] = forcings_or_precip.esmf_field_out.data
                err_handler.check_program_status(config_options, mpi_config)

                if "APCP" in nc_var:
                    try:
                        ind_valid = np.where(destination2 != config_options.globalNdv)
                        if forcings_or_precip.input_frequency == 60.0:
                            destination2[ind_valid] = destination2[ind_valid] / 3600.0
                        elif forcings_or_precip.input_frequency == 360.0:
                            destination2[ind_valid] = destination2[ind_valid] / 21600.0
                        del ind_valid
                    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                        config_options.errMsg = "Unable to run NDV search on NBM precipitation: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    destination1[:] = destination2[:]
                err_handler.check_program_status(config_options, mpi_config)

                var_tmp_elem = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Regridding NBM {nc_var}"
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        var_tmp_elem = id_tmp.variables[nc_var][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to extract data from NBM file: " + forcings_or_precip.file_in1 + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(forcings_or_precip, var_tmp_elem, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    forcings_or_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = f"Unable to place NBM {tag} into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    forcings_or_precip.esmf_field_out_elem = esmf_regridobj_call_retry_partial(forcings_or_precip.regridObj_elem, forcings_or_precip.esmf_field_in_elem, forcings_or_precip.esmf_field_out_elem)
                except ValueError as ve:
                    config_options.errMsg = f"Unable to regrid NBM {tag}: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    forcings_or_precip.esmf_field_out_elem.data[np.where(forcings_or_precip.regridded_mask_elem == 0)] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to run mask search on NBM supplemental precipitation: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                destination1_elem = forcings_or_precip.regridded_precip1_elem if is_supp else forcings_or_precip.regridded_forcings1_elem[forcings_or_precip.input_map_output[force_count]]
                destination2_elem = forcings_or_precip.regridded_precip2_elem if is_supp else forcings_or_precip.regridded_forcings2_elem[forcings_or_precip.input_map_output[force_count]]

                destination2_elem[:] = forcings_or_precip.esmf_field_out_elem.data
                err_handler.check_program_status(config_options, mpi_config)

                if "APCP" in nc_var:
                    try:
                        ind_valid_elem = np.where(destination2_elem != config_options.globalNdv)
                        if forcings_or_precip.input_frequency == 60.0:
                            destination2_elem[ind_valid_elem] = destination2_elem[ind_valid_elem] / 3600.0
                        elif forcings_or_precip.input_frequency == 360.0:
                            destination2_elem[ind_valid_elem] = destination2_elem[ind_valid_elem] / 21600.0
                        del ind_valid_elem
                    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                        config_options.errMsg = "Unable to run NDV search on NBM precipitation: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    destination1_elem[:] = destination2_elem[:]
                err_handler.check_program_status(config_options, mpi_config)

    finally:
        if mpi_config.rank == 0 and id_tmp is not None:
            try:
                id_tmp.close()
            except Exception as e:
                config_options.errMsg = f"Unable to close NetCDF file: {nbm_tmp_nc} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
            try:
                os_utils.os_remove_retry(nbm_tmp_nc)
            except Exception as e:
                config_options.errMsg = f"Unable to remove temporary NBM NetCDF file: {nbm_tmp_nc} - {e}\n{traceback.format_exc()}"
                err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

@static_vars(last_file=None)
def regrid_ndfd(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid NDFD forcing data to the WRF-Hydro domain."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No NDFD regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config, True)
        return

    config_options.statusMsg = "Regrid NDFD"
    err_handler.log_msg(config_options, mpi_config)

    hour = input_forcings.fcst_hour2
    current_cycle = config_options.current_fcst_cycle
    forecast_time = config_options.current_time

    ndfd_files = ("tmp", "wdir", "wspd", "qpf")
    fill_values = {"tmp": 288.0, "wdir": 45.0, "wspd": 0.71, "qpf": 0}

    reuse_prev_file = f"{input_forcings.file_in2}-{current_cycle}" == regrid_ndfd.last_file
    regrid_ndfd.last_file = f"{input_forcings.file_in2}-{current_cycle}"

    for i, ndfd_var in enumerate(ndfd_files):
        grb_file = input_forcings.file_in2.replace("%FIELD%", ndfd_var)

        tmp_file = os.path.join(config_options.scratch_dir, f"temp_ndfd_conus_{ndfd_var}.nc")
        if not reuse_prev_file and mpi_config.rank == 0:
            if os.path.isfile(tmp_file):
                config_options.statusMsg = f"Deleting old temporary file: {tmp_file}"
                err_handler.log_msg(config_options, mpi_config, True)
                try:
                    os_utils.os_remove_retry(tmp_file)
                except OSError:
                    config_options.errMsg = f"Unable to remove file: {tmp_file}"
                    err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        id_tmp = None
        try:
            if reuse_prev_file:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Cycle unchanged, reusing temporary file: {tmp_file}"
                    err_handler.log_msg(config_options, mpi_config, True)
                id_tmp = ioMod.open_netcdf_forcing(tmp_file, config_options, mpi_config)
            else:
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"New forecast cycle, creating temporary file from: {grb_file}, cycle hour {current_cycle.hour}"
                    err_handler.log_msg(config_options, mpi_config, True)
                if not WGRIB2_env:
                    cmd = [f":d={current_cycle.strftime('%Y%m%d%H')}:"]
                else:
                    cmd = f"$WGRIB2 -match :d={current_cycle.strftime('%Y%m%d%H')}: -vt {grb_file} | sort -t = -k 2 | $WGRIB2 -i -netcdf {tmp_file} {grb_file}"
                id_tmp = ioMod.open_grib2(grb_file, tmp_file, cmd, config_options, mpi_config, inputVar=None, special_case=True)

            skip_file = np.ubyte(0)
            if mpi_config.rank == 0:
                times = [datetime.utcfromtimestamp(t) for t in id_tmp["time"][:]]
                if ndfd_var != "qpf":
                    if forecast_time > times[-1] + timedelta(hours=3):
                        config_options.statusMsg = "Forecast time beyond NDFD range, skipping"
                        err_handler.log_msg(config_options, mpi_config, True)
                        skip_file = 1
                    elif forecast_time in times:
                        time_index = times.index(forecast_time)
                        config_options.statusMsg = f"Found exact time {forecast_time} in NDFD file at index {time_index} for variable {ndfd_var}"
                        err_handler.log_msg(config_options, mpi_config, True)
                    else:
                        time_index = min(range(len(times)), key=lambda i: abs(times[i] - forecast_time))
                        config_options.statusMsg = f"Exact time {forecast_time} not found in NDFD file, using closest time at {times[time_index]}"
                        err_handler.log_msg(config_options, mpi_config, True)
                else:
                    if forecast_time > times[-1] - timedelta(hours=6):
                        config_options.statusMsg = "Forecast time beyond NDFD precip range, skipping"
                        err_handler.log_msg(config_options, mpi_config, True)
                        skip_file = 1
                    else:
                        time_index = int(hour // 6)
                        config_options.statusMsg = f"Forecast hour {forecast_time} will use precip from {times[time_index] - timedelta(hours=6)} to {times[time_index]}"
                        err_handler.log_msg(config_options, mpi_config, True)

            skip_file = mpi_config.broadcast_parameter(skip_file, config_options, param_type=np.ubyte)
            err_handler.check_program_status(config_options, mpi_config)

            if skip_file:
                if config_options.grid_type == "gridded":
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[i], :, :] = config_options.globalNdv
                elif config_options.grid_type == "hydrofabric":
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[i], :] = config_options.globalNdv
                elif config_options.grid_type == "unstructured":
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[i], :] = config_options.globalNdv
                    input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[i], :] = config_options.globalNdv
                continue

            if mpi_config.rank == 0:
                config_options.statusMsg = "Processing NDFD variable: " + ndfd_var
                err_handler.log_msg(config_options, mpi_config, True)

            calc_regrid_flag = check_regrid_status(id_tmp, i, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Calculating NDFD regridding weights."
                    err_handler.log_msg(config_options, mpi_config, True)
                calculate_weights(id_tmp, i, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta)
                err_handler.check_program_status(config_options, mpi_config)

            var_tmp = None
            if mpi_config.rank == 0:
                try:
                    var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[i]][time_index, :, :]
                    var_tmp = var_tmp.filled(config_options.globalNdv)
                    if config_options.grid_type == "unstructured":
                        var_tmp_elem = id_tmp.variables[input_forcings.netcdf_var_names[i]][time_index, :, :]
                        var_tmp_elem = var_tmp_elem.filled(config_options.globalNdv)
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = f"Unable to extract NDFD variable: {input_forcings.netcdf_var_names[i]} from: {input_forcings.tmpFile} ({err})"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "unstructured":
                var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                err_handler.check_program_status(config_options, mpi_config)
            try:
                if config_options.grid_type == "gridded":
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                elif config_options.grid_type == "hydrofabric":
                    input_forcings.esmf_field_in.data[:] = var_sub_tmp
                elif config_options.grid_type == "unstructured":
                    input_forcings.esmf_field_in.data[:] = var_sub_tmp
                    input_forcings.esmf_field_in_elem.data[:] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                if config_options.grid_type == "unstructured":
                    input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
            except ValueError as ve:
                config_options.errMsg = f"Unable to regrid input NDFD forcing variable using ESMF: {ve}"
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = config_options.globalNdv
                if config_options.grid_type == "unstructured":
                    input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                config_options.errMsg = f"Unable to calculate mask from input NDFD regridded forcings: {npe}"
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            def u_v_comps(wspd, wdir):
                rads = wdir * (np.pi / 180.0)
                u = -1 * wspd * np.sin(rads)
                v = -1 * wspd * np.cos(rads)
                return u, v

            if ndfd_var == "wspd":
                try:
                    ind_valid = np.where(input_forcings.esmf_field_out.data != config_options.globalNdv)
                    wspd = input_forcings.esmf_field_out.data[ind_valid]
                    wdir = input_forcings.regridded_forcings2[input_forcings.input_map_output[i - 1]][ind_valid]

                    u, v = u_v_comps(wspd, wdir)
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[i - 1]][ind_valid] = u
                    input_forcings.esmf_field_out.data[ind_valid] = v

                    if config_options.grid_type == "unstructured":
                        ind_valid = np.where(input_forcings.esmf_field_out_elem.data != config_options.globalNdv)
                        wspd = input_forcings.esmf_field_out_elem.data[ind_valid]
                        wdir = input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[i - 1]][ind_valid]

                        u, v = u_v_comps(wspd, wdir)
                        input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[i - 1]][ind_valid] = u
                        input_forcings.esmf_field_out_elem.data[ind_valid] = v

                    del ind_valid, u, v

                except (IndexError, ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    config_options.errMsg = f"Unable to convert NDFD wind speed/dir to U/V components: {npe}"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            if ndfd_var == "qpf":
                try:
                    hours = 6
                    ind_valid = np.where(input_forcings.esmf_field_out.data != config_options.globalNdv)
                    input_forcings.esmf_field_out.data[ind_valid] = input_forcings.esmf_field_out.data[ind_valid] / (hours * 3600.0)
                    if config_options.grid_type == "unstructured":
                        ind_valid = np.where(input_forcings.esmf_field_out_elem.data != config_options.globalNdv)
                        input_forcings.esmf_field_out_elem.data[ind_valid] = input_forcings.esmf_field_out_elem.data[ind_valid] / (hours * 3600.0)
                    del ind_valid
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    config_options.errMsg = f"Unable to run NDV search on NDFD QPF precipitation: {npe}"
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

            try:
                if config_options.grid_type == "gridded":
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[i], :, :] = input_forcings.esmf_field_out.data
                elif config_options.grid_type == "hydrofabric":
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[i], :] = input_forcings.esmf_field_out.data
                elif config_options.grid_type == "unstructured":
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[i], :] = input_forcings.esmf_field_out.data
                    input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[i], :] = input_forcings.esmf_field_out_elem.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)
        finally:
            if id_tmp is not None:
                try:
                    id_tmp.close()
                except OSError as e:
                    config_options.errMsg = f"Unable to close NDFD temp file {tmp_file}: {e}"
                    err_handler.log_critical(config_options, mpi_config)

            if not reuse_prev_file and mpi_config.rank == 0 and os.path.isfile(tmp_file):
                try:
                    os_utils.os_remove_retry(tmp_file)
                except FileNotFoundError:
                    config_options.statusMsg = f"NetCDF file not found, continuing: {input_forcings.tmpFile}"
                    err_handler.log_warning(config_options, mpi_config)

                except Exception as e:
                    config_options.errMsg = f"Unable to remove scratch file {tmp_file}: {e}\n{traceback.format_exc()}"
                    err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)


def regrid_aorc_aws(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid AORC AWS forcing data to the WRF-Hydro domain."""
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    fill_values = {"TMP": 288.0, "SPFH": 0.005, "PRES": 101300.0, "APCP": 0, "UGRD": 1.0, "VGRD": 1.0, "DSWRF": 80.0, "DLWRF": 310.0}
    with timing_block(f"regrid step 1 | {mpi_config.rank}: get dataset"):
        mpi_config.comm.barrier()
        id_tmp = config_options.aws_obj

        config_options.statusMsg = "Processing Custom NetCDF Forcing Variables"
        err_handler.log_msg(config_options, mpi_config, True)

    with timing_block(f"regrid step 2 | {mpi_config.rank}: loop over variables"):
        for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
            if mpi_config.rank == 0:
                config_options.statusMsg = "Processing Custom NetCDF Forcing Variable: " + nc_var
                err_handler.log_msg(config_options, mpi_config, True)
            with timing_block(f"regrid step 2.1 | {mpi_config.rank}: check regrid status"):
                calc_regrid_flag = check_regrid_status(id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config)

            with timing_block(f"regrid step 2.2 | {mpi_config.rank}: calculate weights"):
                if calc_regrid_flag:
                    calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta, lat_var="y", lon_var="x")

            with timing_block(f"regrid step 2.3 | {mpi_config.rank}: ERA5-interim mask"):
                if 23 in config_options.input_forcings:
                    input_forcings.regridded_mask_AORC = input_forcings.regridded_mask
                    if config_options.grid_type == "unstructured":
                        input_forcings.regridded_mask_elem_AORC = input_forcings.regridded_mask_elem

            if config_options.grid_type == "gridded":
                var_tmp = None
                fill = fill_values.get(input_forcings.grib_vars[force_count], config_options.globalNdv)
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding Custom netCDF input variable: " + nc_var
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        config_options.statusMsg = f"Using {fill} to replace missing values in input"
                        err_handler.log_msg(config_options, mpi_config, True)
                        var_tmp = id_tmp[nc_var].to_masked_array().filled(fill)
                    except Exception as err:
                        config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input Custom netCDF forcing variables using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = fill
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to calculate mask from input Custom netCDF regridded forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if nc_var == "APCP_surface":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != fill)
                        input_forcings.esmf_field_out.data[ind_valid] = input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        del ind_valid
                    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                        config_options.errMsg = "Unable to run NDV search on Custom netCDF precipitation: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                var_tmp = None
                fill = fill_values.get(input_forcings.grib_vars[force_count], config_options.globalNdv)
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding Custom netCDF input variable: " + nc_var
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        config_options.statusMsg = f"Using {fill} to replace missing values in input"
                        err_handler.log_msg(config_options, mpi_config, True)
                        var_tmp = id_tmp[nc_var].to_masked_array().filled(fill)
                    except Exception as err:
                        config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input Custom netCDF forcing variables using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = fill
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to calculate mask from input Custom netCDF regridded forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if nc_var == "APCP_surface":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != fill)
                        input_forcings.esmf_field_out.data[ind_valid] = input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        del ind_valid
                    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                        config_options.errMsg = "Unable to run NDV search on Custom netCDF precipitation: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)

                var_tmp_elem = None
                fill = fill_values.get(input_forcings.grib_vars[force_count], config_options.globalNdv)
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding Custom netCDF input variable: " + nc_var
                    err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        config_options.statusMsg = f"Using {fill} to replace missing values in input"
                        err_handler.log_msg(config_options, mpi_config, True)
                        var_tmp_elem = id_tmp[nc_var].to_masked_array().filled(fill)
                    except Exception as err:
                        config_options.errMsg = "Unable to extract " + nc_var + " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                        err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(input_forcings, var_tmp_elem, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem = esmf_regridobj_call_retry_partial(input_forcings.regridObj_elem, input_forcings.esmf_field_in_elem, input_forcings.esmf_field_out_elem)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid input Custom netCDF forcing variables using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem.data[np.where(input_forcings.regridded_mask_elem == 0)] = fill
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to calculate mask from input Custom netCDF regridded forcings: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if nc_var == "APCP_surface":
                    try:
                        ind_valid_elem = np.where(input_forcings.esmf_field_out_elem.data != fill)
                        input_forcings.esmf_field_out_elem.data[ind_valid_elem] = input_forcings.esmf_field_out_elem.data[ind_valid_elem] / 3600.0
                        del ind_valid_elem
                    except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                        config_options.errMsg = "Unable to run NDV search on Custom netCDF precipitation: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2_elem[input_forcings.input_map_output[force_count], :]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "hydrofabric":
                with timing_block(f"regrid step 2.4 | {mpi_config.rank}: get fill value"):
                    var_tmp = None
                    fill = fill_values.get(input_forcings.grib_vars[force_count], config_options.globalNdv)

                if mpi_config.rank == 0:
                    with timing_block(f"regrid step 2.5.1 | {mpi_config.rank}: regrid messages"):
                        config_options.statusMsg = "Regridding Custom netCDF input variable: " + nc_var
                        err_handler.log_msg(config_options, mpi_config, True)
                    try:
                        with timing_block(f"regrid step 2.5.2 | {mpi_config.rank}: messages2"):
                            config_options.statusMsg = f"Using {fill} to replace missing values in input"
                            err_handler.log_msg(config_options, mpi_config, True)
                        with timing_block(f"regrid step 2.5.3 | {mpi_config.rank}: fill missing"):
                            var_tmp = id_tmp[nc_var].to_masked_array().filled(fill)

                    except Exception as err:
                        with timing_block(f"regrid step 2.5.4 | {mpi_config.rank}: extract error"):
                            config_options.errMsg = f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({str(err)})"
                            err_handler.log_critical(config_options, mpi_config)

                with timing_block(f"regrid step 2.5.5 | {mpi_config.rank}: check program status"):
                    err_handler.check_program_status(config_options, mpi_config)

                with timing_block(f"regrid step 2.6 | {mpi_config.rank}: scatter input variable"):
                    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
                    err_handler.check_program_status(config_options, mpi_config)
                with timing_block(f"regrid step 2.7 | {mpi_config.rank}: place local array into local ESMF field"):
                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                with timing_block(f"regrid step 2.8 | {mpi_config.rank}: regrid input variable ESMF"):
                    try:
                        input_forcings.esmf_field_out = esmf_regridobj_call_retry_partial(input_forcings.regridObj, input_forcings.esmf_field_in, input_forcings.esmf_field_out)
                    except ValueError as ve:
                        config_options.errMsg = "Unable to regrid input Custom netCDF forcing variables using ESMF: " + str(ve)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                with timing_block(f"regrid step 2.9 | {mpi_config.rank}: set pixels outside input domain to missing value"):
                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = fill
                    except (ValueError, ArithmeticError) as npe:
                        config_options.errMsg = "Unable to calculate mask from input Custom netCDF regridded forcings: " + str(npe)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                with timing_block(f"regrid step 2.10 | {mpi_config.rank}: convert hourly precipitation to mm/s"):
                    if nc_var == "APCP_surface":
                        try:
                            ind_valid = np.where(input_forcings.esmf_field_out.data != fill)
                            input_forcings.esmf_field_out.data[ind_valid] = input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                            ind_valid = np.where(input_forcings.esmf_field_out.data < 0.0)
                            input_forcings.esmf_field_out.data[ind_valid] = 0.0
                            del ind_valid
                        except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                            config_options.errMsg = "Unable to run NDV search on Custom netCDF precipitation: " + str(npe)
                            err_handler.log_critical(config_options, mpi_config)
                        err_handler.check_program_status(config_options, mpi_config)

                with timing_block(f"regrid step 2.11 | {mpi_config.rank}: place regridded data into local array"):
                    try:
                        input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
                        err_handler.log_critical(config_options, mpi_config)
                    err_handler.check_program_status(config_options, mpi_config)

                with timing_block(f"regrid step 2.12 | {mpi_config.rank}: update previous regridded field for first timestep"):
                    if config_options.current_output_step == 1:
                        input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :] = input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :]
                    err_handler.check_program_status(config_options, mpi_config)


def check_regrid_status(id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Check regrid status function."""
    esmf_field_retry_partial = functools.partial(esmf_field_retry, mpi_config, config_options, err_handler)

    if not input_forcings.esmf_field_out:
        if config_options.grid_type == "gridded":
            try:
                input_forcings.esmf_field_out = esmf_field_retry_partial(wrf_hydro_geo_meta.esmf_grid, name=input_forcings.product_name + "FORCING_REGRIDDED")
            except ESMF.ESMPyException as esmf_error:
                config_options.errMsg = "Unable to create " + input_forcings.product_name + " destination ESMF field object: " + str(esmf_error)
                err_handler.log_critical(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            try:
                input_forcings.esmf_field_out = esmf_field_retry_partial(wrf_hydro_geo_meta.esmf_grid, meshloc=ESMF.MeshLoc.NODE, name=input_forcings.product_name + "FORCING_REGRIDDED")
            except ESMF.ESMPyException as esmf_error:
                config_options.errMsg = "Unable to create " + input_forcings.product_name + " destination ESMF field node mesh object: " + str(esmf_error)
                err_handler.log_critical(config_options, mpi_config)
            try:
                input_forcings.esmf_field_out_elem = esmf_field_retry_partial(wrf_hydro_geo_meta.esmf_grid, meshloc=ESMF.MeshLoc.ELEMENT, name=input_forcings.product_name + "FORCING_REGRIDDED")
            except ESMF.ESMPyException as esmf_error:
                config_options.errMsg = "Unable to create " + input_forcings.product_name + " destination ESMF field element mesh object: " + str(esmf_error)
                err_handler.log_critical(config_options, mpi_config)
        elif config_options.grid_type == "hydrofabric":
            try:
                input_forcings.esmf_field_out = esmf_field_retry_partial(wrf_hydro_geo_meta.esmf_grid, meshloc=ESMF.MeshLoc.ELEMENT, name=input_forcings.product_name + "FORCING_REGRIDDED")
            except ESMF.ESMPyException as esmf_error:
                config_options.errMsg = "Unable to create " + input_forcings.product_name + " destination ESMF field element mesh object: " + str(esmf_error)
                err_handler.log_critical(config_options, mpi_config)

    err_handler.check_program_status(config_options, mpi_config)

    calc_regrid_flag = False

    if input_forcings.nx_global is None or input_forcings.ny_global is None:
        force_count = 9 if config_options.include_lqfrac else 8
        if config_options.grid_type == "gridded":
            input_forcings.regridded_forcings1 = np.empty([force_count, wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32)
            input_forcings.regridded_forcings2 = np.empty([force_count, wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32)
        elif config_options.grid_type == "unstructured":
            input_forcings.regridded_forcings1 = np.empty([force_count, wrf_hydro_geo_meta.ny_local], np.float32)
            input_forcings.regridded_forcings2 = np.empty([force_count, wrf_hydro_geo_meta.ny_local], np.float32)
            input_forcings.regridded_forcings1_elem = np.empty([force_count, wrf_hydro_geo_meta.ny_local_elem], np.float32)
            input_forcings.regridded_forcings2_elem = np.empty([force_count, wrf_hydro_geo_meta.ny_local_elem], np.float32)
        elif config_options.grid_type == "hydrofabric":
            input_forcings.regridded_forcings1 = np.empty([force_count, wrf_hydro_geo_meta.ny_local], np.float32)
            input_forcings.regridded_forcings2 = np.empty([force_count, wrf_hydro_geo_meta.ny_local], np.float32)

    if mpi_config.rank == 0:
        if input_forcings.nx_global is None or input_forcings.ny_global is None:
            calc_regrid_flag = True
        else:
            if mpi_config.rank == 0:
                if config_options.aws:
                    if id_tmp.variables[input_forcings.netcdf_var_names[force_count]].shape[0] != input_forcings.ny_global and id_tmp.variables[input_forcings.netcdf_var_names[force_count]].shape[1] != input_forcings.nx_global:
                        calc_regrid_flag = True
                else:
                    if id_tmp.variables[input_forcings.netcdf_var_names[force_count]].shape[1] != input_forcings.ny_global and id_tmp.variables[input_forcings.netcdf_var_names[force_count]].shape[2] != input_forcings.nx_global:
                        calc_regrid_flag = True

    calc_regrid_flag = mpi_config.broadcast_parameter(calc_regrid_flag, config_options, param_type=bool)
    err_handler.check_program_status(config_options, mpi_config)

    return calc_regrid_flag


def check_supp_pcp_regrid_status(id_tmp, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config):
    """Check supplemental precipitation regrid status function."""
    esmf_field_retry_partial = functools.partial(esmf_field_retry, mpi_config, config_options, err_handler)

    if not supplemental_precip.esmf_field_out:
        if config_options.grid_type == "gridded":
            try:
                supplemental_precip.esmf_field_out = esmf_field_retry_partial(wrf_hydro_geo_meta.esmf_grid, name=supplemental_precip.product_name + "SUPP_PCP_REGRIDDED")
            except ESMF.ESMPyException as esmf_error:
                config_options.errMsg = "Unable to create " + supplemental_precip.product_name + " destination ESMF field object: " + str(esmf_error)
                err_handler.err_out(config_options)
        elif config_options.grid_type == "unstructured":
            try:
                supplemental_precip.esmf_field_out = esmf_field_retry_partial(wrf_hydro_geo_meta.esmf_grid, meshloc=ESMF.MeshLoc.NODE, name=supplemental_precip.product_name + "SUPP_PCP_REGRIDDED")
            except ESMF.ESMPyException as esmf_error:
                config_options.errMsg = "Unable to create " + supplemental_precip.product_name + " destination ESMF node field object: " + str(esmf_error)
                err_handler.err_out(config_options)

            try:
                supplemental_precip.esmf_field_out_elem = esmf_field_retry_partial(wrf_hydro_geo_meta.esmf_grid, meshloc=ESMF.MeshLoc.ELEMENT, name=supplemental_precip.product_name + "SUPP_PCP_REGRIDDED")
            except ESMF.ESMPyException as esmf_error:
                config_options.errMsg = "Unable to create " + supplemental_precip.product_name + " destination ESMF element field object: " + str(esmf_error)
                err_handler.err_out(config_options)

        elif config_options.grid_type == "hydrofabric":
            try:
                supplemental_precip.esmf_field_out = esmf_field_retry_partial(wrf_hydro_geo_meta.esmf_grid, meshloc=ESMF.MeshLoc.ELEMENT, name=supplemental_precip.product_name + "SUPP_PCP_REGRIDDED")
            except ESMF.ESMPyException as esmf_error:
                config_options.errMsg = "Unable to create " + supplemental_precip.product_name + " destination ESMF element field object: " + str(esmf_error)
                err_handler.err_out(config_options)

    calc_regrid_flag = False

    if supplemental_precip.nx_global is None or supplemental_precip.ny_global is None:
        if config_options.grid_type == "gridded":
            supplemental_precip.regridded_precip1 = np.empty([wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32)
            supplemental_precip.regridded_precip2 = np.empty([wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32)
            supplemental_precip.regridded_rqi1 = np.empty([wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32)
            supplemental_precip.regridded_rqi2 = np.empty([wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32)
            supplemental_precip.regridded_rqi1[:, :] = config_options.globalNdv
            supplemental_precip.regridded_rqi2[:, :] = config_options.globalNdv
        elif config_options.grid_type == "unstructured":
            supplemental_precip.regridded_precip1 = np.empty([wrf_hydro_geo_meta.ny_local], np.float32)
            supplemental_precip.regridded_precip2 = np.empty([wrf_hydro_geo_meta.ny_local], np.float32)
            supplemental_precip.regridded_rqi1 = np.empty([wrf_hydro_geo_meta.ny_local], np.float32)
            supplemental_precip.regridded_rqi2 = np.empty([wrf_hydro_geo_meta.ny_local], np.float32)
            supplemental_precip.regridded_rqi1[:] = config_options.globalNdv
            supplemental_precip.regridded_rqi2[:] = config_options.globalNdv
            supplemental_precip.regridded_precip1_elem = np.empty([wrf_hydro_geo_meta.ny_local_elem], np.float32)
            supplemental_precip.regridded_precip2_elem = np.empty([wrf_hydro_geo_meta.ny_local_elem], np.float32)
            supplemental_precip.regridded_rqi1_elem = np.empty([wrf_hydro_geo_meta.ny_local_elem], np.float32)
            supplemental_precip.regridded_rqi2_elem = np.empty([wrf_hydro_geo_meta.ny_local_elem], np.float32)
            supplemental_precip.regridded_rqi1_elem[:] = config_options.globalNdv
            supplemental_precip.regridded_rqi2_elem[:] = config_options.globalNdv

        elif config_options.grid_type == "hydrofabric":
            supplemental_precip.regridded_precip1 = np.empty([wrf_hydro_geo_meta.ny_local], np.float32)
            supplemental_precip.regridded_precip2 = np.empty([wrf_hydro_geo_meta.ny_local], np.float32)
            supplemental_precip.regridded_rqi1 = np.empty([wrf_hydro_geo_meta.ny_local], np.float32)
            supplemental_precip.regridded_rqi2 = np.empty([wrf_hydro_geo_meta.ny_local], np.float32)
            supplemental_precip.regridded_rqi1[:] = config_options.globalNdv
            supplemental_precip.regridded_rqi2[:] = config_options.globalNdv

    if mpi_config.rank == 0:
        if supplemental_precip.nx_global is None or supplemental_precip.ny_global is None:
            calc_regrid_flag = True
        else:
            if mpi_config.rank == 0:
                ncvar = id_tmp.variables[supplemental_precip.netcdf_var_names[0]]
                ndims = len(ncvar.dimensions)
                if ndims == 2:
                    latdim = 0
                    londim = 1
                else:
                    latdim = 1
                    londim = 2
                if ncvar.shape[latdim] != supplemental_precip.ny_global and ncvar.shape[londim] != supplemental_precip.nx_global:
                    calc_regrid_flag = True

    if config_options.grid_type == "gridded":
        if supplemental_precip.regridded_precip2 is None:
            supplemental_precip.regridded_precip2 = np.empty([wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32)
        if supplemental_precip.regridded_precip1 is None:
            supplemental_precip.regridded_precip1 = np.empty([wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32)
    elif config_options.grid_type == "unstructured":
        if supplemental_precip.regridded_precip2 is None:
            supplemental_precip.regridded_precip2 = np.empty([wrf_hydro_geo_meta.ny_local], np.float32)
        if supplemental_precip.regridded_precip1 is None:
            supplemental_precip.regridded_precip1 = np.empty([wrf_hydro_geo_meta.ny_local], np.float32)
        if supplemental_precip.regridded_precip2_elem is None:
            supplemental_precip.regridded_precip2_elem = np.empty([wrf_hydro_geo_meta.ny_local_elem], np.float32)
        if supplemental_precip.regridded_precip1_elem is None:
            supplemental_precip.regridded_precip1_elem = np.empty([wrf_hydro_geo_meta.ny_local_elem], np.float32)

    elif config_options.grid_type == "hydrofabric":
        if supplemental_precip.regridded_precip2 is None:
            supplemental_precip.regridded_precip2 = np.empty([wrf_hydro_geo_meta.ny_local], np.float32)
        if supplemental_precip.regridded_precip1 is None:
            supplemental_precip.regridded_precip1 = np.empty([wrf_hydro_geo_meta.ny_local], np.float32)

    calc_regrid_flag = mpi_config.broadcast_parameter(calc_regrid_flag, config_options, param_type=bool)

    mpi_config.comm.barrier()
    return calc_regrid_flag


def get_weight_file_names(mpi_config: MpiConfig, config_options: ConfigOptions, input_forcings: GeoMetaWrfHydro) -> tuple[str | None, str | None]:
    if not config_options.weightsDir:
        return None, None

    grid_key = input_forcings.product_name
    geogrid_key = Path(config_options.geogrid).stem
    file_key = f"{grid_key}_{geogrid_key}"
    hash_key = hashlib.md5(file_key.encode()).hexdigest()[:8]
    hash_key += f"_{mpi_config.uid64}"

    weight_file = os.path.join(config_options.weightsDir, f"ESMF_weight_{file_key}.nc4")

    if config_options.grid_type == "unstructured":
        weight_file_elem = os.path.join(config_options.weightsDir, f"ESMF_weight_{file_key}_elem.nc4")
    else:
        weight_file_elem = None

    return weight_file, weight_file_elem


def load_weight_file(mpi_config: MpiConfig, config_options: ConfigOptions, input_forcings: GeoMetaWrfHydro, weight_file: str, element_mode: bool) -> None:
    """Load cached weight file."""
    os_utils.assert_path_exists_retry(mpi_config, config_options, err_handler, weight_file)

    if not element_mode:
        msg_augment = " "
        field_in = input_forcings.esmf_field_in
        field_out = input_forcings.esmf_field_out
        target_object_attr_name = "regridObj"
    else:
        msg_augment = " mesh element "
        field_in = input_forcings.esmf_field_in_elem
        field_out = input_forcings.esmf_field_out_elem
        target_object_attr_name = "regridObj_elem"

    config_options.statusMsg = f"RANK: {mpi_config.rank}: Loading cached ESMF{msg_augment}weight object for {input_forcings.product_name} from {weight_file}"
    err_handler.log_msg(config_options, mpi_config, True)

    err_handler.check_program_status(config_options, mpi_config)
    try:
        begin = monotonic()
        regrid = esmf_regridfromfile_retry(mpi_config, config_options, err_handler, field_in, field_out, weight_file)
        setattr(input_forcings, target_object_attr_name, regrid)
        end = monotonic()
    except (IOError, ValueError, ESMF.ESMPyException) as esmf_error:
        config_options.errMsg = f"Unable to load cached ESMF{msg_augment}weight file: " + str(esmf_error)
        err_handler.log_warning(config_options, mpi_config)
    else:
        config_options.statusMsg = f"RANK: {mpi_config.rank}: Finished loading{msg_augment}weight object with ESMF, took {end - begin} seconds"
        err_handler.log_msg(config_options, mpi_config, True)

    err_handler.check_program_status(config_options, mpi_config)


def make_regrid(mpi_config: MpiConfig, config_options: ConfigOptions, input_forcings: GeoMetaWrfHydro, weight_file: str | None, fill: bool, element_mode: bool) -> None:
    """Create regrid object using ESMF."""
    assert isinstance(fill, bool)

    if not element_mode:
        msg_augment = " "
        field_in = input_forcings.esmf_field_in
        field_out = input_forcings.esmf_field_out
        target_object_attr_name = "regridObj"
    else:
        msg_augment = " mesh element "
        field_in = input_forcings.esmf_field_in_elem
        field_out = input_forcings.esmf_field_out_elem
        target_object_attr_name = "regridObj_elem"

    start_msg = f"RANK: {mpi_config.rank}: Creating{msg_augment}weight object from ESMF. weight_file={weight_file}"
    if mpi_config.rank == 0:
        config_options.statusMsg = start_msg
        err_handler.log_msg(config_options, mpi_config, True)

    extrap_method = ESMF.ExtrapMethod.CREEP_FILL if fill else ESMF.ExtrapMethod.NONE
    regrid_method = (ESMF.RegridMethod.BILINEAR, ESMF.RegridMethod.NEAREST_STOD)[input_forcings.regridOpt - 1]

    err_handler.check_program_status(config_options, mpi_config)
    try:
        begin = monotonic()
        regrid = esmf_regrid_retry(mpi_config, config_options, err_handler, field_in, field_out, src_mask_values=np.array([0, config_options.globalNdv]), regrid_method=regrid_method, extrap_method=extrap_method, unmapped_action=ESMF.UnmappedAction.IGNORE, filename=weight_file)
        setattr(input_forcings, target_object_attr_name, regrid)
        end = monotonic()
    except (RuntimeError, ImportError, ESMF.ESMPyException) as esmf_error:
        config_options.errMsg = f"RANK: {mpi_config.rank}: Failed: {start_msg}. Unable to regrid input data from ESMF: " + str(esmf_error)
        err_handler.log_critical(config_options, mpi_config)
        etype, value, tb = sys.exc_info()
        traceback.print_exception(etype, value, tb)
    else:
        if mpi_config.rank == 0:
            config_options.statusMsg = f"RANK: {mpi_config.rank}: Finished: {start_msg}, took {end - begin} seconds"
            err_handler.log_msg(config_options, mpi_config, True)

    err_handler.check_program_status(config_options, mpi_config)


def execute_regrid(mpi_config: MpiConfig, config_options: ConfigOptions, input_forcings: GeoMetaWrfHydro, weight_file: str, element_mode: bool) -> None:
    """Execute regrid operation."""
    if not element_mode:
        field_in = input_forcings.esmf_field_in
        field_out = input_forcings.esmf_field_out
        regrid_object = input_forcings.regridObj
        target_object_attr_name = "esmf_field_out"
    else:
        field_in = input_forcings.esmf_field_in_elem
        field_out = input_forcings.esmf_field_out_elem
        regrid_object = input_forcings.regridObj_elem
        target_object_attr_name = "esmf_field_out_elem"

    err_handler.check_program_status(config_options, mpi_config)
    try:
        setattr(input_forcings, target_object_attr_name, regrid_object(field_in, field_out))
    except ValueError as ve:
        config_options.errMsg = "Unable to extract regridded data from ESMF regridded field: " + str(ve)
        err_handler.log_critical(config_options, mpi_config)
        if weight_file is not None:
            config_options.statusMsg = f"Deleting if exists: {weight_file}"
            err_handler.log_msg(config_options, mpi_config, True)
            try:
                os_utils.os_remove_retry(weight_file)
            except FileNotFoundError:
                pass

    err_handler.check_program_status(config_options, mpi_config)


def calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, wrf_hydro_geo_meta, lat_var="latitude", lon_var="longitude", fill=False):
    """Function to calculate ESMF weights."""
    esmf_field_retry_partial = functools.partial(esmf_field_retry, mpi_config, config_options, err_handler)
    esmf_grid_retry_partial = functools.partial(esmf_grid_retry, mpi_config, config_options, err_handler)

    config_options.statusMsg = "Calculate Weights"
    err_handler.log_msg(config_options, mpi_config, True)

    if mpi_config.rank == 0:
        if config_options.aws:
            try:
                input_forcings.ny_global = id_tmp.variables[input_forcings.netcdf_var_names[force_count]].shape[0]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract Y shape size from: " + input_forcings.netcdf_var_names[force_count] + " from aws object"
                err_handler.log_critical(config_options, mpi_config)
            try:
                input_forcings.nx_global = id_tmp.variables[input_forcings.netcdf_var_names[force_count]].shape[1]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract X shape size from: " + input_forcings.netcdf_var_names[force_count] + " from aws object"
                err_handler.log_critical(config_options, mpi_config)
        else:
            try:
                input_forcings.ny_global = id_tmp.variables[input_forcings.netcdf_var_names[force_count]].shape[1]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract Y shape size from: " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
            try:
                input_forcings.nx_global = id_tmp.variables[input_forcings.netcdf_var_names[force_count]].shape[2]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract X shape size from: " + input_forcings.netcdf_var_names[force_count] + " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    input_forcings.ny_global = mpi_config.broadcast_parameter(input_forcings.ny_global, config_options, param_type=int)
    err_handler.check_program_status(config_options, mpi_config)
    input_forcings.nx_global = mpi_config.broadcast_parameter(input_forcings.nx_global, config_options, param_type=int)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.esmf_grid_in = esmf_grid_retry_partial(np.array([input_forcings.ny_global, input_forcings.nx_global]), staggerloc=ESMF.StaggerLoc.CENTER, coord_sys=ESMF.CoordSys.SPH_DEG)
    except ESMF.ESMPyException as esmf_error:
        config_options.errMsg = "Unable to create source ESMF grid from temporary file: " + input_forcings.tmpFile + " (" + str(esmf_error) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.x_lower_bound = input_forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][1]
        input_forcings.x_upper_bound = input_forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][1]
        input_forcings.y_lower_bound = input_forcings.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][0]
        input_forcings.y_upper_bound = input_forcings.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][0]
        input_forcings.nx_local = input_forcings.x_upper_bound - input_forcings.x_lower_bound
        input_forcings.ny_local = input_forcings.y_upper_bound - input_forcings.y_lower_bound
    except (ValueError, KeyError, AttributeError) as err:
        config_options.errMsg = "Unable to extract local X/Y boundaries from global grid from temporary file: " + input_forcings.tmpFile + " (" + str(err) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if input_forcings.nx_local < 2 or input_forcings.ny_local < 2:
        config_options.errMsg = "You have either specified too many cores for: " + input_forcings.product_name + ", or your input forcing grid is too small to process. Local grid must have x/y dimension size of 2."
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    mask = input_forcings.esmf_grid_in.add_item(ESMF.GridItem.MASK, ESMF.StaggerLoc.CENTER)
    mask[:] = np.ones(mask.shape)

    border = input_forcings.border
    if border > 0:
        try:
            mask = input_forcings.esmf_grid_in.add_item(ESMF.GridItem.MASK, ESMF.StaggerLoc.CENTER)
            if mpi_config.rank == 0:
                config_options.statusMsg = "Trimming input forcing `{}` by {} grid cells".format(input_forcings.product_name, border)
                err_handler.log_msg(config_options, mpi_config, True)

            gmask = np.ones([input_forcings.ny_global, input_forcings.nx_global])
            gmask[:+border, :] = 0.0
            gmask[-border:, :] = 0.0
            gmask[:, :+border] = 0.0
            gmask[:, -border:] = 0.0

            mask[:, :] = mpi_config.scatter_array(input_forcings, gmask, config_options)
            err_handler.check_program_status(config_options, mpi_config)
        except Exception as e:
            LOG.error(f"{e}")

    lat_tmp = None
    lon_tmp = None
    if mpi_config.rank == 0:
        if input_forcings.product_name == "NWM":
            nwm_geogrid = nc.Dataset(config_options.nwm_geogrid)

            if hasattr(config_options, "aws_obj") and config_options.aws_obj is not None:
                aws_x = config_options.aws_obj.x.values
                aws_y = config_options.aws_obj.y.values

                full_lat = nwm_geogrid.variables["XLAT_M"][:][0, :, :]
                full_lon = nwm_geogrid.variables["XLONG_M"][:][0, :, :]

                nwm_crs = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
                transformer = Transformer.from_crs(nwm_crs, "EPSG:4326", always_xy=True)

                x_min_proj, x_max_proj = aws_x.min(), aws_x.max()
                y_min_proj, y_max_proj = aws_y.min(), aws_y.max()

                lon_min, lat_min = transformer.transform(x_min_proj, y_min_proj)
                lon_max, lat_max = transformer.transform(x_max_proj, y_max_proj)

                buffer = 0.1
                lat_min -= buffer
                lat_max += buffer
                lon_min -= buffer
                lon_max += buffer

                lat_mask = (full_lat >= lat_min) & (full_lat <= lat_max)
                lon_mask = (full_lon >= lon_min) & (full_lon <= lon_max)
                y_indices, x_indices = np.where(lat_mask & lon_mask)

                if len(y_indices) > 0 and len(x_indices) > 0:
                    y_min_idx, y_max_idx = y_indices.min(), y_indices.max() + 1
                    x_min_idx, x_max_idx = x_indices.min(), x_indices.max() + 1
                    lat_tmp = full_lat[y_min_idx:y_max_idx, x_min_idx:x_max_idx]
                    lon_tmp = full_lon[y_min_idx:y_max_idx, x_min_idx:x_max_idx]
                else:
                    lat_tmp = full_lat
                    lon_tmp = full_lon
            else:
                lat_tmp = nwm_geogrid.variables["XLAT_M"][:][0, :, :]
                lon_tmp = nwm_geogrid.variables["XLONG_M"][:][0, :, :]
            nwm_geogrid.close()
        else:
            if len(id_tmp.variables[lat_var].shape) == 3:
                lat_tmp = id_tmp.variables[lat_var][0, :, :]
                lon_tmp = id_tmp.variables[lon_var][0, :, :]
            elif len(id_tmp.variables[lon_var].shape) == 2:
                lat_tmp = id_tmp.variables[lat_var][:, :]
                lon_tmp = id_tmp.variables[lon_var][:, :]
            elif len(id_tmp.variables[lat_var].shape) == 1:
                if config_options.aws:
                    lat_tmp = id_tmp.variables[lat_var][:].values
                    lon_tmp = id_tmp.variables[lon_var][:].values
                else:
                    lat_tmp = id_tmp.variables[lat_var][:]
                    lon_tmp = id_tmp.variables[lon_var][:]
                lon_tmp, lat_tmp = np.meshgrid(lon_tmp, lat_tmp)

    err_handler.check_program_status(config_options, mpi_config)

    var_tmp = lat_tmp if mpi_config.rank == 0 else None
    var_sub_lat_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    var_tmp = lon_tmp if mpi_config.rank == 0 else None
    var_sub_lon_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.esmf_lats = input_forcings.esmf_grid_in.get_coords(1)
    except ESMF.GridException as ge:
        config_options.errMsg = "Unable to locate latitude coordinate object within input ESMF grid: " + str(ge)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.esmf_lons = input_forcings.esmf_grid_in.get_coords(0)
    except ESMF.GridException as ge:
        config_options.errMsg = "Unable to locate longitude coordinate object within input ESMF grid: " + str(ge)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    input_forcings.esmf_lats[:, :] = var_sub_lat_tmp
    input_forcings.esmf_lons[:, :] = var_sub_lon_tmp
    del var_sub_lat_tmp, var_sub_lon_tmp, lat_tmp, lon_tmp

    if config_options.grid_type == "gridded":
        try:
            input_forcings.esmf_field_in = esmf_field_retry_partial(input_forcings.esmf_grid_in, name=input_forcings.product_name + "_NATIVE")
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = "Unable to create ESMF field object: " + str(esmf_error)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    if config_options.grid_type == "unstructured":
        try:
            input_forcings.esmf_field_in = esmf_field_retry_partial(input_forcings.esmf_grid_in, name=input_forcings.product_name + "_NATIVE")
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = "Unable to create ESMF field object: " + str(esmf_error)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.esmf_field_in_elem = esmf_field_retry_partial(input_forcings.esmf_grid_in, name=input_forcings.product_name + "_NATIVE_ELEMENT")
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = "Unable to create ESMF field object: " + str(esmf_error)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    elif config_options.grid_type == "hydrofabric":
        try:
            input_forcings.esmf_field_in = esmf_field_retry_partial(input_forcings.esmf_grid_in, name=input_forcings.product_name + "_NATIVE")
        except ESMF.ESMPyException as esmf_error:
            config_options.errMsg = "Unable to create ESMF field object: " + str(esmf_error)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        if config_options.aws:
            var_tmp = id_tmp[input_forcings.netcdf_var_names[force_count]].to_masked_array()
        else:
            var_tmp = id_tmp[input_forcings.netcdf_var_names[force_count]][0, :, :]
        var_tmp.fill(1)
        var_tmp = var_tmp.filled(0)

        if input_forcings.product_name == "NWM":
            var_tmp = np.asarray(var_tmp, dtype=np.float64)
    else:
        var_tmp = None
    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    if config_options.grid_type == "gridded":
        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp

    elif config_options.grid_type == "unstructured":
        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp

    elif config_options.grid_type == "hydrofabric":
        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp

    common_args = (mpi_config, config_options, input_forcings)
    weight_file, weight_file_elem = get_weight_file_names(*common_args)

    if input_forcings.regridObj is None:
        if config_options.weightsDir is not None:
            if not os.path.exists(weight_file):
                make_regrid(*common_args, weight_file=weight_file, fill=fill, element_mode=False)

            if config_options.grid_type == "unstructured" and (not os.path.exists(weight_file_elem)):
                make_regrid(*common_args, weight_file=weight_file_elem, fill=fill, element_mode=True)

            load_weight_file(*common_args, weight_file, element_mode=False)
            if config_options.grid_type == "unstructured":
                load_weight_file(*common_args, weight_file, element_mode=True)
        else:
            make_regrid(*common_args, weight_file=None, fill=fill, element_mode=False)
            if config_options.grid_type == "unstructured":
                make_regrid(*common_args, weight_file=None, fill=fill, element_mode=True)

    execute_regrid(*common_args, weight_file, element_mode=False)
    if config_options.grid_type == "gridded":
        input_forcings.regridded_mask[:, :] = np.round(input_forcings.esmf_field_out.data[:, :])
    elif config_options.grid_type != "gridded":
        input_forcings.regridded_mask[:] = np.round(input_forcings.esmf_field_out.data[:])

    if config_options.grid_type == "unstructured":
        execute_regrid(*common_args, weight_file, element_mode=True)
        input_forcings.regridded_mask_elem[:] = np.round(input_forcings.esmf_field_out_elem.data[:])

    err_handler.check_program_status(config_options, mpi_config)


def calculate_supp_pcp_weights(supplemental_precip, id_tmp, tmp_file, config_options, mpi_config, lat_var="latitude", lon_var="longitude"):
    """Calculate supplemental precip ESMF weights."""
    esmf_field_retry_partial = functools.partial(esmf_field_retry, mpi_config, config_options, err_handler)
    esmf_grid_retry_partial = functools.partial(esmf_grid_retry, mpi_config, config_options, err_handler)
    esmf_regrid_retry_partial = functools.partial(esmf_regrid_retry, mpi_config, config_options, err_handler)
    esmf_regridobj_call_retry_partial = functools.partial(esmf_regridobj_call_retry, mpi_config, config_options, err_handler)

    ndims = 0
    if mpi_config.rank == 0:
        ncvar = id_tmp.variables[supplemental_precip.netcdf_var_names[0]]
        ndims = len(ncvar.dimensions)
        if ndims == 3:
            latdim = 1
            londim = 2
        elif ndims == 2:
            latdim = 0
            londim = 1
        else:
            latdim = londim = -1
            config_options.errMsg = "Unable to determine lat/lon grid size from " + tmp_file
            err_handler.err_out(config_options)

        try:
            supplemental_precip.ny_global = id_tmp.variables[supplemental_precip.netcdf_var_names[0]].shape[latdim]
        except (ValueError, KeyError, AttributeError, Exception) as err:
            config_options.errMsg = "Unable to extract Y shape size from: " + supplemental_precip.netcdf_var_names[0] + " from: " + tmp_file + " (" + str(err) + ", " + type(err) + ")"
            err_handler.err_out(config_options)
        try:
            supplemental_precip.nx_global = id_tmp.variables[supplemental_precip.netcdf_var_names[0]].shape[londim]
        except (ValueError, KeyError, AttributeError, Exception) as err:
            config_options.errMsg = "Unable to extract X shape size from: " + supplemental_precip.netcdf_var_names[0] + " from: " + tmp_file + " (" + str(err) + ", " + type(err) + ")"
            err_handler.err_out(config_options)

    supplemental_precip.ny_global = mpi_config.broadcast_parameter(supplemental_precip.ny_global, config_options, param_type=int)
    supplemental_precip.nx_global = mpi_config.broadcast_parameter(supplemental_precip.nx_global, config_options, param_type=int)

    try:
        supplemental_precip.esmf_grid_in = esmf_grid_retry_partial(np.array([supplemental_precip.ny_global, supplemental_precip.nx_global]), staggerloc=ESMF.StaggerLoc.CENTER, coord_sys=ESMF.CoordSys.SPH_DEG)
        supplemental_precip.esmf_grid_in.add_item(ESMF.GridItem.MASK, ESMF.StaggerLoc.CENTER)

    except ESMF.ESMPyException as esmf_error:
        config_options.errMsg = "Unable to create source ESMF grid from temporary file: " + tmp_file + " (" + str(esmf_error) + ")"
        err_handler.err_out(config_options)

    try:
        supplemental_precip.x_lower_bound = supplemental_precip.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][1]
        supplemental_precip.x_upper_bound = supplemental_precip.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][1]
        supplemental_precip.y_lower_bound = supplemental_precip.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][0]
        supplemental_precip.y_upper_bound = supplemental_precip.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][0]
        supplemental_precip.nx_local = supplemental_precip.x_upper_bound - supplemental_precip.x_lower_bound
        supplemental_precip.ny_local = supplemental_precip.y_upper_bound - supplemental_precip.y_lower_bound
    except (ValueError, KeyError, AttributeError) as err:
        config_options.errMsg = "Unable to extract local X/Y boundaries from global grid from temporary file: " + tmp_file + " (" + str(err) + ")"
        err_handler.err_out(config_options)

    if supplemental_precip.nx_local < 2 or supplemental_precip.ny_local < 2:
        config_options.errMsg = "You have either specified too many cores for: " + supplemental_precip.product_name + ", or your input forcing grid is too small to process. Local grid must have x/y dimension size of 2."
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    lat_tmp = lon_tmp = None
    if mpi_config.rank == 0:
        if len(id_tmp.variables[lat_var].shape) == 3:
            lat_tmp = id_tmp.variables[lat_var][0, :]
            lon_tmp = id_tmp.variables[lon_var][0, :]
        elif len(id_tmp.variables[lon_var].shape) == 2:
            lat_tmp = id_tmp.variables[lat_var][:]
            lon_tmp = id_tmp.variables[lon_var][:]
        elif len(id_tmp.variables[lat_var].shape) == 1:
            lat_tmp = np.repeat(id_tmp.variables[lat_var][:][:, np.newaxis], supplemental_precip.nx_global, axis=1)
            lon_tmp = np.tile(id_tmp.variables[lon_var][:], (supplemental_precip.ny_global, 1))

    var_tmp = lat_tmp if mpi_config.rank == 0 else None
    var_sub_lat_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)

    var_tmp = lon_tmp if mpi_config.rank == 0 else None
    var_sub_lon_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)

    try:
        supplemental_precip.esmf_lats = supplemental_precip.esmf_grid_in.get_coords(1)
    except ESMF.GridException as ge:
        config_options.errMsg = "Unable to locate latitude coordinate object within supplemental precip ESMF grid: " + str(ge)
        err_handler.err_out(config_options)

    try:
        supplemental_precip.esmf_lons = supplemental_precip.esmf_grid_in.get_coords(0)
    except ESMF.GridException as ge:
        config_options.errMsg = "Unable to locate longitude coordinate object within supplemental precip ESMF grid: " + str(ge)
        err_handler.err_out(config_options)

    supplemental_precip.esmf_lats[:, :] = var_sub_lat_tmp
    supplemental_precip.esmf_lons[:, :] = var_sub_lon_tmp
    del var_sub_lat_tmp, var_sub_lon_tmp, lat_tmp, lon_tmp

    if config_options.grid_type == "gridded":
        supplemental_precip.esmf_field_in = esmf_field_retry_partial(supplemental_precip.esmf_grid_in, name=supplemental_precip.product_name + "_NATIVE")

        if mpi_config.rank == 0:
            if ndims == 3:
                var_tmp = id_tmp[supplemental_precip.netcdf_var_names[0]][0, :]
            elif ndims == 2:
                var_tmp = id_tmp[supplemental_precip.netcdf_var_names[0]][:]
            else:
                var_tmp = None
            var_tmp[:] = np.where(var_tmp == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue, 0.0, 1.0)
        else:
            var_tmp = None
        var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
        mpi_config.comm.barrier()

        supplemental_precip.esmf_field_in.data[:] = var_sub_tmp

        supplemental_precip.regridObj = esmf_regrid_retry_partial(supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out, src_mask_values=np.array([0]), regrid_method=ESMF.RegridMethod.BILINEAR, unmapped_action=ESMF.UnmappedAction.IGNORE)

        supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
        supplemental_precip.regridded_mask[:] = supplemental_precip.esmf_field_out.data[:]

    elif config_options.grid_type == "unstructured":
        supplemental_precip.esmf_field_in = esmf_field_retry_partial(supplemental_precip.esmf_grid_in, name=supplemental_precip.product_name + "_NATIVE")

        if mpi_config.rank == 0:
            if ndims == 3:
                var_tmp = id_tmp[supplemental_precip.netcdf_var_names[0]][0, :].data
            elif ndims == 2:
                var_tmp = id_tmp[supplemental_precip.netcdf_var_names[0]][:].data
            else:
                var_tmp = None
            var_tmp[:] = np.where(var_tmp == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue, 0.0, 1.0)
        else:
            var_tmp = None
        var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
        mpi_config.comm.barrier()

        supplemental_precip.esmf_field_in.data[:] = var_sub_tmp

        if supplemental_precip.regridOpt == 1:
            supplemental_precip.regridObj = esmf_regrid_retry_partial(supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out, src_mask_values=np.array([0]), regrid_method=ESMF.RegridMethod.BILINEAR, unmapped_action=ESMF.UnmappedAction.IGNORE)
        elif supplemental_precip.regridOpt == 2:
            supplemental_precip.regridObj = esmf_regrid_retry_partial(supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out, src_mask_values=np.array([0]), regrid_method=ESMF.RegridMethod.NEAREST_STOD, unmapped_action=ESMF.UnmappedAction.IGNORE)

        supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
        supplemental_precip.regridded_mask[:] = supplemental_precip.esmf_field_out.data[:]

        supplemental_precip.esmf_field_in_elem = esmf_field_retry_partial(supplemental_precip.esmf_grid_in, name=supplemental_precip.product_name + "_NATIVE")

        if mpi_config.rank == 0:
            if ndims == 3:
                var_tmp_elem = id_tmp[supplemental_precip.netcdf_var_names[0]][0, :].data
            elif ndims == 2:
                var_tmp_elem = id_tmp[supplemental_precip.netcdf_var_names[0]][:].data
            else:
                var_tmp_elem = None
            var_tmp[:] = np.where(var_tmp == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue, 0.0, 1.0)
        else:
            var_tmp_elem = None
        var_sub_tmp_elem = mpi_config.scatter_array(supplemental_precip, var_tmp_elem, config_options)
        mpi_config.comm.barrier()

        supplemental_precip.esmf_field_in_elem.data[:] = var_sub_tmp_elem

        supplemental_precip.regridObj_elem = esmf_regrid_retry_partial(supplemental_precip.esmf_field_in_elem, supplemental_precip.esmf_field_out_elem, src_mask_values=np.array([0]), regrid_method=ESMF.RegridMethod.BILINEAR, unmapped_action=ESMF.UnmappedAction.IGNORE)

        supplemental_precip.esmf_field_out_elem = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj_elem, supplemental_precip.esmf_field_in_elem, supplemental_precip.esmf_field_out_elem)
        supplemental_precip.regridded_mask_elem[:] = supplemental_precip.esmf_field_out_elem.data[:]

    elif config_options.grid_type == "hydrofabric":
        supplemental_precip.esmf_field_in = esmf_field_retry_partial(supplemental_precip.esmf_grid_in, name=supplemental_precip.product_name + "_NATIVE")

        if mpi_config.rank == 0:
            if ndims == 3:
                var_tmp = id_tmp[supplemental_precip.netcdf_var_names[0]][0, :]
            elif ndims == 2:
                var_tmp = id_tmp[supplemental_precip.netcdf_var_names[0]][:]
            else:
                var_tmp = None
            var_tmp[:] = np.where(var_tmp == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue, 0.0, 1.0)
            var_tmp[:] = np.where(var_tmp == 9.999e20, 0.0, 1.0)
        else:
            var_tmp = None
        var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
        mpi_config.comm.barrier()

        supplemental_precip.esmf_field_in.data[:] = var_sub_tmp

        if supplemental_precip.regridOpt == 2:
            supplemental_precip.regridObj = esmf_regrid_retry_partial(supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out, src_mask_values=np.array([0]), regrid_method=ESMF.RegridMethod.NEAREST_STOD, unmapped_action=ESMF.UnmappedAction.IGNORE, extrap_method=ESMF.ExtrapMethod.NEAREST_STOD)
        elif supplemental_precip.regridOpt == 1:
            supplemental_precip.regridObj = esmf_regrid_retry_partial(supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out, src_mask_values=np.array([0]), regrid_method=ESMF.RegridMethod.BILINEAR, unmapped_action=ESMF.UnmappedAction.IGNORE, extrap_method=ESMF.ExtrapMethod.NEAREST_STOD)

        supplemental_precip.esmf_field_out = esmf_regridobj_call_retry_partial(supplemental_precip.regridObj, supplemental_precip.esmf_field_in, supplemental_precip.esmf_field_out)
        supplemental_precip.regridded_mask[:] = supplemental_precip.esmf_field_out.data[:]
