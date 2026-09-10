"""Regridding module file for regridding input forcing files."""

from __future__ import annotations

import hashlib
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from time import monotonic, time
from typing import TYPE_CHECKING

# import mpi4py.util.pool as mpi_pool
# For ESMF + shapely 2.x, shapely must be imported first, to avoid segfault "address not mapped to object" stemming from calls such as:
# /usr/local/esmf/lib/libO/Linux.gfortran.64.openmpi.default/libesmf_fullylinked.so(get_geom+0x36)
import shapely

# from mpi4py.futures import MPIPoolExecutor
from mpi4py.futures import MPICommExecutor

from .. import os_utils

try:
    import esmpy as ESMF
except ImportError:
    import ESMF

import dask
import dask.delayed
import netCDF4 as nc
import numpy as np
import pandas as pd
from pyproj import Transformer

from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core import (
    err_handler,
    ioMod,
    timeInterpMod,
)

if TYPE_CHECKING:
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.config import (
        ConfigOptions,
    )
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.geoMod import (
        GeoMeta,
    )
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.suppPrecipMod import (
        supplemental_precip,
    )
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.forcingInputMod import (
        InputForcings,
    )
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.parallel import (
        MpiConfig,
    )
import logging
from ..esmf_utils import (
    esmf_field_retry,
    esmf_grid_retry,
    esmf_mesh_retry,
    esmf_regrid_retry,
    esmf_regridfromfile_retry,
    esmf_regridobj_call_retry,
)

LOG = logging.getLogger("FORCING")


if "WGRIB2" not in os.environ:
    WGRIB2_env = False
else:
    WGRIB2_env = True

NETCDF = "NETCDF"
GRIB2 = "GRIB2"

next_file_number = 0


class Partials:
    """A simple list of partials for common function / method calls."""

    def __init__(self, mpi_config: MpiConfig, config_options: ConfigOptions):
        args1 = (mpi_config, config_options, err_handler)
        args2 = (config_options, mpi_config)

        self.esmf_regridobj_call_retry_partial = partial(
            esmf_regridobj_call_retry, *args1
        )
        self.esmf_field_retry_partial = partial(esmf_field_retry, *args1)
        self.esmf_grid_retry_partial = partial(esmf_grid_retry, *args1)
        self.esmf_regrid_retry_partial = partial(esmf_regrid_retry, *args1)
        self.esmf_mesh_retry_partial = partial(esmf_mesh_retry, *args1)
        # NOTE these OS functions are collective and must be called by all ranks. The rank-specific behavior is handled within them.
        self.close_rank_0_partial = partial(os_utils.close_rank_0, *args1)
        self.close_anyrank_partial = partial(os_utils.close, *args1)
        self.os_remove_rank_0_partial = partial(os_utils.os_remove_rank_0, *args1)
        self.os_remove_anyrank_partial = partial(os_utils.os_remove, *args1)
        # NOTE need to use positional arg for param `debug` here, to allow for positional arg `msg` afterwards.
        self.log_debug = partial(err_handler.log_msg, *args2, True)
        self.log_info = partial(err_handler.log_msg, *args2, False)
        self.log_warn = partial(err_handler.log_warning, *args2)
        self.log_err = partial(err_handler.log_error, *args2)
        self.log_crit = partial(err_handler.log_critical, *args2)


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
    pt = Partials(mpi_config, config_options)

    if mpi_config.rank == 0:
        try:
            pt.log_debug(f"{name} file being used: {input_file}")

            os.symlink(input_file, tmpFile)
        except:
            pt.log_crit(f"Unable to create link: {input_file} to: {tmpFile}")
    err_handler.check_program_status(config_options, mpi_config)


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
    pt = Partials(mpi_config, config_options)
    ds = None

    try:
        # If the expected file is missing, this means we are allowing missing files, simply
        # exit out of this routine as the regridded fields have already been set to NDV.
        if not os.path.isfile(input_forcings.file_in2):
            if mpi_config.rank == 0:
                pt.log_debug("No AK AnA in_2 file found for this timestep.")
            return

        # Check to see if the regrid complete flag for this
        # output time step is true. This entails the necessary
        # inputs have already been regridded and we can move on.
        if input_forcings.regridComplete:
            if mpi_config.rank == 0:
                pt.log_debug("No AK AnA regridding required for this timestep.")
            return

        # Only rank‐0 opens the file
        if mpi_config.rank == 0:
            from netCDF4 import Dataset

            try:
                ds = Dataset(input_forcings.file_in2, "r")
            except Exception as e:
                pt.log_crit(
                    f"Unable to open input NetCDF file: {input_forcings.file_in2} ({e})"
                )

            # turn off automatic masking but keep scaling
            ds.set_auto_scale(True)
            ds.set_auto_mask(False)

        if input_forcings.nx_global is None or input_forcings.ny_global is None:
            # This is the first timestep.
            if mpi_config.rank == 0:
                input_forcings.ny_global = ds.dimensions["y"].size
                input_forcings.nx_global = ds.dimensions["x"].size

            input_forcings.ny_global = mpi_config.broadcast_parameter(
                input_forcings.ny_global, config_options, param_type=int
            )
            err_handler.check_program_status(config_options, mpi_config)
            input_forcings.nx_global = mpi_config.broadcast_parameter(
                input_forcings.nx_global, config_options, param_type=int
            )
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                try:
                    # noinspection PyTypeChecker
                    input_forcings.esmf_grid_in = pt.esmf_grid_retry_partial(
                        np.array([input_forcings.ny_global, input_forcings.nx_global]),
                        staggerloc=ESMF.StaggerLoc.CENTER,
                        coord_sys=ESMF.CoordSys.SPH_DEG,
                    )
                except ESMF.ESMPyException as esmf_error:
                    pt.log_crit(
                        f"Unable to create source ESMF grid from netCDF file: {input_forcings.file_in} ({str(esmf_error)})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.x_lower_bound = (
                        input_forcings.esmf_grid_in.lower_bounds[
                            ESMF.StaggerLoc.CENTER
                        ][1]
                    )
                    input_forcings.x_upper_bound = (
                        input_forcings.esmf_grid_in.upper_bounds[
                            ESMF.StaggerLoc.CENTER
                        ][1]
                    )
                    input_forcings.y_lower_bound = (
                        input_forcings.esmf_grid_in.lower_bounds[
                            ESMF.StaggerLoc.CENTER
                        ][0]
                    )
                    input_forcings.y_upper_bound = (
                        input_forcings.esmf_grid_in.upper_bounds[
                            ESMF.StaggerLoc.CENTER
                        ][0]
                    )
                    input_forcings.nx_local = (
                        input_forcings.x_upper_bound - input_forcings.x_lower_bound
                    )
                    input_forcings.ny_local = (
                        input_forcings.y_upper_bound - input_forcings.y_lower_bound
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract local X/Y boundaries from global grid from netCDF file: {input_forcings.file_in} ({str(err)})"
                    )
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "unstructured":
                try:
                    input_forcings.esmf_grid_in = pt.esmf_mesh_retry_partial(
                        filename=config_options.geogrid,
                        filetype=ESMF.FileFormat.ESMFMESH,
                    )
                    input_forcings.esmf_grid_in_elem = pt.esmf_mesh_retry_partial(
                        filename=config_options.geogrid,
                        filetype=ESMF.FileFormat.ESMFMESH,
                    )
                except ESMF.ESMPyException as esmf_error:
                    pt.log_crit(
                        f"Unable to create source ESMF Mesh from netCDF file: {input_forcings.file_in} ({str(esmf_error)})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.nx_local = len(
                        input_forcings.esmf_grid_in.esmf_grid_in.coords[0][1]
                    )
                    input_forcings.ny_local = len(
                        input_forcings.esmf_grid_in.esmf_grid_in.coords[0][1]
                    )
                    input_forcings.nx_local_elem = len(
                        input_forcings.esmf_grid_in_elem.esmf_grid_in.coords[0][1]
                    )
                    input_forcings.ny_local_elem = len(
                        input_forcings.esmf_grid_in_elem.esmf_grid_in.coords[0][1]
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract local X/Y boundaries from global mesh file: {input_forcings.file_in} ({str(err)})"
                    )
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "hydrofabric":
                try:
                    input_forcings.esmf_grid_in = pt.esmf_mesh_retry_partial(
                        filename=config_options.geogrid,
                        filetype=ESMF.FileFormat.ESMFMESH,
                    )
                except ESMF.ESMPyException as esmf_error:
                    pt.log_crit(
                        f"Unable to create source ESMF Mesh from netCDF file: {input_forcings.file_in} ({str(esmf_error)})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.nx_local = len(
                        input_forcings.esmf_grid_in.esmf_grid_in.coords[0][1]
                    )
                    input_forcings.ny_local = len(
                        input_forcings.esmf_grid_in.esmf_grid_in.coords[0][1]
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract local X/Y boundaries from global mesh file: {input_forcings.file_in} ({str(err)})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

            # Create out regridded numpy arrays to hold the regridded data.
            if config_options.grid_type == "gridded":
                input_forcings.regridded_forcings1 = np.empty(
                    [9, wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local],
                    np.float32,
                )
                input_forcings.regridded_forcings2 = np.empty(
                    [9, wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local],
                    np.float32,
                )
            elif config_options.grid_type == "unstructured":
                input_forcings.regridded_forcings1 = np.empty(
                    [9, wrf_hydro_geo_meta.ny_local], np.float32
                )
                input_forcings.regridded_forcings2 = np.empty(
                    [9, wrf_hydro_geo_meta.ny_local], np.float32
                )
                input_forcings.regridded_forcings1_elem = np.empty(
                    [9, wrf_hydro_geo_meta.ny_local_elem], np.float32
                )
                input_forcings.regridded_forcings2_elem = np.empty(
                    [9, wrf_hydro_geo_meta.ny_local_elem], np.float32
                )
            elif config_options.grid_type == "unstructured":
                input_forcings.regridded_forcings1 = np.empty(
                    [9, wrf_hydro_geo_meta.ny_local], np.float32
                )
                input_forcings.regridded_forcings2 = np.empty(
                    [9, wrf_hydro_geo_meta.ny_local], np.float32
                )

        for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
            var_tmp = None
            var_tmp_elem = None
            if mpi_config.rank == 0:
                pt.log_debug(
                    f"Processing input AK AnA variable: {nc_var} from {input_forcings.file_in2}"
                )
                LOG.debug(f"{config_options.statusMsg}")
                if config_options.grid_type == "gridded":
                    try:
                        var_tmp = ds.variables[nc_var][0, :, :]
                        var_tmp = np.float32(var_tmp)

                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {nc_var} from: {input_forcings.file_in2} ({str(err)})"
                        )
                elif config_options.grid_type == "unstructured":
                    try:
                        var_tmp = ds.variables[nc_var][0, :]
                        var_tmp = np.float32(var_tmp)

                        var_tmp_elem = ds.variables[nc_var][0, :]
                        var_tmp_elem = np.float32(var_tmp_elem)
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {nc_var} from: {input_forcings.file_in2} ({str(err)})"
                        )
                elif config_options.grid_type == "hydrofabric":
                    try:
                        var_tmp = ds.variables[nc_var][0, :]
                        var_tmp = np.float32(var_tmp)

                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {nc_var} from: {input_forcings.file_in2} ({str(err)})"
                        )

            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)
                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract ExtAnA forcing data from the AK AnA field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)
                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ]
            elif config_options.grid_type == "unstructured":
                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = var_tmp[wrf_hydro_geo_meta.mesh_inds]
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :, :
                    ] = var_tmp[wrf_hydro_geo_meta.mesh_inds_elem]
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract ExtAnA forcing data from the AK AnA mesh field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)
                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                    input_forcings.regridded_forcings1_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ]
            elif config_options.grid_type == "hydrofabric":
                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = var_tmp[wrf_hydro_geo_meta.mesh_inds]
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract ExtAnA forcing data from the AK AnA mesh field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)
                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
    finally:
        if mpi_config.rank == 0 and ds is not None:
            try:
                ds.close()
            except OSError:
                pt.log_crit(
                    f"Unable to close input NetCDF file: {input_forcings.file_in2}"
                )
        err_handler.check_program_status(config_options, mpi_config)


def _regrid_ak_ext_ana_pcp_stage4(
    supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Read in and regrid Alaska ExtAna supplemental Stage IV precip data.

    Function for handling regridding of Alaska ExtAna supplemental Stage IV precip data.
    :param supplemental_precip:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.exists(supplemental_precip.file_in1):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if supplemental_precip.regridComplete:
        if mpi_config.rank == 0:
            pt.log_debug("No StageIV regridding required for this timestep.")
        return

    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.

    file_name = f"STAGEIV_AK_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    stage4_tmp_nc = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    lat_var = "latitude"
    lon_var = "longitude"

    id_tmp = None
    try:
        if supplemental_precip.file_type != NETCDF:
            pt.os_remove_rank_0_partial(stage4_tmp_nc)

            # Create a temporary NetCDF file from the GRIB2 file.
            if WGRIB2_env:
                cmd = f'$WGRIB2 -match "APCP:surface:0-6 hour acc fcst" {supplemental_precip.file_in2} -netcdf {stage4_tmp_nc}'
            else:
                cmd = "APCP:surface:0-6 hour acc fcst"

            if mpi_config.rank == 0:
                pt.log_debug(f"WGRIB2 command: {cmd}")
            id_tmp = ioMod.open_grib2(
                supplemental_precip.file_in2,
                stage4_tmp_nc,
                cmd,
                config_options,
                mpi_config,
                inputVar=None,
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link(
                "STAGEIV-PCP",
                supplemental_precip.file_in2,
                stage4_tmp_nc,
                config_options,
                mpi_config,
            )
            id_tmp = ioMod.open_netcdf_forcing(
                stage4_tmp_nc, config_options, mpi_config, False, lat_var, lon_var
            )

        # Check to see if we need to calculate regridding weights.
        calc_regrid_flag = check_supp_pcp_regrid_status(
            id_tmp, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
        )
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                pt.log_debug("Calculating STAGE IV regridding weights.")
            calculate_supp_pcp_weights(
                supplemental_precip,
                id_tmp,
                stage4_tmp_nc,
                config_options,
                mpi_config,
                lat_var,
                lon_var,
            )
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    )
                try:
                    var_tmp = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :]
                    var_tmp = np.where(
                        var_tmp
                        == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                        0.0,
                        var_tmp,
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract precipitation from STAGE IV file: {supplemental_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)
        elif config_options.grid_type == "unstructured":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    )
                try:
                    var_tmp = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :]
                    var_tmp = np.where(
                        var_tmp
                        == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                        0.0,
                        var_tmp,
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract precipitation from STAGE IV file: {supplemental_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the input variables.
            var_tmp_elem = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    )
                try:
                    var_tmp_elem = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :]
                    var_tmp = np.where(
                        var_tmp
                        == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                        0.0,
                        var_tmp,
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract precipitation from STAGE IV file: {supplemental_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(
                supplemental_precip, var_tmp_elem, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    )
                try:
                    var_tmp = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :]
                    var_tmp = np.where(
                        var_tmp
                        == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                        0.0,
                        var_tmp,
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract precipitation from STAGE IV file: {supplemental_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place STAGE IV precipitation into local ESMF field: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj,
                        supplemental_precip.esmf_field_in,
                        supplemental_precip.esmf_field_out,
                    )
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid STAGE IV supplemental precipitation: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:, :] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the 6-hourly precipitation total to a rate of mm/s
            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2 != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2[ind_valid] = (
                    supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                pt.log_crit(
                    f"Unable to run NDV search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:, :] = (
                    supplemental_precip.regridded_precip2[:, :]
                )
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place STAGE IV precipitation into local ESMF field: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj,
                        supplemental_precip.esmf_field_in,
                        supplemental_precip.esmf_field_out,
                    )
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid STAGE IV supplemental precipitation: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the 6-hourly precipitation total to a rate of mm/s
            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2 != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2[ind_valid] = (
                    supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                pt.log_crit(
                    f"Unable to run NDV search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = (
                    supplemental_precip.regridded_precip2[:]
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place STAGE IV precipitation into local ESMF field: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj_elem,
                        supplemental_precip.esmf_field_in_elem,
                        supplemental_precip.esmf_field_out_elem,
                    )
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid STAGE IV supplemental precipitation: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                supplemental_precip.esmf_field_out_elem.data[
                    np.where(supplemental_precip.regridded_mask_elem == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2_elem[:] = (
                supplemental_precip.esmf_field_out_elem.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the 6-hourly precipitation total to a rate of mm/s
            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2_elem
                    != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2_elem[ind_valid] = (
                    supplemental_precip.regridded_precip2_elem[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                pt.log_crit(
                    f"Unable to run NDV search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1_elem[:] = (
                    supplemental_precip.regridded_precip2_elem[:]
                )
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place STAGE IV precipitation into local ESMF field: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj,
                        supplemental_precip.esmf_field_in,
                        supplemental_precip.esmf_field_out,
                    )
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid STAGE IV supplemental precipitation: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the 6-hourly precipitation total to a rate of mm/s
            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2 != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2[ind_valid] = (
                    supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                pt.log_crit(
                    f"Unable to run NDV search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = (
                    supplemental_precip.regridded_precip2[:]
                )
            err_handler.check_program_status(config_options, mpi_config)

    finally:
        pt.close_rank_0_partial(id_tmp)
        pt.os_remove_rank_0_partial(stage4_tmp_nc)


def regrid_ak_ext_ana_pcp(
    supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Read in and regrid Alaska ExtAna supplemental precip data.

    Function for handling regridding of Alaska ExtAna supplemental precip data.
    :param supplemental_precip:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    if supplemental_precip.ext_ana == "STAGE4":
        supplemental_precip.netcdf_var_names.append("APCP_surface")
        # supplemental_precip.netcdf_var_names.append('A_PCP_GDS5_SFC_acc6h')
        _regrid_ak_ext_ana_pcp_stage4(
            supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
        )
        supplemental_precip.netcdf_var_names.pop()
    else:  # MRMS
        supplemental_precip.netcdf_var_names.append(
            "MultiSensorQPE01H_0mabovemeansealevel"
        )
        # supplemental_precip.netcdf_var_names.append('A_PCP_GDS5_SFC_acc6h')
        regrid_mrms_hourly(
            supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
        )
        supplemental_precip.netcdf_var_names.pop()


def _regrid_conus_ext_ana_pcp_stage4(
    supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Read in and regrid Alaska ExtAna supplemental Stage IV precip data.

    Function for handling regridding of Alaska ExtAna supplemental Stage IV precip data.
    :param supplemental_precip:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.exists(supplemental_precip.file_in1):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if supplemental_precip.regridComplete:
        if mpi_config.rank == 0:
            pt.log_debug("No StageIV regridding required for this timestep.")
        return

    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    file_name = f"STAGEIV_CONUS_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    stage4_tmp_nc = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    lat_var = "latitude"
    lon_var = "longitude"

    id_tmp = None
    try:
        if supplemental_precip.file_type != NETCDF:
            pt.os_remove_rank_0_partial(stage4_tmp_nc)

            # Create a temporary NetCDF file from the GRIB2 file.
            if WGRIB2_env:
                cmd = f'$WGRIB2 -match "APCP:surface:0-1 hour acc fcst" {supplemental_precip.file_in2} -netcdf {stage4_tmp_nc}'
            else:
                cmd = "APCP:surface:0-1 hour acc fcst"

            if mpi_config.rank == 0:
                pt.log_debug(f"WGRIB2 command: {cmd}")
            id_tmp = ioMod.open_grib2(
                supplemental_precip.file_in2,
                stage4_tmp_nc,
                cmd,
                config_options,
                mpi_config,
                inputVar=None,
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link(
                "STAGEIV-PCP",
                supplemental_precip.file_in2,
                stage4_tmp_nc,
                config_options,
                mpi_config,
            )
            id_tmp = ioMod.open_netcdf_forcing(
                stage4_tmp_nc, config_options, mpi_config, False, lat_var, lon_var
            )

        # Check to see if we need to calculate regridding weights.
        calc_regrid_flag = check_supp_pcp_regrid_status(
            id_tmp, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
        )
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                pt.log_debug("Calculating STAGE IV regridding weights.")
            calculate_supp_pcp_weights(
                supplemental_precip,
                id_tmp,
                stage4_tmp_nc,
                config_options,
                mpi_config,
                lat_var,
                lon_var,
            )
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    )
                try:
                    var_tmp = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :]
                    var_tmp = np.where(
                        var_tmp
                        == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                        0.0,
                        var_tmp,
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract precipitation from STAGE IV file: {supplemental_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)
        elif config_options.grid_type == "unstructured":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    )
                try:
                    var_tmp = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :].data
                    var_tmp = np.where(
                        var_tmp
                        == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                        0.0,
                        var_tmp,
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract precipitation from STAGE IV file: {supplemental_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the input variables.
            var_tmp_elem = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    )
                try:
                    var_tmp_elem = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :].data
                    var_tmp_elem = np.where(
                        var_tmp_elem
                        == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                        0.0,
                        var_tmp_elem,
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract precipitation from STAGE IV file: {supplemental_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(
                supplemental_precip, var_tmp_elem, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding STAGE IV '{supplemental_precip.netcdf_var_names[-1]}' Precipitation."
                    )
                try:
                    var_tmp = id_tmp.variables[
                        supplemental_precip.netcdf_var_names[-1]
                    ][0, :, :]
                    var_tmp = np.where(
                        var_tmp
                        == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                        0.0,
                        var_tmp,
                    )
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract precipitation from STAGE IV file: {supplemental_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place STAGE IV precipitation into local ESMF field: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj,
                        supplemental_precip.esmf_field_in,
                        supplemental_precip.esmf_field_out,
                    )
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid STAGE IV supplemental precipitation: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:, :] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the 6-hourly precipitation total to a rate of mm/s
            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2 != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2[ind_valid] = (
                    supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                pt.log_crit(
                    f"Unable to run NDV search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:, :] = (
                    supplemental_precip.regridded_precip2[:, :]
                )
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place STAGE IV precipitation into local ESMF field: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj,
                        supplemental_precip.esmf_field_in,
                        supplemental_precip.esmf_field_out,
                    )
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid STAGE IV supplemental precipitation: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the 6-hourly precipitation total to a rate of mm/s
            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2 != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2[ind_valid] = (
                    supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                pt.log_crit(
                    f"Unable to run NDV search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = (
                    supplemental_precip.regridded_precip2[:]
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place STAGE IV precipitation into local ESMF field: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj_elem,
                        supplemental_precip.esmf_field_in_elem,
                        supplemental_precip.esmf_field_out_elem,
                    )
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid STAGE IV supplemental precipitation: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                supplemental_precip.esmf_field_out_elem.data[
                    np.where(supplemental_precip.regridded_mask_elem == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2_elem[:] = (
                supplemental_precip.esmf_field_out_elem.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the 6-hourly precipitation total to a rate of mm/s
            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2_elem
                    != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2_elem[ind_valid] = (
                    supplemental_precip.regridded_precip2_elem[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                pt.log_crit(
                    f"Unable to run NDV search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1_elem[:] = (
                    supplemental_precip.regridded_precip2_elem[:]
                )
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place STAGE IV precipitation into local ESMF field: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj,
                        supplemental_precip.esmf_field_in,
                        supplemental_precip.esmf_field_out,
                    )
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid STAGE IV supplemental precipitation: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert fill value to globalNdv. Convert 1 hr precip to mm/s
            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2 != 9.999e20
                )  # config_options.globalNdv)
                supplemental_precip.regridded_precip2[ind_valid] = (
                    supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                )
                invalid = np.where(supplemental_precip.regridded_precip2 == 9.999e20)
                supplemental_precip.regridded_precip2[invalid] = (
                    config_options.globalNdv
                )
                del ind_valid
                del invalid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                pt.log_crit(
                    f"Unable to run NDV search on STAGE IV supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)
            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = (
                    supplemental_precip.regridded_precip2[:]
                )
            err_handler.check_program_status(config_options, mpi_config)

    finally:
        pt.close_rank_0_partial(id_tmp)
        pt.os_remove_rank_0_partial(stage4_tmp_nc)
    err_handler.check_program_status(config_options, mpi_config)


def regrid_conus_ext_ana_pcp(
    supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Read in and regrid CONUS ExtAna supplemental precip data.

    Function for handling regridding of CONUS  ExtAna supplemental precip data.
    :param supplemental_precip:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    if supplemental_precip.ext_ana == "STAGE4":
        supplemental_precip.netcdf_var_names.append("APCP_surface")
        # supplemental_precip.netcdf_var_names.append('A_PCP_GDS5_SFC_acc6h')
        _regrid_conus_ext_ana_pcp_stage4(
            supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
        )
        supplemental_precip.netcdf_var_names.pop()
    else:  # MRMS
        supplemental_precip.netcdf_var_names.append(
            "MultiSensorQPE01H_0mabovemeansealevel"
        )
        # supplemental_precip.netcdf_var_names.append('A_PCP_GDS5_SFC_acc6h')
        regrid_mrms_hourly(
            supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
        )
        supplemental_precip.netcdf_var_names.pop()


def regrid_conus_hrrr(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid CONUS HRRR data.

    Function for handling regridding of HRRR data.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        if mpi_config.rank == 0:
            pt.log_debug("No HRRR in_2 file found for this timestep.")
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            pt.log_debug("No HRRR regridding required for this timestep.")
        return

    # Create a path for a temporary NetCDF file
    file_name = f"HRRR_CONUS_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    input_forcings.tmpFile = str(
        Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}"
    )

    id_tmp = None
    try:
        pt.log_info("Regrid CONUS HRRR")

        if input_forcings.input_force_types != NETCDF:
            pt.os_remove_rank_0_partial(input_forcings.tmpFile)

            # Build GRIB2 to NetCDF conversion
            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    pt.log_debug(f"Converting HRRR Variable: {grib_var}")
                if 0 < input_forcings.cycle_freq < 60:
                    time_str = (
                        f"{input_forcings.fcst_min1}-{input_forcings.fcst_min2} min acc fcst"
                        if grib_var == "APCP"
                        else f"{input_forcings.fcst_min2} min fcst"
                    )
                    sub_rem = int(input_forcings.fcst_min1) % 60
                    sub_id = int(sub_rem / input_forcings.cycle_freq)
                else:
                    time_str = (
                        f"{input_forcings.fcst_hour1}-{input_forcings.fcst_hour2} hour acc fcst"
                        if grib_var == "APCP"
                        else f"{input_forcings.fcst_hour2} hour fcst"
                    )
                fields.append(
                    f":{grib_var}:{input_forcings.grib_levels[force_count]}:{time_str}:"
                )
            fields.append(":(HGT):(surface):")

            # Create a temporary NetCDF file from the GRIB2 file.
            if WGRIB2_env:
                pattern = "|".join(fields)
                cmd = f'$WGRIB2 -match "({pattern})" {input_forcings.file_in2} -netcdf {input_forcings.tmpFile}'
            else:
                cmd = f"({'|'.join(fields)})"

            id_tmp = ioMod.open_grib2(
                input_forcings.file_in2,
                input_forcings.tmpFile,
                cmd,
                config_options,
                mpi_config,
                inputVar=None,
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link(
                "HRRR",
                input_forcings.file_in2,
                input_forcings.tmpFile,
                config_options,
                mpi_config,
            )
            id_tmp = ioMod.open_netcdf_forcing(
                input_forcings.tmpFile, config_options, mpi_config
            )

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                pt.log_debug(f"Processing HRRR Variable: {grib_var}")

            calc_regrid_flag = check_regrid_status(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                wrf_hydro_geo_meta,
                mpi_config,
            )
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    pt.log_debug("Calculating HRRR regridding weights.")
                calculate_weights(
                    id_tmp,
                    force_count,
                    input_forcings,
                    config_options,
                    mpi_config,
                    wrf_hydro_geo_meta,
                )
                err_handler.check_program_status(config_options, mpi_config)

                # # Read in the HRRR height field, which is used for downscaling purposes.
                # if mpi_config.rank == 0:
                #     config_options.statusMsg = "Reading in HRRR elevation data."
                #     err_handler.log_msg(config_options, mpi_config, True)
                # cmd = "$WGRIB2 " + input_forcings.file_in2 + " -match " + \
                #       "\":(HGT):(surface):\" " + \
                #       " -netcdf " + input_forcings.tmpFileHeight
                # id_tmp_height = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFileHeight,
                #                                  cmd, config_options, mpi_config, 'HGT_surface')
                # err_handler.check_program_status(config_options, mpi_config)

                if config_options.grid_type == "gridded":
                    # Regrid the height variable.
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            if 0 < input_forcings.cycle_freq < 60:
                                var_tmp = id_tmp.variables["HGT_surface"][sub_id]
                            else:
                                var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = f"Unable to extract HRRR elevation from {input_forcings.tmpFile}: {err}"

                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place input NetCDF HRRR data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding HRRR surface elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid HRRR surface elevation using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to perform HRRR mask search on elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract regridded HRRR elevation data from ESMF: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "unstructured":
                    # Regrid the height variable.
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            if 0 < input_forcings.cycle_freq < 60:
                                var_tmp = id_tmp.variables["HGT_surface"][sub_id]
                            else:
                                var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = f"Unable to extract HRRR elevation from {input_forcings.tmpFile}: {err}"

                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place input NetCDF HRRR data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding HRRR surface elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid HRRR surface elevation using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to perform HRRR mask search on elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract regridded HRRR elevation data from ESMF: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Regrid the height variable.
                    var_tmp_elem = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = f"Unable to extract HRRR elevation from {input_forcings.tmpFile}: {err}"

                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(
                        input_forcings, var_tmp_elem, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place input NetCDF HRRR data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding HRRR surface elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out_elem = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj_elem,
                                input_forcings.esmf_field_in_elem,
                                input_forcings.esmf_field_out_elem,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid HRRR surface elevation using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out_elem.data[
                            np.where(input_forcings.regridded_mask_elem == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to perform HRRR mask search on elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = (
                            input_forcings.esmf_field_out_elem.data
                        )
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract regridded HRRR elevation data from ESMF: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    # Regrid the height variable.
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            if 0 < input_forcings.cycle_freq < 60:
                                var_tmp = id_tmp.variables["HGT_surface"][sub_id]
                            else:
                                var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            config_options.errMsg = f"Unable to extract HRRR elevation from {input_forcings.tmpFile}: {err}"

                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place input NetCDF HRRR data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding HRRR surface elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid HRRR surface elevation using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to perform HRRR mask search on elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract regridded HRRR elevation data from ESMF: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                # Close the temporary NetCDF file and remove it.
                # if mpi_config.rank == 0:
                #     try:
                #         id_tmp_height.close()
                #     except OSError:
                #         config_options.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                #         err_handler.log_critical(config_options, mpi_config)
                #
                #     try:
                #         os_utils.os_remove_retry(input_forcings.tmpFileHeight)
                #     except OSError:
                #         config_options.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                #         err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                # Regrid the input variables.
                var_tmp = None
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Processing input HRRR variable: {input_forcings.netcdf_var_names[force_count]}"
                    )
                    try:
                        if 0 < input_forcings.cycle_freq < 60:
                            var_tmp = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][sub_id, :, :]
                        else:
                            var_tmp = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][0, :, :]
                        if grib_var == "APCP":
                            var_tmp /= 3600  # convert hourly accumulated precip to instantaneous rate
                        if grib_var == "CPOFP":
                            var_tmp[var_tmp >= 0] = (
                                100 - var_tmp[var_tmp >= 0]
                            ) / 100  # convert frozen fraction to liquid fraction
                            var_tmp[var_tmp < 0] = (
                                1.0  # assume all liquid if not specifically given
                            )
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place input HRRR data into ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding Input HRRR Field: {input_forcings.netcdf_var_names[force_count]}"
                    )
                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(f"Unable to regrid input HRRR forcing data: {ve}")
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to perform mask test on regridded HRRR forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract regridded HRRR forcing data from the ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ]
                # mpi_config.comm.barrier()

            elif config_options.grid_type == "unstructured":
                # Regrid the input variables.
                var_tmp = None
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Processing input HRRR variable: {input_forcings.netcdf_var_names[force_count]}"
                    )
                    try:
                        if 0 < input_forcings.cycle_freq < 60:
                            var_tmp = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][sub_id, :, :]
                        else:
                            var_tmp = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][0, :, :]
                        if grib_var == "APCP":
                            var_tmp /= 3600  # convert hourly accumulated precip to instantaneous rate
                        if grib_var == "CPOFP":
                            var_tmp[var_tmp >= 0] = (
                                100 - var_tmp[var_tmp >= 0]
                            ) / 100  # convert frozen fraction to liquid fraction
                            var_tmp[var_tmp < 0] = (
                                1.0  # assume all liquid if not specifically given
                            )
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place input HRRR data into ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding Input HRRR Field: {input_forcings.netcdf_var_names[force_count]}"
                    )
                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(f"Unable to regrid input HRRR forcing data: {ve}")
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to perform mask test on regridded HRRR forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract regridded HRRR forcing data from the ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                # mpi_config.comm.barrier()

                # Regrid the input variables.
                var_tmp_elem = None
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Processing input HRRR variable: {input_forcings.netcdf_var_names[force_count]}"
                    )
                    try:
                        if 0 < input_forcings.cycle_freq < 60:
                            var_tmp_elem = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][sub_id, :, :]
                        else:
                            var_tmp_elem = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][0, :, :]
                        if grib_var == "APCP":
                            var_tmp_elem /= 3600  # convert hourly accumulated precip to instantaneous rate
                        if grib_var == "CPOFP":
                            var_tmp_elem[var_tmp_elem >= 0] = (
                                100 - var_tmp_elem[var_tmp_elem >= 0]
                            ) / 100  # convert frozen fraction to liquid fraction
                            var_tmp_elem[var_tmp_elem < 0] = (
                                1.0  # assume all liquid if not specifically given
                            )
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(
                    input_forcings, var_tmp_elem, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place input HRRR data into ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding Input HRRR Field: {input_forcings.netcdf_var_names[force_count]}"
                    )
                try:
                    input_forcings.esmf_field_out_elem = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj_elem,
                            input_forcings.esmf_field_in_elem,
                            input_forcings.esmf_field_out_elem,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(f"Unable to regrid input HRRR forcing data: {ve}")
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out_elem.data[
                        np.where(input_forcings.regridded_mask_elem == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to perform mask test on regridded HRRR forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract regridded HRRR forcing data from the ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ]
                # mpi_config.comm.barrier()

            elif config_options.grid_type == "hydrofabric":
                # Regrid the input variables.
                var_tmp = None
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Processing input HRRR variable: {input_forcings.netcdf_var_names[force_count]}"
                    )
                    try:
                        if 0 < input_forcings.cycle_freq < 60:
                            var_tmp = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][sub_id, :, :]
                        else:
                            var_tmp = id_tmp.variables[
                                input_forcings.netcdf_var_names[force_count]
                            ][0, :, :]
                        if grib_var == "APCP":
                            var_tmp /= 3600  # convert hourly accumulated precip to instantaneous rate
                        if grib_var == "CPOFP":
                            var_tmp[var_tmp >= 0] = (
                                100 - var_tmp[var_tmp >= 0]
                            ) / 100  # convert frozen fraction to liquid fraction
                            var_tmp[var_tmp < 0] = (
                                1.0  # assume all liquid if not specifically given
                            )
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place input HRRR data into ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding Input HRRR Field: {input_forcings.netcdf_var_names[force_count]}"
                    )
                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(f"Unable to regrid input HRRR forcing data: {ve}")
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to perform mask test on regridded HRRR forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract regridded HRRR forcing data from the ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                # mpi_config.comm.barrier()

    finally:
        pt.close_rank_0_partial(id_tmp)
        pt.os_remove_rank_0_partial(input_forcings.tmpFile)


def regrid_conus_rap(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid CONUS RAP 13km data.

    Function for handling regridding of RAP 13km conus data.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            pt.log_debug("No RAP regridding required for this timestep.")
        return

    # Create a path for a temporary NetCDF file
    file_name = f"RAP_CONUS_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    input_forcings.tmpFile = str(
        Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}"
    )

    err_handler.check_program_status(config_options, mpi_config)

    id_tmp = None
    try:
        pt.log_info("Regrid CONUS RAP")
        if input_forcings.input_force_types != NETCDF:
            pt.os_remove_rank_0_partial(input_forcings.tmpFile)

            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    pt.log_debug(f"Converting CONUS RAP Variable: {grib_var}")
                time_str = (
                    f"{input_forcings.fcst_hour1}-{input_forcings.fcst_hour2} hour acc fcst"
                    if grib_var in ("APCP", "FROZR")
                    else f"{input_forcings.fcst_hour2} hour fcst"
                )
                fields.append(
                    f":{grib_var}:{input_forcings.grib_levels[force_count]}:{time_str}:"
                )
            fields.append(":(HGT):(surface):")

            # Create a temporary NetCDF file from the GRIB2 file.
            if WGRIB2_env:
                cmd = f'$WGRIB2 -match "({"|".join(fields)})" {input_forcings.file_in2} -netcdf {input_forcings.tmpFile}'
            else:
                cmd = f"({'|'.join(fields)})"

            id_tmp = ioMod.open_grib2(
                input_forcings.file_in2,
                input_forcings.tmpFile,
                cmd,
                config_options,
                mpi_config,
                inputVar=None,
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link(
                "RAP",
                input_forcings.file_in2,
                input_forcings.tmpFile,
                config_options,
                mpi_config,
            )
            id_tmp = ioMod.open_netcdf_forcing(
                input_forcings.tmpFile, config_options, mpi_config
            )

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                pt.log_debug(f"Processing Conus RAP Variable: {grib_var}")

            calc_regrid_flag = check_regrid_status(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                wrf_hydro_geo_meta,
                mpi_config,
            )
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    pt.log_debug("Calculating RAP regridding weights.")
                calculate_weights(
                    id_tmp,
                    force_count,
                    input_forcings,
                    config_options,
                    mpi_config,
                    wrf_hydro_geo_meta,
                )
                err_handler.check_program_status(config_options, mpi_config)

                # Read in the RAP height field, which is used for downscaling purposes.
                # if mpi_config.rank == 0:
                #     config_options.statusMsg = "Reading in RAP elevation data."
                #     err_handler.log_msg(config_options, mpi_config, True)
                # cmd = "$WGRIB2 " + input_forcings.file_in2 + " -match " + \
                #       "\":(HGT):(surface):\" " + \
                #       " -netcdf " + input_forcings.tmpFileHeight
                # id_tmp_height = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFileHeight,
                #                                  cmd, config_options, mpi_config, 'HGT_surface')
                # err_handler.check_program_status(config_options, mpi_config)
                if config_options.grid_type == "gridded":
                    # Regrid the height variable.
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract HGT_surface from : {id_tmp} ({err})"
                            )
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place temporary RAP elevation variable into ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding RAP surface elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid RAP elevation data using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to perform mask search on RAP elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place RAP ESMF elevation field into local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "unstructured":
                    # Regrid the height variable.
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract HGT_surface from : {id_tmp} ({err})"
                            )
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place temporary RAP elevation variable into ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding RAP surface elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid RAP elevation data using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to perform mask search on RAP elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place RAP ESMF elevation field into local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Regrid the height variable.
                    var_tmp_elem = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract HGT_surface from : {id_tmp} ({err})"
                            )
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(
                        input_forcings, var_tmp_elem, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place temporary RAP elevation variable into ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding RAP surface elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out_elem = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj_elem,
                                input_forcings.esmf_field_in_elem,
                                input_forcings.esmf_field_out_elem,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid RAP elevation data using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out_elem.data[
                            np.where(input_forcings.regridded_mask_elem == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to perform mask search on RAP elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = (
                            input_forcings.esmf_field_out_elem.data
                        )
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place RAP ESMF elevation field into local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    # Regrid the height variable.
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract HGT_surface from : {id_tmp} ({err})"
                            )
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place temporary RAP elevation variable into ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding RAP surface elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid RAP elevation data using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to perform mask search on RAP elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place RAP ESMF elevation field into local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                # Close the temporary NetCDF file and remove it.
                # if mpi_config.rank == 0:
                #     try:
                #         id_tmp_height.close()
                #     except OSError:
                #         config_options.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                #         err_handler.log_critical(config_options, mpi_config)
                #
                #     try:
                #         os_utils.os_remove_retry(input_forcings.tmpFileHeight)
                #     except OSError:
                #         config_options.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                #         err_handler.log_critical(config_options, mpi_config)
                # err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                # Regrid the input variables.
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                        if grib_var in ("APCP", "FROZR"):
                            var_tmp /= 3600  # convert hourly accumulated precip to instantaneous rate
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local RAP array into ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding Input RAP Field: {input_forcings.netcdf_var_names[force_count]}"
                    )
                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid RAP variable: {input_forcings.netcdf_var_names[force_count]}{ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to run mask calculation on RAP variable: {input_forcings.netcdf_var_names[force_count]} ({npe})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if force_count < 8:
                    try:
                        input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place RAP ESMF data into local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    # handle liquid-phase precip calculation
                    RAINRATE = 3  # TODO: determine this programmatically
                    total_pcp = np.ma.masked_values(
                        input_forcings.regridded_forcings2[RAINRATE],
                        config_options.globalNdv,
                    )
                    frozn_pcp = np.ma.masked_values(
                        input_forcings.esmf_field_out.data, config_options.globalNdv
                    )
                    # LOG.debug(f"rank {mpi_config.rank} has {(frozn_pcp > total_pcp).sum()} instances of frozn_pcp > total_pcp")
                    frz_fract = frozn_pcp / total_pcp
                    frz_fract[frz_fract > 1] = 1
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = (1 - frz_fract).filled(1.0)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                # Regrid the input variables.
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                        if grib_var in ("APCP", "FROZR"):
                            var_tmp /= 3600  # convert hourly accumulated precip to instantaneous rate
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local RAP array into ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding Input RAP Field: {input_forcings.netcdf_var_names[force_count]}"
                    )
                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid RAP variable: {input_forcings.netcdf_var_names[force_count]}{ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to run mask calculation on RAP variable: {input_forcings.netcdf_var_names[force_count]} ({npe})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if force_count < 8:
                    try:
                        input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place RAP ESMF data into local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    # handle liquid-phase precip calculation
                    RAINRATE = 3  # TODO: determine this programmatically
                    total_pcp = np.ma.masked_values(
                        input_forcings.regridded_forcings2[RAINRATE],
                        config_options.globalNdv,
                    )
                    frozn_pcp = np.ma.masked_values(
                        input_forcings.esmf_field_out.data, config_options.globalNdv
                    )
                    # LOG.debug(f"rank {mpi_config.rank} has {(frozn_pcp > total_pcp).sum()} instances of frozn_pcp > total_pcp")
                    frz_fract = frozn_pcp / total_pcp
                    frz_fract[frz_fract > 1] = 1
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = (1 - frz_fract).filled(1.0)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

                # Regrid the input variables.
                var_tmp_elem = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp_elem = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                        if grib_var in ("APCP", "FROZR"):
                            var_tmp_elem /= 3600  # convert hourly accumulated precip to instantaneous rate
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(
                    input_forcings, var_tmp_elem, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local RAP array into ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding Input RAP Field: {input_forcings.netcdf_var_names[force_count]}"
                    )
                try:
                    input_forcings.esmf_field_out_elem = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj_elem,
                            input_forcings.esmf_field_in_elem,
                            input_forcings.esmf_field_out_elem,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid RAP variable: {input_forcings.netcdf_var_names[force_count]}{ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out_elem.data[
                        np.where(input_forcings.regridded_mask_elem == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to run mask calculation on RAP variable: {input_forcings.netcdf_var_names[force_count]} ({npe})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if force_count < 8:
                    try:
                        input_forcings.regridded_forcings2_elem[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.esmf_field_out_elem.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place RAP ESMF element data into local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    # handle liquid-phase precip calculation
                    RAINRATE = 3  # TODO: determine this programmatically
                    total_pcp = np.ma.masked_values(
                        input_forcings.regridded_forcings2_elem[RAINRATE],
                        config_options.globalNdv,
                    )
                    frozn_pcp = np.ma.masked_values(
                        input_forcings.esmf_field_out_elem.data,
                        config_options.globalNdv,
                    )
                    # LOG.debug(f"rank {mpi_config.rank} has {(frozn_pcp > total_pcp).sum()} instances of frozn_pcp > total_pcp")
                    frz_fract = frozn_pcp / total_pcp
                    frz_fract[frz_fract > 1] = 1
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = (1 - frz_fract).filled(1.0)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "hydrofabric":
                # Regrid the input variables.
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                        if grib_var in ("APCP", "FROZR"):
                            var_tmp /= 3600  # convert hourly accumulated precip to instantaneous rate
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local RAP array into ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding Input RAP Field: {input_forcings.netcdf_var_names[force_count]}"
                    )
                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid RAP variable: {input_forcings.netcdf_var_names[force_count]}{ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to run mask calculation on RAP variable: {input_forcings.netcdf_var_names[force_count]} ({npe})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if force_count < 8:
                    try:
                        input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place RAP ESMF data into local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    # handle liquid-phase precip calculation
                    RAINRATE = 3  # TODO: determine this programmatically
                    total_pcp = np.ma.masked_values(
                        input_forcings.regridded_forcings2[RAINRATE],
                        config_options.globalNdv,
                    )
                    frozn_pcp = np.ma.masked_values(
                        input_forcings.esmf_field_out.data, config_options.globalNdv
                    )
                    # LOG.debug(f"rank {mpi_config.rank} has {(frozn_pcp > total_pcp).sum()} instances of frozn_pcp > total_pcp")
                    frz_fract = frozn_pcp / total_pcp
                    frz_fract[frz_fract > 1] = 1
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = (1 - frz_fract).filled(1.0)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

    finally:
        pt.close_rank_0_partial(id_tmp)
        pt.os_remove_rank_0_partial(input_forcings.tmpFile)


def regrid_cfsv2(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid global CFSv2 forecast data.

    Function for handling regridding of global CFSv2 forecast data.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        # Check to see if we are running NWM-custom interpolation/bias
        # correction on incoming CFSv2 data. Because of the nature, we
        # need to regrid bias-corrected data every hour.
        if mpi_config.rank == 0:
            pt.log_debug("No need to read in new CFSv2 data at this time.")
        return

    # Create a path for a temporary NetCDF file
    file_name = f"CFSv2_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    input_forcings.tmpFile = str(
        Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}"
    )

    err_handler.check_program_status(config_options, mpi_config)

    id_tmp = None
    try:
        pt.log_info("Regrid CFSv2")
        if input_forcings.input_force_types != NETCDF:
            pt.os_remove_rank_0_partial(input_forcings.tmpFile)

            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    pt.log_debug(f"Converting CFSv2 Variable: {grib_var}")
                fields.append(
                    f":{grib_var}:{input_forcings.grib_levels[force_count]}:{input_forcings.fcst_hour2} hour fcst:"
                )
            fields.append(":(HGT):(surface):")

            # Create a temporary NetCDF file from the GRIB2 file.
            if WGRIB2_env:
                cmd = f'$WGRIB2 -match "({"|".join(fields)})" {input_forcings.file_in2} -netcdf {input_forcings.tmpFile}'
            else:
                cmd = f"({'|'.join(fields)})"

            id_tmp = ioMod.open_grib2(
                input_forcings.file_in2,
                input_forcings.tmpFile,
                cmd,
                config_options,
                mpi_config,
                inputVar=None,
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link(
                "CFSv2",
                input_forcings.file_in2,
                input_forcings.tmpFile,
                config_options,
                mpi_config,
            )
            id_tmp = ioMod.open_netcdf_forcing(
                input_forcings.tmpFile, config_options, mpi_config
            )

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                pt.log_debug(f"Processing CFSv2 Variable: {grib_var}")

            calc_regrid_flag = check_regrid_status(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                wrf_hydro_geo_meta,
                mpi_config,
            )
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    pt.log_debug("Calculate CFSv2 regridding weights.")

                calculate_weights(
                    id_tmp,
                    force_count,
                    input_forcings,
                    config_options,
                    mpi_config,
                    wrf_hydro_geo_meta,
                )
                err_handler.check_program_status(config_options, mpi_config)

                # Read in the RAP height field, which is used for downscaling purposes.
                # if mpi_config.rank == 0:
                #     config_options.statusMsg = "Reading in CFSv2 elevation data."
                #     err_handler.log_msg(config_options, mpi_config, True)
                #
                # cmd = "$WGRIB2 " + input_forcings.file_in2 + " -match " + \
                #       "\":(HGT):(surface):\" " + \
                #       " -netcdf " + input_forcings.tmpFileHeight
                # id_tmp_height = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFileHeight,
                #                                  cmd, config_options, mpi_config, 'HGT_surface')
                # err_handler.check_program_status(config_options, mpi_config)

                if config_options.grid_type == "gridded":
                    # Regrid the height variable.
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract HGT_surface from file: {input_forcings.file_in2} ({err})"
                            )
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place CFSv2 elevation data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding CFSv2 elevation data to the WRF-Hydro domain."
                        )

                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid CFSv2 elevation data to the WRF-Hydro domain: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to run mask calculation on CFSv2 elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract CFSv2 regridded elevation data from ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "unstructured":
                    # Regrid the height variable.
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract HGT_surface from file: {input_forcings.file_in2} ({err})"
                            )
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place CFSv2 elevation data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding CFSv2 elevation data to the WRF-Hydro domain."
                        )

                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid CFSv2 elevation data to the WRF-Hydro domain: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to run mask calculation on CFSv2 elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract CFSv2 regridded elevation data from ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Regrid the height variable.
                    var_tmp_elem = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract HGT_surface from file: {input_forcings.file_in2} ({err})"
                            )
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(
                        input_forcings, var_tmp_elem, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place CFSv2 elevation data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding CFSv2 elevation data to the WRF-Hydro domain."
                        )

                    try:
                        input_forcings.esmf_field_out_elem = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj_elem,
                                input_forcings.esmf_field_in_elem,
                                input_forcings.esmf_field_out_elem,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid CFSv2 elevation data to the WRF-Hydro domain: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out_elem.data[
                            np.where(input_forcings.regridded_mask_elem == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to run mask calculation on CFSv2 elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = (
                            input_forcings.esmf_field_out_elem.data
                        )
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract CFSv2 regridded elevation data from ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    # Regrid the height variable.
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract HGT_surface from file: {input_forcings.file_in2} ({err})"
                            )
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place CFSv2 elevation data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding CFSv2 elevation data to the WRF-Hydro domain."
                        )

                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid CFSv2 elevation data to the WRF-Hydro domain: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to run mask calculation on CFSv2 elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract CFSv2 regridded elevation data from ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                # Close the temporary NetCDF file and remove it.
                # if mpi_config.rank == 0:
                #     try:
                #         id_tmp_height.close()
                #     except OSError:
                #         config_options.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                #         err_handler.log_critical(config_options, mpi_config)
                # err_handler.check_program_status(config_options, mpi_config)
                #
                # if mpi_config.rank == 0:
                #     try:
                #         os_utils.os_remove_retry(input_forcings.tmpFileHeight)
                #     except OSError:
                #         config_options.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                #         err_handler.log_critical(config_options, mpi_config)
                # err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                # Regrid the input variables.
                var_tmp = None
                if mpi_config.rank == 0:
                    if not config_options.runCfsNldasBiasCorrect:
                        pt.log_debug(
                            f"Regridding CFSv2 variable: {input_forcings.netcdf_var_names[force_count]}"
                        )
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from file: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                # Scatter the global CFSv2 data to the local processors.
                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                # Assign local CFSv2 data to the input forcing object.. IF..... we are running the
                # bias correction. These grids are interpolated in a separate routine, AFTER bias
                # correction has taken place.
                if config_options.runCfsNldasBiasCorrect:
                    if (
                        input_forcings.coarse_input_forcings1 is None
                    ):  # and config_options.current_output_step == 1:
                        # if not np.any(input_forcings.coarse_input_forcings1) and not \
                        #        np.any(input_forcings.coarse_input_forcings2) and \
                        #        ConfigOptions.current_output_step == 1:
                        # We need to create NumPy arrays to hold the CFSv2 global data.
                        input_forcings.coarse_input_forcings1 = np.empty(
                            [9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64
                        )

                    if (
                        input_forcings.coarse_input_forcings2 is None
                    ):  # and config_options.current_output_step == 1:
                        # if not np.any(input_forcings.coarse_input_forcings1) and not \
                        #        np.any(input_forcings.coarse_input_forcings2) and \
                        #        ConfigOptions.current_output_step == 1:
                        # We need to create NumPy arrays to hold the CFSv2 global data.
                        input_forcings.coarse_input_forcings2 = np.empty(
                            [9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64
                        )

                    try:
                        input_forcings.coarse_input_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = f"Unable to place local CFSv2 input variable: {input_forcings.netcdf_var_names[force_count]} into local numpy array. ({err})"
                    # except TypeError:
                    #    LOG.error(f"{input_forcings.coarse_input_forcings2}, {input_forcings.input_map_output}, {force_count}")

                    if config_options.current_output_step == 1:
                        input_forcings.coarse_input_forcings1[
                            input_forcings.input_map_output[force_count], :, :
                        ] = input_forcings.coarse_input_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ]
                else:
                    input_forcings.coarse_input_forcings2 = None
                    input_forcings.coarse_input_forcings1 = None
                err_handler.check_program_status(config_options, mpi_config)

                # Only regrid the current files if we did not specify the NLDAS2 NWM bias correction, which needs to take place
                # first before any regridding can take place. That takes place in the bias-correction routine.
                if not config_options.runCfsNldasBiasCorrect:
                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place CFSv2 forcing data into temporary ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid CFSv2 variable: {input_forcings.netcdf_var_names[force_count]} ({ve})"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to run mask calculation on CFSv2 variable: {input_forcings.netcdf_var_names[force_count]} ({npe})"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract ESMF field data for CFSv2: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # If we are on the first timestep, set the previous regridded field to be
                    # the latest as there are no states for time 0.
                    if config_options.current_output_step == 1:
                        input_forcings.regridded_forcings1[
                            input_forcings.input_map_output[force_count], :, :
                        ] = input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ]
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    # Set regridded arrays to dummy values as they are regridded later in the bias correction routine.
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :, :
                    ] = config_options.globalNdv
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = config_options.globalNdv

            elif config_options.grid_type == "unstructured":
                # Regrid the input variables.
                var_tmp = None
                if mpi_config.rank == 0:
                    if not config_options.runCfsNldasBiasCorrect:
                        pt.log_debug(
                            f"Regridding CFSv2 variable: {input_forcings.netcdf_var_names[force_count]}"
                        )
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from file: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                # Scatter the global CFSv2 data to the local processors.
                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                # Assign local CFSv2 data to the input forcing object.. IF..... we are running the
                # bias correction. These grids are interpolated in a separate routine, AFTER bias
                # correction has taken place.
                if config_options.runCfsNldasBiasCorrect:
                    if (
                        input_forcings.coarse_input_forcings1 is None
                    ):  # and config_options.current_output_step == 1:
                        # if not np.any(input_forcings.coarse_input_forcings1) and not \
                        #        np.any(input_forcings.coarse_input_forcings2) and \
                        #        ConfigOptions.current_output_step == 1:
                        # We need to create NumPy arrays to hold the CFSv2 global data.
                        input_forcings.coarse_input_forcings1 = np.empty(
                            [9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64
                        )

                    if (
                        input_forcings.coarse_input_forcings2 is None
                    ):  # and config_options.current_output_step == 1:
                        # if not np.any(input_forcings.coarse_input_forcings1) and not \
                        #        np.any(input_forcings.coarse_input_forcings2) and \
                        #        ConfigOptions.current_output_step == 1:
                        # We need to create NumPy arrays to hold the CFSv2 global data.
                        input_forcings.coarse_input_forcings2 = np.empty(
                            [9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64
                        )

                    try:
                        input_forcings.coarse_input_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = f"Unable to place local CFSv2 input variable: {input_forcings.netcdf_var_names[force_count]} into local numpy array. ({err})"
                    # except TypeError:
                    #    LOG.error(f"{input_forcings.coarse_input_forcings2}, {input_forcings.input_map_output}, {force_count}")

                    if config_options.current_output_step == 1:
                        input_forcings.coarse_input_forcings1[
                            input_forcings.input_map_output[force_count], :, :
                        ] = input_forcings.coarse_input_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ]
                else:
                    input_forcings.coarse_input_forcings2 = None
                    input_forcings.coarse_input_forcings1 = None
                err_handler.check_program_status(config_options, mpi_config)

                # Only regrid the current files if we did not specify the NLDAS2 NWM bias correction, which needs to take place
                # first before any regridding can take place. That takes place in the bias-correction routine.
                if not config_options.runCfsNldasBiasCorrect:
                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place CFSv2 forcing data into temporary ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid CFSv2 variable: {input_forcings.netcdf_var_names[force_count]} ({ve})"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to run mask calculation on CFSv2 variable: {input_forcings.netcdf_var_names[force_count]} ({npe})"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract ESMF field data for CFSv2: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # If we are on the first timestep, set the previous regridded field to be
                    # the latest as there are no states for time 0.
                    if config_options.current_output_step == 1:
                        input_forcings.regridded_forcings1[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :
                        ]
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    # Set regridded arrays to dummy values as they are regridded later in the bias correction routine.
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = config_options.globalNdv
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = config_options.globalNdv

                # Regrid the input variables.
                var_tmp_elem = None
                if mpi_config.rank == 0:
                    if not config_options.runCfsNldasBiasCorrect:
                        pt.log_debug(
                            f"Regridding CFSv2 variable: {input_forcings.netcdf_var_names[force_count]}"
                        )
                    try:
                        var_tmp_elem = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from file: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                # Scatter the global CFSv2 data to the local processors.
                var_sub_tmp_elem = mpi_config.scatter_array(
                    input_forcings, var_tmp_elem, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                # Assign local CFSv2 data to the input forcing object.. IF..... we are running the
                # bias correction. These grids are interpolated in a separate routine, AFTER bias
                # correction has taken place.
                if config_options.runCfsNldasBiasCorrect:
                    if (
                        input_forcings.coarse_input_forcings1_elem is None
                    ):  # and config_options.current_output_step == 1:
                        # if not np.any(input_forcings.coarse_input_forcings1) and not \
                        #        np.any(input_forcings.coarse_input_forcings2) and \
                        #        ConfigOptions.current_output_step == 1:
                        # We need to create NumPy arrays to hold the CFSv2 global data.
                        input_forcings.coarse_input_forcings1_elem = np.empty(
                            [9, var_sub_tmp_elem.shape[0], var_sub_tmp_elem.shape[1]],
                            np.float64,
                        )

                    if (
                        input_forcings.coarse_input_forcings2_elem is None
                    ):  # and config_options.current_output_step == 1:
                        # if not np.any(input_forcings.coarse_input_forcings1) and not \
                        #        np.any(input_forcings.coarse_input_forcings2) and \
                        #        ConfigOptions.current_output_step == 1:
                        # We need to create NumPy arrays to hold the CFSv2 global data.
                        input_forcings.coarse_input_forcings2_elem = np.empty(
                            [9, var_sub_tmp_elem.shape[0], var_sub_tmp_elem.shape[1]],
                            np.float64,
                        )

                    try:
                        input_forcings.coarse_input_forcings2_elem[
                            input_forcings.input_map_output[force_count], :, :
                        ] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = f"Unable to place local CFSv2 input variable: {input_forcings.netcdf_var_names[force_count]} into local numpy array. ({err})"
                    # except TypeError:
                    #    LOG.error(f"{input_forcings.coarse_input_forcings2}, {input_forcings.input_map_output}, {force_count})

                    if config_options.current_output_step == 1:
                        input_forcings.coarse_input_forcings1_elem[
                            input_forcings.input_map_output[force_count], :, :
                        ] = input_forcings.coarse_input_forcings2_elem[
                            input_forcings.input_map_output[force_count], :, :
                        ]
                else:
                    input_forcings.coarse_input_forcings2_elem = None
                    input_forcings.coarse_input_forcings1_elem = None
                err_handler.check_program_status(config_options, mpi_config)

                # Only regrid the current files if we did not specify the NLDAS2 NWM bias correction, which needs to take place
                # first before any regridding can take place. That takes place in the bias-correction routine.
                if not config_options.runCfsNldasBiasCorrect:
                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place CFSv2 forcing data into temporary ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj_elem,
                                input_forcings.esmf_field_in_elem,
                                input_forcings.esmf_field_out_elem,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid CFSv2 variable: {input_forcings.netcdf_var_names[force_count]} ({ve})"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out_elem.data[
                            np.where(input_forcings.regridded_mask_elem == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to run mask calculation on CFSv2 variable: {input_forcings.netcdf_var_names[force_count]} ({npe})"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.regridded_forcings2_elem[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.esmf_field_out_elem.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract ESMF field data for CFSv2: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # If we are on the first timestep, set the previous regridded field to be
                    # the latest as there are no states for time 0.
                    if config_options.current_output_step == 1:
                        input_forcings.regridded_forcings1_elem[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.regridded_forcings2_elem[
                            input_forcings.input_map_output[force_count], :
                        ]
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    # Set regridded arrays to dummy values as they are regridded later in the bias correction routine.
                    input_forcings.regridded_forcings1_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = config_options.globalNdv
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = config_options.globalNdv

            elif config_options.grid_type == "hydrofabric":
                # Regrid the input variables.
                var_tmp = None
                if mpi_config.rank == 0:
                    if not config_options.runCfsNldasBiasCorrect:
                        pt.log_debug(
                            f"Regridding CFSv2 variable: {input_forcings.netcdf_var_names[force_count]}"
                        )
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from file: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                # Scatter the global CFSv2 data to the local processors.
                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                # Assign local CFSv2 data to the input forcing object.. IF..... we are running the
                # bias correction. These grids are interpolated in a separate routine, AFTER bias
                # correction has taken place.
                if config_options.runCfsNldasBiasCorrect:
                    if (
                        input_forcings.coarse_input_forcings1 is None
                    ):  # and config_options.current_output_step == 1:
                        # if not np.any(input_forcings.coarse_input_forcings1) and not \
                        #        np.any(input_forcings.coarse_input_forcings2) and \
                        #        ConfigOptions.current_output_step == 1:
                        # We need to create NumPy arrays to hold the CFSv2 global data.
                        input_forcings.coarse_input_forcings1 = np.empty(
                            [9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64
                        )

                    if (
                        input_forcings.coarse_input_forcings2 is None
                    ):  # and config_options.current_output_step == 1:
                        # if not np.any(input_forcings.coarse_input_forcings1) and not \
                        #        np.any(input_forcings.coarse_input_forcings2) and \
                        #        ConfigOptions.current_output_step == 1:
                        # We need to create NumPy arrays to hold the CFSv2 global data.
                        input_forcings.coarse_input_forcings2 = np.empty(
                            [9, var_sub_tmp.shape[0], var_sub_tmp.shape[1]], np.float64
                        )

                    try:
                        input_forcings.coarse_input_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        config_options.errMsg = f"Unable to place local CFSv2 input variable: {input_forcings.netcdf_var_names[force_count]} into local numpy array. ({err})"
                    # except TypeError:
                    #    LOG.error(f"{input_forcings.coarse_input_forcings2}, {input_forcings.input_map_output}, {force_count}")

                    if config_options.current_output_step == 1:
                        input_forcings.coarse_input_forcings1[
                            input_forcings.input_map_output[force_count], :, :
                        ] = input_forcings.coarse_input_forcings2[
                            input_forcings.input_map_output[force_count], :, :
                        ]
                else:
                    input_forcings.coarse_input_forcings2 = None
                    input_forcings.coarse_input_forcings1 = None
                err_handler.check_program_status(config_options, mpi_config)

                # Only regrid the current files if we did not specify the NLDAS2 NWM bias correction, which needs to take place
                # first before any regridding can take place. That takes place in the bias-correction routine.
                if not config_options.runCfsNldasBiasCorrect:
                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place CFSv2 forcing data into temporary ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid CFSv2 variable: {input_forcings.netcdf_var_names[force_count]} ({ve})"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to run mask calculation on CFSv2 variable: {input_forcings.netcdf_var_names[force_count]} ({npe})"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract ESMF field data for CFSv2: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # If we are on the first timestep, set the previous regridded field to be
                    # the latest as there are no states for time 0.
                    if config_options.current_output_step == 1:
                        input_forcings.regridded_forcings1[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :
                        ]
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    # Set regridded arrays to dummy values as they are regridded later in the bias correction routine.
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = config_options.globalNdv
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = config_options.globalNdv

    finally:
        pt.close_rank_0_partial(id_tmp)
        pt.os_remove_rank_0_partial(input_forcings.tmpFile)


def regrid_nwm(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid custom input NetCDF hourly forcing files.

    Function for handling regridding of custom input NetCDF hourly forcing files.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

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
            pt.log_debug("No NWM NetCDF regridding required for this timestep.")
        return

    # Open the input NetCDF file containing necessary data.
    id_tmp = ioMod.open_netcdf_forcing(
        input_forcings.file_in2, config_options, mpi_config, open_on_all_procs=True
    )

    pt.log_info("Regrid NWM Custom NetCDF Forcing Variables")

    for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
        if mpi_config.rank == 0:
            pt.log_debug(f"Processing Custom NetCDF Forcing Variable: {nc_var}")
        calc_regrid_flag = check_regrid_status(
            id_tmp,
            force_count,
            input_forcings,
            config_options,
            wrf_hydro_geo_meta,
            mpi_config,
        )

        if calc_regrid_flag:
            calculate_weights(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                mpi_config,
                wrf_hydro_geo_meta,
            )

        input_forcings.height = None
        if mpi_config.rank == 0:
            pt.log_debug(
                f"Unable to locate HGT_surface in: {input_forcings.file_in2}. Downscaling will not be available."
            )

        if config_options.grid_type == "gridded":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding Custom netCDF input variable: {nc_var}")
                try:
                    var_tmp = id_tmp.variables[nc_var][:][0, :, :]
                except Exception as err:
                    pt.log_crit(
                        f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place local array into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid input Custom netCDF forcing variables using ESMF: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :, :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place local ESMF regridded data into local array: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :, :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :, :
                ]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding Custom netCDF input variable: {nc_var}")
                try:
                    var_tmp = id_tmp.variables[nc_var][:][0, :, :]
                except Exception as err:
                    pt.log_crit(
                        f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place local array into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid input Custom netCDF forcing variables using ESMF: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place local ESMF regridded data into local array: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the input variables.
            var_tmp_elem = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding Custom netCDF input variable: {nc_var}")
                try:
                    var_tmp_elem = id_tmp.variables[nc_var][:][0, :, :]
                except Exception as err:
                    pt.log_crit(
                        f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(
                input_forcings, var_tmp_elem, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place local array into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out_elem = (
                    pt.esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj_elem,
                        input_forcings.esmf_field_in_elem,
                        input_forcings.esmf_field_out_elem,
                    )
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid input Custom netCDF forcing variables using ESMF: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2_elem[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out_elem.data
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place local ESMF regridded data into local array: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1_elem[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2_elem[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding Custom netCDF input variable: {nc_var}")
                try:
                    var_tmp = id_tmp.variables[nc_var][:][0, :, :]
                except Exception as err:
                    pt.log_crit(
                        f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place local array into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid input Custom netCDF forcing variables using ESMF: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place local ESMF regridded data into local array: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)
    pt.close_rank_0_partial(id_tmp)


def regrid_nwm_aws(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid AWS NWM Forcing Data Downloaded from Server.

    Function for handling regridding of AWS NWM Forcing Data Downloaded from Server.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            pt.log_debug("No NWM NetCDF regridding required for this timestep.")
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

            id_tmp = id_tmp.assign_coords(
                longitude=(["y", "x"], lon_coords), latitude=(["y", "x"], lat_coords)
            )

    pt.log_info("Regrid NWM Custom zarr Forcing Variables")

    for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
        if mpi_config.rank == 0:
            pt.log_debug(f"Processing Custom zarr Forcing Variable: {nc_var}")
        calc_regrid_flag = check_regrid_status(
            id_tmp,
            force_count,
            input_forcings,
            config_options,
            wrf_hydro_geo_meta,
            mpi_config,
        )
        if calc_regrid_flag:
            calculate_weights(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                mpi_config,
                wrf_hydro_geo_meta,
            )

        input_forcings.height = None
        if mpi_config.rank == 0:
            pt.log_info(
                f"Unable to locate HGT_surface in: {input_forcings.file_in2}. Downscaling will not be available."
            )

        if config_options.grid_type == "gridded":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding Custom zarr input variable: {nc_var}")
                try:
                    var_tmp = id_tmp[nc_var].to_masked_array()
                except Exception as err:
                    pt.log_crit(
                        f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place local array into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid input Custom zarr forcing variables using ESMF: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :, :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place local ESMF regridded data into local array: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :, :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :, :
                ]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding Custom zarr input variable: {nc_var}")
                try:
                    var_tmp = id_tmp[nc_var].to_masked_array()
                except Exception as err:
                    pt.log_crit(
                        f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place local array into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid input Custom zarr forcing variables using ESMF: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place local ESMF regridded data into local array: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the input variables.
            var_tmp_elem = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding Custom zarr input variable: {nc_var}")
                try:
                    var_tmp_elem = id_tmp[nc_var].to_masked_array()
                except Exception as err:
                    pt.log_crit(
                        f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(
                input_forcings, var_tmp_elem, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place local array into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out_elem = (
                    pt.esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj_elem,
                        input_forcings.esmf_field_in_elem,
                        input_forcings.esmf_field_out_elem,
                    )
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid input Custom zarr forcing variables using ESMF: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2_elem[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out_elem.data
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place local ESMF regridded data into local array: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1_elem[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2_elem[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding Custom zarr input variable: {nc_var}")
                try:
                    var_tmp = id_tmp[nc_var].to_masked_array()
                except Exception as err:
                    pt.log_crit(
                        f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            # convert to array for NWM
            if input_forcings.product_name == "NWM":
                var_tmp = np.asarray(var_tmp, dtype=np.float64)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place local array into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid input Custom zarr forcing variables using ESMF: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place local ESMF regridded data into local array: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)
    pt.close_rank_0_partial(id_tmp)


def regrid_custom_hourly_netcdf(
    input_forcings, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Regrid Custom Hourly NetCDF Forcing Data.

    Function for handling regridding of custom input NetCDF hourly forcing files.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    with timing_block("Regrid AORC AWS"):
        # Flag to jump to different regridding function soley for AORC AWS data
        if config_options.aws:
            regrid_aorc_aws(
                input_forcings, config_options, wrf_hydro_geo_meta, mpi_config
            )
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
                pt.log_debug(
                    "No Custom Hourly NetCDF regridding required for this timestep."
                )
            return

        # Open the input NetCDF file containing necessary data.
        id_tmp = ioMod.open_netcdf_forcing(
            input_forcings.file_in2, config_options, mpi_config, open_on_all_procs=True
        )

        fill_values = {
            "TMP": 288.0,
            "SPFH": 0.005,
            "PRES": 101300.0,
            "APCP": 0,
            "UGRD": 1.0,
            "VGRD": 1.0,
            "DSWRF": 80.0,
            "DLWRF": 310.0,
        }

        pt.log_info("Regrid Custom Hourly NetCDF Forcing Variables")

        for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
            if mpi_config.rank == 0:
                pt.log_debug(f"Processing Custom NetCDF Forcing Variable: {nc_var}")
            calc_regrid_flag = check_regrid_status(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                wrf_hydro_geo_meta,
                mpi_config,
            )

            if calc_regrid_flag:
                calculate_weights(
                    id_tmp,
                    force_count,
                    input_forcings,
                    config_options,
                    mpi_config,
                    wrf_hydro_geo_meta,
                )

                # Flag to set regridded mask for AORC to overlay with ERA5-Interim blend
                if 23 in config_options.input_forcings:
                    input_forcings.regridded_mask_AORC = input_forcings.regridded_mask
                    if config_options.grid_type == "unstructured":
                        input_forcings.regridded_mask_elem_AORC = (
                            input_forcings.regridded_mask_elem
                        )

                # Read in the RAP height field, which is used for downscaling purposes.
                if "HGT_surface" in id_tmp.variables.keys():
                    if config_options.grid_type == "gridded":
                        # Regrid the height variable.
                        if mpi_config.rank == 0:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        else:
                            var_tmp = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp = mpi_config.scatter_array(
                            input_forcings, var_tmp, config_options
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to place NetCDF elevation data into the ESMF field object: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            pt.log_debug(
                                "Regridding elevation data to the WRF-Hydro domain."
                            )
                        try:
                            input_forcings.esmf_field_out = (
                                pt.esmf_regridobj_call_retry_partial(
                                    input_forcings.regridObj,
                                    input_forcings.esmf_field_in,
                                    input_forcings.esmf_field_out,
                                )
                            )
                        except ValueError as ve:
                            pt.log_crit(
                                f"Unable to regrid elevation data to the WRF-Hydro domain using ESMF: {ve}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        # Set any pixel cells outside the input domain to the global missing value.
                        try:
                            input_forcings.esmf_field_out.data[
                                np.where(input_forcings.regridded_mask == 0)
                            ] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            pt.log_crit(
                                f"Unable to compute mask on elevation data: {npe}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.height[:, :] = (
                                input_forcings.esmf_field_out.data
                            )
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract ESMF regridded elevation data to a local array: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                    elif config_options.grid_type == "unstructured":
                        # Regrid the height variable.
                        if mpi_config.rank == 0:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        else:
                            var_tmp = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp = mpi_config.scatter_array(
                            input_forcings, var_tmp, config_options
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to place NetCDF elevation data into the ESMF field object: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            pt.log_debug(
                                "Regridding elevation data to the WRF-Hydro domain."
                            )
                        try:
                            input_forcings.esmf_field_out = (
                                pt.esmf_regridobj_call_retry_partial(
                                    input_forcings.regridObj,
                                    input_forcings.esmf_field_in,
                                    input_forcings.esmf_field_out,
                                )
                            )
                        except ValueError as ve:
                            pt.log_crit(
                                f"Unable to regrid elevation data to the WRF-Hydro domain using ESMF: {ve}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        # Set any pixel cells outside the input domain to the global missing value.
                        try:
                            input_forcings.esmf_field_out.data[
                                np.where(input_forcings.regridded_mask == 0)
                            ] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            pt.log_crit(
                                f"Unable to compute mask on elevation data: {npe}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.height[:] = (
                                input_forcings.esmf_field_out.data
                            )
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract ESMF regridded elevation data to a local array: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        # Regrid the height variable.
                        if mpi_config.rank == 0:
                            var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                        else:
                            var_tmp_elem = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp_elem = mpi_config.scatter_array(
                            input_forcings, var_tmp_elem, config_options
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_in_elem.data[:, :] = (
                                var_sub_tmp_elem
                            )
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to place NetCDF elevation data into the ESMF field object: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            pt.log_debug(
                                "Regridding elevation data to the WRF-Hydro domain."
                            )
                        try:
                            input_forcings.esmf_field_out_elem = (
                                pt.esmf_regridobj_call_retry_partial(
                                    input_forcings.regridObj_elem,
                                    input_forcings.esmf_field_in_elem,
                                    input_forcings.esmf_field_out_elem,
                                )
                            )
                        except ValueError as ve:
                            pt.log_crit(
                                f"Unable to regrid elevation data to the WRF-Hydro domain using ESMF: {ve}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        # Set any pixel cells outside the input domain to the global missing value.
                        try:
                            input_forcings.esmf_field_out_elem.data[
                                np.where(input_forcings.regridded_mask_elem == 0)
                            ] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            pt.log_crit(
                                f"Unable to compute mask on elevation data: {npe}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.height_elem[:] = (
                                input_forcings.esmf_field_out_elem.data
                            )
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract ESMF regridded elevation data to a local array: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                    elif config_options.grid_type == "hydrofabric":
                        # Regrid the height variable.
                        if mpi_config.rank == 0:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        else:
                            var_tmp = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp = mpi_config.scatter_array(
                            input_forcings, var_tmp, config_options
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to place NetCDF elevation data into the ESMF field object: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            pt.log_debug(
                                "Regridding elevation data to the WRF-Hydro domain."
                            )
                        try:
                            input_forcings.esmf_field_out = (
                                pt.esmf_regridobj_call_retry_partial(
                                    input_forcings.regridObj,
                                    input_forcings.esmf_field_in,
                                    input_forcings.esmf_field_out,
                                )
                            )
                        except ValueError as ve:
                            pt.log_crit(
                                f"Unable to regrid elevation data to the WRF-Hydro domain using ESMF: {ve}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        # Set any pixel cells outside the input domain to the global missing value.
                        try:
                            input_forcings.esmf_field_out.data[
                                np.where(input_forcings.regridded_mask == 0)
                            ] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            pt.log_crit(
                                f"Unable to compute mask on elevation data: {npe}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            input_forcings.height[:] = (
                                input_forcings.esmf_field_out.data
                            )
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract ESMF regridded elevation data to a local array: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                else:
                    input_forcings.height = None
                    if mpi_config.rank == 0:
                        pt.log_info(
                            f"Unable to locate HGT_surface in: {input_forcings.file_in2}. Downscaling will not be available."
                        )

                # close netCDF file on non-root ranks
                if mpi_config.rank != 0:
                    id_tmp.close()

            if config_options.grid_type == "gridded":
                # Regrid the input variables.
                var_tmp = None
                fill = fill_values.get(
                    input_forcings.grib_vars[force_count], config_options.globalNdv
                )
                if mpi_config.rank == 0:
                    pt.log_debug(f"Regridding Custom netCDF input variable: {nc_var}")
                    try:
                        pt.log_debug(f"Using {fill} to replace missing values in input")
                        var_tmp = id_tmp.variables[nc_var][:].filled(fill)[0, :, :]
                    except Exception as err:
                        pt.log_crit(
                            f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local array into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid input Custom netCDF forcing variables using ESMF: {ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = fill
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to calculate mask from input Custom netCDF regridded forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Convert the hourly precipitation total to a rate of mm/s
                if nc_var == "APCP_surface":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != fill)
                        # Flag to set regridded mask for AORC to overlay with ERA5-Interim blend
                        input_forcings.esmf_field_out.data[ind_valid] = (
                            input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        )
                        del ind_valid
                    except (
                        ValueError,
                        ArithmeticError,
                        AttributeError,
                        KeyError,
                    ) as npe:
                        pt.log_crit(
                            f"Unable to run NDV search on Custom netCDF precipitation: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local ESMF regridded data into local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                # Regrid the input variables.
                var_tmp = None
                fill = fill_values.get(
                    input_forcings.grib_vars[force_count], config_options.globalNdv
                )
                if mpi_config.rank == 0:
                    pt.log_debug(f"Regridding Custom netCDF input variable: {nc_var}")
                    try:
                        pt.log_debug(f"Using {fill} to replace missing values in input")
                        var_tmp = id_tmp.variables[nc_var][:].filled(fill)[0, :, :]
                    except Exception as err:
                        pt.log_crit(
                            f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local array into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid input Custom netCDF forcing variables using ESMF: {ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = fill
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to calculate mask from input Custom netCDF regridded forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Convert the hourly precipitation total to a rate of mm/s
                if nc_var == "APCP_surface":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != fill)
                        input_forcings.esmf_field_out.data[ind_valid] = (
                            input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        )
                        del ind_valid
                    except (
                        ValueError,
                        ArithmeticError,
                        AttributeError,
                        KeyError,
                    ) as npe:
                        pt.log_crit(
                            f"Unable to run NDV search on Custom netCDF precipitation: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local ESMF regridded data into local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

                # Regrid the input variables.
                var_tmp_elem = None
                fill = fill_values.get(
                    input_forcings.grib_vars[force_count], config_options.globalNdv
                )
                if mpi_config.rank == 0:
                    pt.log_debug(f"Regridding Custom netCDF input variable: {nc_var}")
                    try:
                        pt.log_debug(f"Using {fill} to replace missing values in input")
                        var_tmp_elem = id_tmp.variables[nc_var][:].filled(fill)[0, :, :]
                    except Exception as err:
                        pt.log_crit(
                            f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(
                    input_forcings, var_tmp_elem, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local array into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj_elem,
                            input_forcings.esmf_field_in_elem,
                            input_forcings.esmf_field_out_elem,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid input Custom netCDF forcing variables using ESMF: {ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out_elem.data[
                        np.where(input_forcings.regridded_mask_elem == 0)
                    ] = fill
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to calculate mask from input Custom netCDF regridded forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Convert the hourly precipitation total to a rate of mm/s
                if nc_var == "APCP_surface":
                    try:
                        ind_valid_elem = np.where(
                            input_forcings.esmf_field_out_elem.data != fill
                        )
                        input_forcings.esmf_field_out_elem.data[ind_valid_elem] = (
                            input_forcings.esmf_field_out_elem.data[ind_valid_elem]
                            / 3600.0
                        )
                        del ind_valid_elem
                    except (
                        ValueError,
                        ArithmeticError,
                        AttributeError,
                        KeyError,
                    ) as npe:
                        pt.log_crit(
                            f"Unable to run NDV search on Custom netCDF precipitation: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local ESMF regridded data into local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "hydrofabric":
                # Regrid the input variables.
                var_tmp = None
                fill = fill_values.get(
                    input_forcings.grib_vars[force_count], config_options.globalNdv
                )
                if mpi_config.rank == 0:
                    pt.log_debug(f"Regridding Custom netCDF input variable: {nc_var}")
                    try:
                        pt.log_debug(f"Using {fill} to replace missing values in input")
                        var_tmp = id_tmp.variables[nc_var][:].filled(fill)[0, :, :]
                    except Exception as err:
                        pt.log_crit(
                            f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)
                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local array into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid input Custom netCDF forcing variables using ESMF: {ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = fill
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to calculate mask from input Custom netCDF regridded forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if nc_var == "DSWRF_surface":
                    ind_valid = np.where(input_forcings.esmf_field_out.data < 0.0)
                    input_forcings.esmf_field_out.data[ind_valid] = 0.0
                # Convert the hourly precipitation total to a rate of mm/s
                if nc_var == "APCP_surface":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != fill)
                        input_forcings.esmf_field_out.data[ind_valid] = (
                            input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        )
                        ind_valid = np.where(input_forcings.esmf_field_out.data < 0.0)
                        input_forcings.esmf_field_out.data[ind_valid] = 0.0
                        del ind_valid
                    except (
                        ValueError,
                        ArithmeticError,
                        AttributeError,
                        KeyError,
                    ) as npe:
                        pt.log_crit(
                            f"Unable to run NDV search on Custom netCDF precipitation: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local ESMF regridded data into local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)
        pt.close_rank_0_partial(id_tmp)


def regrid_era5(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Rebgrid ERA5-Interim Forcing Variables.

    Function for handling regridding of a single ERA5 netcdf custom file.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in1):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            pt.log_debug("No ERA5-Interim regridding required for this timestep.")
        return

    # Open the input NetCDF file containing necessary data.
    id_tmp = ioMod.open_netcdf_forcing(
        input_forcings.file_in2, config_options, mpi_config, open_on_all_procs=True
    )

    # create netcdf time stamp
    time = nc.num2date(
        id_tmp.variables["time"][:].data,
        units=id_tmp.variables["time"].units,
        only_use_cftime_datetimes=False,
    )
    # Find the timestamp index based on latest timestamp requested
    # by the forcings engine
    seconds_index = np.abs(
        (
            pd.to_datetime(time) - pd.to_datetime(config_options.current_time)
        ).total_seconds()
    )
    ind = np.where(seconds_index == np.min(seconds_index))[0][0]

    pt.log_info("Regrid Custom Hourly NetCDF Forcing Variables")

    for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
        if mpi_config.rank == 0:
            pt.log_debug(f"Processing ERA-5 Interim Forcing Variable: {nc_var}")
        calc_regrid_flag = check_regrid_status(
            id_tmp,
            force_count,
            input_forcings,
            config_options,
            wrf_hydro_geo_meta,
            mpi_config,
        )
        if calc_regrid_flag:
            calculate_weights(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                mpi_config,
                wrf_hydro_geo_meta,
            )

            # Read in the ERA5-Interim height field, which is used for downscaling purposes.
            if "Geopotential" in id_tmp.variables.keys():
                LOG.info("Found geopotential height in ERA5-Interim data")
            # To Do, see if NCAR downscaling methods are applicable
            # for reanalysis datasets with coarser resolution
            else:
                input_forcings.height = None
                if mpi_config.rank == 0:
                    pt.log_info(
                        f"Unable to locate Geopoential height in: {input_forcings.file_in2}. Downscaling will not be available."
                    )

            # close netCDF file on non-root ranks
            if mpi_config.rank != 0:
                id_tmp.close()

        if config_options.grid_type == "gridded":
            # Regrid the input variables.
            var_tmp = None

            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding ERA5-Interim input variable: {nc_var}")
                try:
                    pt.log_debug("Using -9999. to replace missing values in input")
                    var_tmp = id_tmp.variables[nc_var][:].filled(-9999.0)[ind, :, :]
                    # Since ERA5-Interim only supplies dew points
                    # we will need to calculate the specific humidity
                    if nc_var == "d2m":
                        var_tmp = var_tmp - 273.15
                        e = 6.112 * np.exp((17.67 * var_tmp) / (var_tmp + 243.5))
                        pres = (
                            id_tmp.variables["sp"][:].filled(-9999.0)[ind, :, :] / 100
                        )
                        var_tmp = (0.622 * e) / (pres - (0.378 * e))
                        del e
                        del pres
                except Exception as err:
                    pt.log_crit(
                        f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place local array into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid input ERA5-Interim forcing variables using ESMF: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[
                    np.where(input_forcings.regridded_mask == 0)
                ] = -9999.0
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to calculate mask from input ERA5-Interim regridded forcings: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total to a rate of mm/s
            if nc_var == "mtpr":
                try:
                    ind_valid = np.where(input_forcings.esmf_field_out.data != -9999.0)
                    input_forcings.esmf_field_out.data[ind_valid] = (
                        input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                    )
                    del ind_valid
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    pt.log_crit(
                        f"Unable to run NDV search on ERA5-Interim precipitation: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :, :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place local ESMF regridded data into local array: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :, :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :, :
                ]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding ERA5-Interim input variable: {nc_var}")
                try:
                    pt.log_debug("Using -9999. to replace missing values in input")
                    var_tmp = id_tmp.variables[nc_var][:].filled(-9999.0)[ind, :, :]
                    # Since ERA5-Interim only supplies dew points
                    # we will need to calculate the specific humidity
                    if nc_var == "d2m":
                        var_tmp = var_tmp - 273.15
                        e = 6.112 * np.exp((17.67 * var_tmp) / (var_tmp + 243.5))
                        pres = (
                            id_tmp.variables["sp"][:].filled(-9999.0)[ind, :, :] / 100
                        )
                        var_tmp = (0.622 * e) / (pres - (0.378 * e))
                        del e
                        del pres
                except Exception as err:
                    pt.log_crit(
                        f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place local array into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid input ERA5-Interim forcing variables using ESMF: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[
                    np.where(input_forcings.regridded_mask == 0)
                ] = -9999.0
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to calculate mask from input ERA5-Interim regridded forcings: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total to a rate of mm/s
            if nc_var == "mtpr":
                try:
                    ind_valid = np.where(input_forcings.esmf_field_out.data != -9999.0)
                    input_forcings.esmf_field_out.data[ind_valid] = (
                        input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                    )
                    del ind_valid
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    pt.log_crit(
                        f"Unable to run NDV search on ERA5-Interim precipitation: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place local ESMF regridded data into local array: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the input variables.
            var_tmp_elem = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding ERA5-Interim input variable: {nc_var}")
                try:
                    pt.log_debug("Using -9999. to replace missing values in input")
                    var_tmp_elem = id_tmp.variables[nc_var][:].filled(-9999.0)[
                        ind, :, :
                    ]
                    # Since ERA5-Interim only supplies dew points
                    # we will need to calculate the specific humidity
                    if nc_var == "d2m":
                        var_tmp_elem = var_tmp_elem - 273.15
                        e = 6.112 * np.exp(
                            (17.67 * var_tmp_elem) / (var_tmp_elem + 243.5)
                        )
                        pres = (
                            id_tmp.variables["sp"][:].filled(-9999.0)[ind, :, :] / 100
                        )
                        var_tmp = (0.622 * e) / (pres - (0.378 * e))
                        del e
                        del pres
                except Exception as err:
                    pt.log_crit(
                        f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(
                input_forcings, var_tmp_elem, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place local array into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out_elem = (
                    pt.esmf_regridobj_call_retry_partial(
                        input_forcings.regridObj_elem,
                        input_forcings.esmf_field_in_elem,
                        input_forcings.esmf_field_out_elem,
                    )
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid input ERA5-Interim forcing variables using ESMF: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out_elem.data[
                    np.where(input_forcings.regridded_mask_elem == 0)
                ] = -9999.0
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to calculate mask from input Custom netCDF regridded forcings: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total to a rate of mm/s
            if nc_var == "mtpr":
                try:
                    ind_valid_elem = np.where(
                        input_forcings.esmf_field_out_elem.data != -9999.0
                    )
                    input_forcings.esmf_field_out_elem.data[ind_valid_elem] = (
                        input_forcings.esmf_field_out_elem.data[ind_valid_elem] / 3600.0
                    )
                    del ind_valid_elem
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    pt.log_crit(
                        f"Unable to run NDV search on ERA5-Interim precipitation: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2_elem[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out_elem.data

            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place local ESMF regridded data into local array: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1_elem[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2_elem[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding ERA5-Interim input variable: {nc_var}")
                try:
                    pt.log_debug("Using -9999. to replace missing values in input")
                    var_tmp = id_tmp.variables[nc_var][:].filled(-9999.0)[ind, :, :]
                    # Since ERA5-Interim only supplies dew points
                    # we will need to calculate the specific humidity
                    if nc_var == "d2m":
                        var_tmp = var_tmp - 273.15
                        e = 6.112 * np.exp((17.67 * var_tmp) / (var_tmp + 243.5))
                        pres = (
                            id_tmp.variables["sp"][:].filled(-9999.0)[ind, :, :] / 100
                        )
                        var_tmp = (0.622 * e) / (pres - (0.378 * e))
                        del e
                        del pres
                except Exception as err:
                    pt.log_crit(
                        f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place local array into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid input ERA5-Interim forcing variables using ESMF: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                input_forcings.esmf_field_out.data[
                    np.where(input_forcings.regridded_mask == 0)
                ] = -9999.0
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to calculate mask from input ERA5-Interim regridded forcings: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total to a rate of mm/s
            if nc_var == "mtpr":
                try:
                    ind_valid = np.where(input_forcings.esmf_field_out.data != -9999.0)
                    input_forcings.esmf_field_out.data[ind_valid] = (
                        input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                    )
                    del ind_valid
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    pt.log_crit(
                        f"Unable to run NDV search on ERA5-Interim precipitation: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place local ESMF regridded data into local array: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                input_forcings.regridded_forcings1[
                    input_forcings.input_map_output[force_count], :
                ] = input_forcings.regridded_forcings2[
                    input_forcings.input_map_output[force_count], :
                ]
            err_handler.check_program_status(config_options, mpi_config)

    pt.close_rank_0_partial(id_tmp)


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
    pt = Partials(mpi_config, config_options)

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            pt.log_debug("No 13km GFS regridding required for this timestep.")
        return

    # Create a temporary NetCDF file name to hold converted GRIB2 data.
    # Previous non-unique tmp files could cause issues on some file systems
    # Unclear how beneficial the reuse of tmp files was/is

    file_uuid = str(mpi_config.uid64)
    file_name = f"GFS_TMP-{mkfilename()}.nc"
    input_forcings.tmpFile = str(
        Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}"
    )
    # err_handler.check_program_status(config_options, mpi_config)

    # We will process each variable at a time. Unfortunately, wgrib2 makes it a bit
    # difficult to handle forecast strings, otherwise this could be done in one command.
    # This makes a compelling case for the use of a GRIB Python API in the future....
    # Incoming shortwave radiation flux.....

    # Loop through all of the input forcings in GFS data. Convert the GRIB2 files
    # to NetCDF, read in the data, regrid it, then map it to the appropriate
    # array slice in the output arrays.

    id_tmp = None
    try:
        pt.log_info("Regridding 13km GFS Variables.")

        if input_forcings.input_force_types != NETCDF:
            pt.os_remove_rank_0_partial(input_forcings.tmpFile)

            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    pt.log_debug(f"Converting 13km GFS Variable: {grib_var}")
                # Create a temporary NetCDF file from the GRIB2 file.
                if grib_var == "PRATE":
                    # By far the most complicated of output variables. We need to calculate
                    # our 'average' PRATE based on our current hour.
                    if input_forcings.fcst_hour2 <= 384:
                        tmp_hr_current = input_forcings.fcst_hour2

                        diff_tmp = tmp_hr_current % 6 if tmp_hr_current % 6 > 0 else 6
                        tmp_hr_previous = tmp_hr_current - diff_tmp

                    else:
                        tmp_hr_previous = input_forcings.fcst_hour1

                    fields.append(
                        f":{grib_var}:{input_forcings.grib_levels[force_count]}:{tmp_hr_previous}-{input_forcings.fcst_hour2} hour ave fcst:"
                    )
                else:
                    fields.append(
                        f":{grib_var}:{input_forcings.grib_levels[force_count]}:{input_forcings.fcst_hour2} hour fcst:"
                    )

            # if calc_regrid_flag:
            fields.append(":(HGT):(surface):")
            if WGRIB2_env:
                cmd = f'$WGRIB2 -match "({"|".join(fields)})" {input_forcings.file_in2} -netcdf {input_forcings.tmpFile}'
            else:
                cmd = f"({'|'.join(fields)})"

            id_tmp = ioMod.open_grib2(
                input_forcings.file_in2,
                input_forcings.tmpFile,
                cmd,
                config_options,
                mpi_config,
                inputVar=None,
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link(
                "GFS",
                input_forcings.file_in2,
                input_forcings.tmpFile,
                config_options,
                mpi_config,
            )
            id_tmp = ioMod.open_netcdf_forcing(
                input_forcings.tmpFile, config_options, mpi_config
            )

        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                pt.log_debug(f"Processing 13km GFS Variable: {grib_var}")

            calc_regrid_flag = check_regrid_status(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                wrf_hydro_geo_meta,
                mpi_config,
            )
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    pt.log_debug("Calculating 13km GFS regridding weights.")
                calculate_weights(
                    id_tmp,
                    force_count,
                    input_forcings,
                    config_options,
                    mpi_config,
                    wrf_hydro_geo_meta,
                )
                err_handler.check_program_status(config_options, mpi_config)

                # Read in the GFS height field, which is used for downscaling purposes.
                # if mpi_config.rank == 0:
                #    config_options.statusMsg = "Reading in 13km GFS elevation data."
                #    err_handler.log_msg(config_options, mpi_config, True)
                # cmd = "$WGRIB2 " + input_forcings.file_in2 + " -match " + \
                #    "\":(HGT):(surface):\" " + \
                #    " -netcdf " + input_forcings.tmpFileHeight
                # time.sleep(1)
                # id_tmp_height = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFileHeight,
                #                                 cmd, config_options, mpi_config, 'HGT_surface')
                # err_handler.check_program_status(config_options, mpi_config)

                # Regrid the height variable.
                if config_options.grid_type == "gridded":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract GFS elevation from: {input_forcings.tmpFile} ({err})"
                            )
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)
                elif config_options.grid_type == "unstructured":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract GFS elevation from: {input_forcings.tmpFile} ({err})"
                            )
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    var_tmp_elem = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract GFS elevation from: {input_forcings.tmpFile} ({err})"
                            )
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(
                        input_forcings, var_tmp_elem, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)
                elif config_options.grid_type == "hydrofabric":
                    var_tmp = None
                    if mpi_config.rank == 0:
                        try:
                            var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract GFS elevation from: {input_forcings.tmpFile} ({err})"
                            )
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local GFS array into an ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    pt.log_debug(
                        "Regridding 13km GFS surface elevation data to the WRF-Hydro domain."
                    )
                if config_options.grid_type == "gridded":
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(f"Unable to regrid GFS elevation data: {ve}")
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to perform mask search on GFS elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "unstructured":
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid GFS elevation data with ESMF mesh nodes: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place local GFS array into an ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_out_elem = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj_elem,
                                input_forcings.esmf_field_in_elem,
                                input_forcings.esmf_field_out_elem,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid GFS elevation data with ESMF mesh elements: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to perform mask search on GFS elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out_elem.data[
                            np.where(input_forcings.regridded_mask_elem == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to perform mask search on GFS elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)
                elif config_options.grid_type == "hydrofabric":
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(f"Unable to regrid GFS elevation data: {ve}")
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to perform mask search on GFS elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                if config_options.grid_type == "gridded":
                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract GFS elevation array from ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)
                elif config_options.grid_type == "unstructured":
                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract GFS elevation array from ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = (
                            input_forcings.esmf_field_out_elem.data
                        )
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract GFS elevation array from ESMF field with mesh elements: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract GFS elevation array from ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                # Close the temporary NetCDF file and remove it.
                # if mpi_config.rank == 0:
                #    try:
                #        id_tmp_height.close()
                #    except OSError:
                #        config_options.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                #        err_handler.log_critical(config_options, mpi_config)

                #    try:
                #        os_utils.os_remove_retry(input_forcings.tmpFileHeight)
                #    except OSError:
                #        config_options.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                #        err_handler.log_critical(config_options, mpi_config)
                # err_handler.check_program_status(config_options, mpi_config)

            # Regrid the input variables.
            if config_options.grid_type == "gridded":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_tmp_elem = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp_elem = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "hydrofabric":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

            # If we are regridding GFS data, and this is precipitation, we need to run calculations
            # on the global precipitation average rates to calculate instantaneous global rates.
            # This is due to GFS's weird nature of doing average rates over different periods.
            if input_forcings.product_name == "GFS_Production_GRIB2":
                if grib_var == "PRATE":
                    if mpi_config.rank == 0:
                        if config_options.grid_type == "gridded":
                            input_forcings.globalPcpRate2 = var_tmp
                            var_tmp = timeInterpMod.gfs_pcp_time_interp(
                                input_forcings, config_options, mpi_config
                            )
                        elif config_options.grid_type == "unstructured":
                            input_forcings.globalPcpRate2 = var_tmp
                            input_forcings.globalPcpRate2_elem = var_tmp_elem
                            var_tmp, var_tmp_elem = timeInterpMod.gfs_pcp_time_interp(
                                input_forcings, config_options, mpi_config
                            )
                        elif config_options.grid_type == "hydrofabric":
                            input_forcings.globalPcpRate2 = var_tmp
                            var_tmp = timeInterpMod.gfs_pcp_time_interp(
                                input_forcings, config_options, mpi_config
                            )

            if grib_var == "CPOFP":
                if mpi_config.rank == 0:
                    # LOG.debug(f"CPOFP stats, min={var_tmp[var_tmp > 0].min()} mean={var_tmp[var_tmp > 0].mean()} max={var_tmp[var_tmp > 0].max()}")
                    var_tmp[var_tmp >= 0] = (
                        100 - var_tmp[var_tmp >= 0]
                    ) / 100  # convert frozen fraction to liquid fraction
                    var_tmp[var_tmp < 0] = (
                        1.0  # assume all liquid if not specifically given
                    )
                    if config_options.grid_type == "unstructured":
                        var_tmp_elem[var_tmp_elem >= 0] = (
                            100 - var_tmp_elem[var_tmp_elem >= 0]
                        ) / 100  # convert frozen fraction to liquid fraction
                        var_tmp_elem[var_tmp_elem < 0] = (
                            1.0  # assume all liquid if not specifically given
                        )

            if config_options.grid_type == "gridded":
                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                mpi_config.comm.barrier()
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "unstructured":
                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                mpi_config.comm.barrier()
                err_handler.check_program_status(config_options, mpi_config)
                var_sub_tmp_elem = mpi_config.scatter_array(
                    input_forcings, var_tmp_elem, config_options
                )
                mpi_config.comm.barrier()
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "hydrofabric":
                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                mpi_config.comm.barrier()
                err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place GFS local array into ESMF element field object: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)
            if config_options.grid_type == "unstructured":
                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place GFS local array into ESMF element field object: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)
                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place GFS local array into ESMF element field object: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)
            elif config_options.grid_type == "hydrofabric":
                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place GFS local array into ESMF element field object: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding Input 13km GFS Field: {input_forcings.netcdf_var_names[force_count]}"
                    )
                try:
                    begin = monotonic()
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                    end = monotonic()
                    if mpi_config.rank == 0:
                        pt.log_debug(f"Regridding took {end - begin} seconds")
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid GFS variable: {input_forcings.netcdf_var_names[force_count]} ({ve})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to run mask search on GFS variable: {input_forcings.netcdf_var_names[force_count]} ({npe})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract GFS ESMF field data to local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)
                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding Input 13km GFS Field for Mesh nodes: {input_forcings.netcdf_var_names[force_count]}"
                    )
                try:
                    begin = monotonic()
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                    end = monotonic()
                    if mpi_config.rank == 0:
                        pt.log_debug(f"Node Regridding took {end - begin} seconds")
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid GFS variable for Mesh Nodes: {input_forcings.netcdf_var_names[force_count]} ({ve})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to run mask search on GFS variable: {input_forcings.netcdf_var_names[force_count]} ({npe})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract GFS ESMF field data to local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)
                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding Input 13km GFS Field for Mesh elements: {input_forcings.netcdf_var_names[force_count]}"
                    )
                try:
                    begin = monotonic()
                    input_forcings.esmf_field_out_elem = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj_elem,
                            input_forcings.esmf_field_in_elem,
                            input_forcings.esmf_field_out_elem,
                        )
                    )
                    end = monotonic()
                    if mpi_config.rank == 0:
                        pt.log_debug(f"Element Regridding took {end - begin} seconds")
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid GFS variable for Mesh Elements: {input_forcings.netcdf_var_names[force_count]} ({ve})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out_elem.data[
                        np.where(input_forcings.regridded_mask_elem == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to run mask search on GFS variable: {input_forcings.netcdf_var_names[force_count]} ({npe})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract GFS ESMF field data to local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)
                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "hydrofabric":
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding Input 13km GFS Field for Mesh Elements: {input_forcings.netcdf_var_names[force_count]}"
                    )
                try:
                    begin = monotonic()
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                    end = monotonic()
                    if mpi_config.rank == 0:
                        pt.log_debug(f"Element Regridding took {end - begin} seconds")
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid GFS variable for Mesh Element: {input_forcings.netcdf_var_names[force_count]} ({ve})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to run mask search on GFS variable: {input_forcings.netcdf_var_names[force_count]} ({npe})"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract GFS ESMF field data to local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)
                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

    finally:
        pt.close_rank_0_partial(id_tmp)
        pt.os_remove_rank_0_partial(input_forcings.tmpFile)


def regrid_nam_nest(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Rebgrid NAM Nest Data.

    Function for handing regridding of input NAM nest data
    fro GRIB2 files.
    :param mpi_config:
    :param wrf_hydro_geo_meta:
    :param input_forcings:
    :param config_options:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        pt.log_debug(
            "No regridding of NAM nest data necessary for this timestep - already completed."
        )
        return

    # Create a path for a temporary NetCDF file
    file_name = f"NAM_CONUS_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    input_forcings.tmpFile = str(
        Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}"
    )

    err_handler.check_program_status(config_options, mpi_config)

    id_tmp = None
    try:
        pt.log_info("Regridding NAM nest data")
        if input_forcings.input_force_types != NETCDF:
            pt.os_remove_rank_0_partial(input_forcings.tmpFile)

            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    pt.log_debug(f"Converting NAM-Nest Variable: {grib_var}")
                fields.append(
                    f":{grib_var}:{input_forcings.grib_levels[force_count]}:{input_forcings.fcst_hour2} hour fcst:"
                )
            fields.append(":(HGT):(surface):")

            # Create a temporary NetCDF file from the GRIB2 file.
            if WGRIB2_env:
                cmd = f'$WGRIB2 -match "({"|".join(fields)})" {input_forcings.file_in2} -netcdf {input_forcings.tmpFile}'
            else:
                cmd = f"({'|'.join(fields)})"

            id_tmp = ioMod.open_grib2(
                input_forcings.file_in2,
                input_forcings.tmpFile,
                cmd,
                config_options,
                mpi_config,
                inputVar=None,
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link(
                "NAM-Nest",
                input_forcings.file_in2,
                input_forcings.tmpFile,
                config_options,
                mpi_config,
            )
            id_tmp = ioMod.open_netcdf_forcing(
                input_forcings.tmpFile, config_options, mpi_config
            )

        # Loop through all of the input forcings in NAM nest data. Convert the GRIB2 files
        # to NetCDF, read in the data, regrid it, then map it to the appropriate
        # array slice in the output arrays.
        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                pt.log_debug(f"Processing NAM Nest Variable: {grib_var}")

            calc_regrid_flag = check_regrid_status(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                wrf_hydro_geo_meta,
                mpi_config,
            )
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    pt.log_debug("Calculating NAM nest regridding weights....")
                calculate_weights(
                    id_tmp,
                    force_count,
                    input_forcings,
                    config_options,
                    mpi_config,
                    wrf_hydro_geo_meta,
                )
                err_handler.check_program_status(config_options, mpi_config)

                # Read in the RAP height field, which is used for downscaling purposes.
                # if mpi_config.rank == 0:
                #     config_options.statusMsg = "Reading in NAM nest elevation data from GRIB2."
                #     err_handler.log_msg(config_options, mpi_config, True)
                # cmd = "$WGRIB2 " + input_forcings.file_in2 + " -match " + \
                #       "\":(HGT):(surface):\" " + \
                #       " -netcdf " + input_forcings.tmpFileHeight
                # id_tmp_height = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFileHeight,
                #                                  cmd, config_options, mpi_config, 'HGT_surface')
                # err_handler.check_program_status(config_options, mpi_config)
                if config_options.grid_type == "gridded":
                    # Regrid the height variable.
                    if mpi_config.rank == 0:
                        var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place NetCDF NAM nest elevation data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding NAM nest elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid NAM nest elevation data to the WRF-Hydro domain using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to compute mask on NAM nest elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract ESMF regridded NAM nest elevation data to a local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "unstructured":
                    # Regrid the height variable.
                    if mpi_config.rank == 0:
                        var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place NetCDF NAM nest elevation data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding NAM nest elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid NAM nest elevation data to the WRF-Hydro domain using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to compute mask on NAM nest elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract ESMF regridded NAM nest elevation data to a local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Regrid the height variable.
                    if mpi_config.rank == 0:
                        var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp_elem = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(
                        input_forcings, var_tmp_elem, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place NetCDF NAM nest elevation data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding NAM nest elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out_elem = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj_elem,
                                input_forcings.esmf_field_in_elem,
                                input_forcings.esmf_field_out_elem,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid NAM nest elevation data to the WRF-Hydro domain using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out_elem.data[
                            np.where(input_forcings.regridded_mask_elem == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to compute mask on NAM nest elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = (
                            input_forcings.esmf_field_out_elem.data
                        )
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract ESMF regridded NAM nest elevation data to a local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    # Regrid the height variable.
                    if mpi_config.rank == 0:
                        var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place NetCDF NAM nest elevation data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding NAM nest elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid NAM nest elevation data to the WRF-Hydro domain using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to compute mask on NAM nest elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract ESMF regridded NAM nest elevation data to a local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                # Close the temporary NetCDF file and remove it.
                # if mpi_config.rank == 0:
                #     try:
                #         id_tmp_height.close()
                #     except OSError:
                #         config_options.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                #         err_handler.log_critical(config_options, mpi_config)
                #
                #     try:
                #         os_utils.os_remove_retry(input_forcings.tmpFileHeight)
                #     except OSError:
                #         config_options.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                #         err_handler.log_critical(config_options, mpi_config)
                # err_handler.check_program_status(config_options, mpi_config)

            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                # Regrid the input variables.
                var_tmp = None
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding NAM nest input variable: {input_forcings.netcdf_var_names[force_count]}"
                    )
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local array into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid input NAM nest forcing variables using ESMF: {ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to calculate mask from input NAM nest regridded forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local ESMF regridded data into local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                # Regrid the input variables.
                var_tmp = None
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding NAM nest input variable: {input_forcings.netcdf_var_names[force_count]}"
                    )
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local array into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid input NAM nest forcing variables using ESMF: {ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to calculate mask from input NAM nest regridded forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local ESMF regridded data into local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

                # Regrid the input variables.
                var_tmp_elem = None
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding NAM nest input variable: {input_forcings.netcdf_var_names[force_count]}"
                    )
                    try:
                        var_tmp_elem = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(
                    input_forcings, var_tmp_elem, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local array into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj_elem,
                            input_forcings.esmf_field_in_elem,
                            input_forcings.esmf_field_out_elem,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid input NAM nest forcing variables using ESMF: {ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out_elem.data[
                        np.where(input_forcings.regridded_mask_elem == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to calculate mask from input NAM nest regridded forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local ESMF regridded data into local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "hydrofabric":
                # Regrid the input variables.
                var_tmp = None
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding NAM nest input variable: {input_forcings.netcdf_var_names[force_count]}"
                    )
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local array into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid input NAM nest forcing variables using ESMF: {ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to calculate mask from input NAM nest regridded forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local ESMF regridded data into local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

    finally:
        pt.close_rank_0_partial(id_tmp)
        pt.os_remove_rank_0_partial(input_forcings.tmpFile)


def regrid_mrms_hourly(
    supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Rebgrid MRMS hourly precipitation.

    Function for handling regridding hourly MRMS precipitation. An RQI mask file
    Is necessary to filter out poor precipitation estimates.
    :param supplemental_precip:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # Do we want to use MRMS data at this timestep? If not, log and continue
    if not config_options.use_data_at_current_time:
        if mpi_config.rank == 0:
            pt.log_info("Exceeded max hours for MRMS precipitation")
        return

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(supplemental_precip.file_in2):
        # does file_in1 exist? (Pass1 vs Pass2 for MRMS, for example)
        if os.path.isfile(supplemental_precip.file_in1):
            supplemental_precip.file_in2 = supplemental_precip.file_in1
        else:
            return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if supplemental_precip.regridComplete:
        if mpi_config.rank == 0:
            pt.log_debug("No MRMS regridding required for this timestep.")
        return
    # MRMS data originally is stored as .gz files. We need to compose a series
    # of temporary paths.
    # 1.) The unzipped GRIB2 precipitation file.
    # 2.) The unzipped GRIB2 RQI file.
    # 3.) A temporary NetCDF file that stores the precipitation grid.
    # 4.) A temporary NetCDF file that stores the RQI grid.
    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.

    file_uuid = str(mpi_config.uid64)
    mrms_tmp_grib2 = str(
        Path(config_options.scratch_dir)
        / f"{file_uuid}_MRMS_PCP_TMP-{mkfilename()}.grib2"
    )
    mrms_tmp_nc = str(
        Path(config_options.scratch_dir) / f"{file_uuid}_MRMS_PCP_TMP-{mkfilename()}.nc"
    )
    mrms_tmp_rqi_grib2 = str(
        Path(config_options.scratch_dir)
        / f"{file_uuid}_MRMS_RQI_TMP-{mkfilename()}.grib2"
    )
    mrms_tmp_rqi_nc = str(
        Path(config_options.scratch_dir) / f"{file_uuid}_MRMS_RQI_TMP-{mkfilename()}.nc"
    )
    # mpi_config.comm.barrier()

    # If the input paths have been set to None, this means input is missing. We will
    # alert the user, and set the final output grids to be the global NDV and return.
    if not supplemental_precip.file_in1 or not supplemental_precip.file_in2:
        if mpi_config.rank == 0:
            pt.log_info(
                "No MRMS Precipitation available. Supplemental precipitation will not be used."
            )
        supplemental_precip.regridded_precip2 = None
        supplemental_precip.regridded_precip1 = None
        if config_options.grid_type == "unstructured":
            supplemental_precip.regridded_precip2_elem = None
            supplemental_precip.regridded_precip1_elem = None
        return

    err_handler.check_program_status(config_options, mpi_config)

    # If the input paths have been set to None, this means input is missing. We will
    # alert the user, and set the final output grids to be the global NDV and return.
    # if not supplemental_precip.file_in1 or not supplemental_precip.file_in2:
    #    if MpiConfig.rank == 0:
    #        ConfigOptions.statusMsg = "No MRMS Precipitation available. Supplemental precipitation will " \
    #                                  "not be used."
    #        errMod.log_msg(ConfigOptions, MpiConfig)
    #    supplemental_precip.regridded_precip2 = None
    #    supplemental_precip.regridded_precip1 = None
    #    return

    id_mrms = None
    id_mrms_rqi = None
    try:
        pt.log_info("Rrgrid MRMS")

        if supplemental_precip.file_type != NETCDF:
            # Unzip MRMS files to temporary locations.
            ioMod.unzip_file(
                supplemental_precip.file_in2, mrms_tmp_grib2, config_options, mpi_config
            )
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.rqiMethod == 1:
                ioMod.unzip_file(
                    supplemental_precip.rqi_file_in2,
                    mrms_tmp_rqi_grib2,
                    config_options,
                    mpi_config,
                )
                err_handler.check_program_status(config_options, mpi_config)

            # Perform a GRIB dump to NetCDF for the MRMS precip and RQI data.
            if WGRIB2_env:
                cmd1 = f"$WGRIB2 {mrms_tmp_grib2} -netcdf {mrms_tmp_nc}"
            else:
                cmd1 = mrms_tmp_grib2

            id_mrms = ioMod.open_grib2(
                mrms_tmp_grib2,
                mrms_tmp_nc,
                cmd1,
                config_options,
                mpi_config,
                supplemental_precip.netcdf_var_names[0],
                special_case=True,
            )
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.rqiMethod == 1:
                if WGRIB2_env:
                    cmd2 = f"$WGRIB2 {mrms_tmp_rqi_grib2} -netcdf {mrms_tmp_rqi_nc}"
                else:
                    cmd2 = mrms_tmp_rqi_grib2

                id_mrms_rqi = ioMod.open_grib2(
                    mrms_tmp_rqi_grib2,
                    mrms_tmp_rqi_nc,
                    cmd2,
                    config_options,
                    mpi_config,
                    supplemental_precip.rqi_netcdf_var_names[0],
                    special_case=True,
                )
                err_handler.check_program_status(config_options, mpi_config)
            else:
                id_mrms_rqi = None

        else:
            create_link(
                "MRMS",
                supplemental_precip.file_in2,
                mrms_tmp_nc,
                config_options,
                mpi_config,
            )
            id_mrms = ioMod.open_netcdf_forcing(mrms_tmp_nc, config_options, mpi_config)
            if supplemental_precip.rqiMethod == 1:
                create_link(
                    "RQI",
                    supplemental_precip.rqi_file_in2,
                    mrms_tmp_rqi_nc,
                    config_options,
                    mpi_config,
                )
                id_mrms_rqi = ioMod.open_netcdf_forcing(
                    mrms_tmp_rqi_nc, config_options, mpi_config
                )
            else:
                id_mrms_rqi = None

        # Check to see if we need to calculate regridding weights.
        calc_regrid_flag = check_supp_pcp_regrid_status(
            id_mrms, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
        )
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                pt.log_debug("Calculating MRMS regridding weights.")
            calculate_supp_pcp_weights(
                supplemental_precip, id_mrms, mrms_tmp_nc, config_options, mpi_config
            )
            err_handler.check_program_status(config_options, mpi_config)

        # Regrid the RQI grid.
        if supplemental_precip.rqiMethod == 1:
            if config_options.grid_type != "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp = id_mrms_rqi.variables[
                            supplemental_precip.rqi_netcdf_var_names[0]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {supplemental_precip.rqi_netcdf_var_names[0]} from: {mrms_tmp_rqi_grib2} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    supplemental_precip, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place MRMS data into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    pt.log_debug("Regridding MRMS RQI Field.")
                try:
                    supplemental_precip.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            supplemental_precip.regridObj,
                            supplemental_precip.esmf_field_in,
                            supplemental_precip.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(f"Unable to regrid MRMS RQI field: {ve}")
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    n_masked = len((supplemental_precip.regridded_mask == 0))
                    if n_masked > 0:
                        if mpi_config.rank == 0:
                            pt.log_debug(
                                f"{n_masked} masked cells in RQI field, will remove"
                            )

                    supplemental_precip.esmf_field_out.data[
                        np.where(supplemental_precip.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to run mask calculation for MRMS RQI data: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                var_tmp = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp = id_mrms_rqi.variables[
                            supplemental_precip.rqi_netcdf_var_names[0]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {supplemental_precip.rqi_netcdf_var_names[0]} from: {mrms_tmp_rqi_grib2} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    supplemental_precip, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place MRMS data into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    pt.log_debug("Regridding MRMS RQI Field.")
                try:
                    supplemental_precip.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            supplemental_precip.regridObj,
                            supplemental_precip.esmf_field_in,
                            supplemental_precip.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(f"Unable to regrid MRMS RQI field: {ve}")
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    n_masked = len((supplemental_precip.regridded_mask == 0))
                    if n_masked > 0:
                        if mpi_config.rank == 0:
                            pt.log_debug(
                                f"{n_masked} masked cells in RQI field, will remove"
                            )

                    supplemental_precip.esmf_field_out.data[
                        np.where(supplemental_precip.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to run mask calculation for MRMS RQI data: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                var_tmp_elem = None
                if mpi_config.rank == 0:
                    try:
                        var_tmp_elem = id_mrms_rqi.variables[
                            supplemental_precip.rqi_netcdf_var_names[0]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract: {supplemental_precip.rqi_netcdf_var_names[0]} from: {mrms_tmp_rqi_grib2} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(
                    supplemental_precip, var_tmp_elem, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place MRMS data into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    pt.log_debug("Regridding MRMS RQI Field.")
                try:
                    supplemental_precip.esmf_field_out_elem = (
                        pt.esmf_regridobj_call_retry_partial(
                            supplemental_precip.regridObj_elem,
                            supplemental_precip.esmf_field_in_elem,
                            supplemental_precip.esmf_field_out_elem,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(f"Unable to regrid MRMS RQI field: {ve}")
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    n_masked = len((supplemental_precip.regridded_mask_elem == 0))
                    if n_masked > 0:
                        if mpi_config.rank == 0:
                            pt.log_debug(
                                f"{n_masked} masked cells in RQI field, will remove"
                            )

                    supplemental_precip.esmf_field_out_elem.data[
                        np.where(supplemental_precip.regridded_mask_elem == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to run mask calculation for MRMS RQI data: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

        if not supplemental_precip.rqiMethod:
            # We will set the RQI field to 1.0 here so no MRMS data gets masked out.
            if config_options.grid_type == "gridded":
                supplemental_precip.regridded_rqi2[:, :] = 1.0
            elif config_options.grid_type == "unstructured":
                supplemental_precip.regridded_rqi2[:] = 1.0
                supplemental_precip.regridded_rqi2_elem[:] = 1.0
            elif config_options.grid_type == "hydrofabric":
                supplemental_precip.regridded_rqi2[:] = 1.0

            if mpi_config.rank == 0:
                pt.log_debug("MRMS Will not be filtered using RQI values.")

        elif supplemental_precip.rqiMethod == 2:
            # Read in the RQI field from monthly climatological files.
            ioMod.read_rqi_monthly_climo(
                config_options, mpi_config, supplemental_precip, wrf_hydro_geo_meta
            )
        elif supplemental_precip.rqiMethod == 1:
            # We are using the MRMS RQI field in realtime
            if config_options.grid_type == "gridded":
                supplemental_precip.regridded_rqi2[:, :] = (
                    supplemental_precip.esmf_field_out.data
                )
            elif config_options.grid_type == "unstructured":
                supplemental_precip.regridded_rqi2[:] = (
                    supplemental_precip.esmf_field_out.data
                )
                supplemental_precip.regridded_rqi2_elem[:] = (
                    supplemental_precip.esmf_field_out_elem.data
                )
            elif config_options.grid_type == "hydrofabric":
                supplemental_precip.regridded_rqi2[:] = (
                    supplemental_precip.esmf_field_out.data
                )

        err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding: {supplemental_precip.netcdf_var_names[0]}")
                try:
                    var_tmp = id_mrms.variables[
                        supplemental_precip.netcdf_var_names[0]
                    ][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract: {supplemental_precip.netcdf_var_names[0]} from: {mrms_tmp_nc} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj,
                        supplemental_precip.esmf_field_in,
                        supplemental_precip.esmf_field_out,
                    )
                )
            except ValueError as ve:
                pt.log_crit(f"Unable to regrid MRMS precipitation: {ve}")
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value, and set negative precip values to 0
            try:
                if len(np.argwhere(supplemental_precip.esmf_field_out.data < 0)) > 0:
                    supplemental_precip.esmf_field_out.data[
                        np.where(supplemental_precip.esmf_field_out.data < 0)
                    ] = config_options.globalNdv
                    # config_options.statusMsg = "WARNING: Found negative precipitation values in MRMS data, setting to 0"
                    # err_handler.log_warning(config_options, mpi_config)

                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv

            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on MRMS supplemental precip: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:, :] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.rqiMethod > 0:
                # Check for any RQI values below the threshold specified by the user.
                # Set these values to global NDV.
                try:
                    ind_filter = np.where(
                        supplemental_precip.regridded_rqi2
                        < supplemental_precip.rqiThresh
                    )
                    if len(ind_filter) > 0:
                        if mpi_config.rank == 0:
                            pt.log_debug(
                                f"Removing {len(ind_filter)} MRMS cells below RQI threshold of {supplemental_precip.rqiThresh}"
                            )
                    supplemental_precip.regridded_precip2[ind_filter] = (
                        config_options.globalNdv
                    )
                    del ind_filter
                except (ValueError, AttributeError, KeyError, ArithmeticError) as npe:
                    pt.log_crit(f"Unable to run MRMS RQI threshold search: {npe}")
                err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.keyValue != 14:
                # Convert the hourly precipitation total to a rate of mm/s
                try:
                    ind_valid = np.where(
                        supplemental_precip.regridded_precip2
                        != config_options.globalNdv
                    )
                    supplemental_precip.regridded_precip2[ind_valid] = (
                        supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                    )
                    del ind_valid
                except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
                    pt.log_crit(
                        f"Unable to run global NDV search on MRMS regridded precip: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:, :] = (
                    supplemental_precip.regridded_precip2[:, :]
                )
                supplemental_precip.regridded_rqi1[:, :] = (
                    supplemental_precip.regridded_rqi2[:, :]
                )
        # mpi_config.comm.barrier()

        elif config_options.grid_type == "unstructured":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding: {supplemental_precip.netcdf_var_names[0]}")
                try:
                    var_tmp = id_mrms.variables[
                        supplemental_precip.netcdf_var_names[0]
                    ][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract: {supplemental_precip.netcdf_var_names[0]} from: {mrms_tmp_nc} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj,
                        supplemental_precip.esmf_field_in,
                        supplemental_precip.esmf_field_out,
                    )
                )
            except ValueError as ve:
                pt.log_crit(f"Unable to regrid MRMS precipitation: {ve}")
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value, and set negative precip values to 0
            try:
                if len(np.argwhere(supplemental_precip.esmf_field_out.data < 0)) > 0:
                    supplemental_precip.esmf_field_out.data[
                        np.where(supplemental_precip.esmf_field_out.data < 0)
                    ] = config_options.globalNdv
                    # config_options.statusMsg = "WARNING: Found negative precipitation values in MRMS data, setting to 0"
                    # err_handler.log_warning(config_options, mpi_config)

                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv

            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on MRMS supplemental precip: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.rqiMethod > 0:
                # Check for any RQI values below the threshold specified by the user.
                # Set these values to global NDV.
                try:
                    ind_filter = np.where(
                        supplemental_precip.regridded_rqi2
                        < supplemental_precip.rqiThresh
                    )
                    if len(ind_filter) > 0:
                        if mpi_config.rank == 0:
                            pt.log_debug(
                                f"Removing {len(ind_filter)} MRMS cells below RQI threshold of {supplemental_precip.rqiThresh}"
                            )
                    supplemental_precip.regridded_precip2[ind_filter] = (
                        config_options.globalNdv
                    )
                    del ind_filter
                except (ValueError, AttributeError, KeyError, ArithmeticError) as npe:
                    pt.log_crit(f"Unable to run MRMS RQI threshold search: {npe}")
                err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.keyValue != 14:
                # Convert the hourly precipitation total to a rate of mm/s
                try:
                    ind_valid = np.where(
                        supplemental_precip.regridded_precip2
                        != config_options.globalNdv
                    )
                    supplemental_precip.regridded_precip2[ind_valid] = (
                        supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                    )
                    del ind_valid
                except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
                    pt.log_crit(
                        f"Unable to run global NDV search on MRMS regridded precip: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = (
                    supplemental_precip.regridded_precip2[:]
                )
                supplemental_precip.regridded_rqi1[:] = (
                    supplemental_precip.regridded_rqi2[:]
                )
            # mpi_config.comm.barrier()

            # Regrid the input variables.
            var_tmp_elem = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding: {supplemental_precip.netcdf_var_names[0]}")
                try:
                    var_tmp_elem = id_mrms.variables[
                        supplemental_precip.netcdf_var_names[0]
                    ][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract: {supplemental_precip.netcdf_var_names[0]} from: {mrms_tmp_nc} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(
                supplemental_precip, var_tmp_elem, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj_elem,
                        supplemental_precip.esmf_field_in_elem,
                        supplemental_precip.esmf_field_out_elem,
                    )
                )
            except ValueError as ve:
                pt.log_crit(f"Unable to regrid MRMS precipitation: {ve}")
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value, and set negative precip values to 0
            try:
                supplemental_precip.esmf_field_out_elem.data[
                    np.where(supplemental_precip.regridded_mask_elem == 0)
                ] = config_options.globalNdv

                if (
                    len(np.argwhere(supplemental_precip.esmf_field_out_elem.data < 0))
                    > 0
                ):
                    supplemental_precip.esmf_field_out_elem.data[
                        np.where(supplemental_precip.esmf_field_out_elem.data < 0)
                    ] = config_options.globalNdv
                    # config_options.statusMsg = "WARNING: Found negative precipitation values in MRMS data, setting to 0"
                    # err_handler.log_warning(config_options, mpi_config)

            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on MRMS supplemental precip: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2_elem[:] = (
                supplemental_precip.esmf_field_out_elem.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.rqiMethod > 0:
                # Check for any RQI values below the threshold specified by the user.
                # Set these values to global NDV.
                try:
                    ind_filter_elem = np.where(
                        supplemental_precip.regridded_rqi2_elem
                        < supplemental_precip.rqiThresh
                    )
                    if len(ind_filter_elem) > 0:
                        if mpi_config.rank == 0:
                            pt.log_debug(
                                f"Removing {len(ind_filter)} MRMS cells below RQI threshold of {supplemental_precip.rqiThresh}"
                            )
                    supplemental_precip.regridded_precip2_elem[ind_filter_elem] = (
                        config_options.globalNdv
                    )
                    del ind_filter_elem
                except (ValueError, AttributeError, KeyError, ArithmeticError) as npe:
                    pt.log_crit(f"Unable to run MRMS RQI threshold search: {npe}")
                err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total to a rate of mm/s
            try:
                ind_valid_elem = np.where(
                    supplemental_precip.regridded_precip2_elem
                    != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2_elem[ind_valid_elem] = (
                    supplemental_precip.regridded_precip2_elem[ind_valid_elem] / 3600.0
                )
                del ind_valid_elem
            except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
                pt.log_crit(
                    f"Unable to run global NDV search on MRMS regridded precip: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1_elem[:] = (
                    supplemental_precip.regridded_precip2_elem[:]
                )
                supplemental_precip.regridded_rqi1_elem[:] = (
                    supplemental_precip.regridded_rqi2_elem[:]
                )
        # mpi_config.comm.barrier()

        elif config_options.grid_type == "hydrofabric":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding: {supplemental_precip.netcdf_var_names[0]}")
                try:
                    var_tmp = id_mrms.variables[
                        supplemental_precip.netcdf_var_names[0]
                    ][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract: {supplemental_precip.netcdf_var_names[0]} from: {mrms_tmp_nc} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj,
                        supplemental_precip.esmf_field_in,
                        supplemental_precip.esmf_field_out,
                    )
                )
            except ValueError as ve:
                pt.log_crit(f"Unable to regrid MRMS precipitation: {ve}")
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value, and set negative precip values to 0
            try:
                if len(np.argwhere(supplemental_precip.esmf_field_out.data < 0)) > 0:
                    supplemental_precip.esmf_field_out.data[
                        np.where(supplemental_precip.esmf_field_out.data < 0)
                    ] = config_options.globalNdv
                    # config_options.statusMsg = "WARNING: Found negative precipitation values in MRMS data, setting to 0"
                    # err_handler.log_warning(config_options, mpi_config)

                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv

            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on MRMS supplemental precip: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.rqiMethod > 0:
                # Check for any RQI values below the threshold specified by the user.
                # Set these values to global NDV.
                try:
                    ind_filter = np.where(
                        supplemental_precip.regridded_rqi2
                        < supplemental_precip.rqiThresh
                    )
                    if len(ind_filter) > 0:
                        if mpi_config.rank == 0:
                            pt.log_debug(
                                f"Removing {len(ind_filter)} MRMS cells below RQI threshold of {supplemental_precip.rqiThresh}"
                            )
                    supplemental_precip.regridded_precip2[ind_filter] = (
                        config_options.globalNdv
                    )
                    del ind_filter
                except (ValueError, AttributeError, KeyError, ArithmeticError) as npe:
                    pt.log_crit(f"Unable to run MRMS RQI threshold search: {npe}")
                err_handler.check_program_status(config_options, mpi_config)

            if supplemental_precip.keyValue != 14:
                # Convert the hourly precipitation total to a rate of mm/s
                try:
                    ind_valid = np.where(
                        supplemental_precip.regridded_precip2
                        != config_options.globalNdv
                    )
                    supplemental_precip.regridded_precip2[ind_valid] = (
                        supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                    )
                    del ind_valid
                except (ValueError, AttributeError, ArithmeticError, KeyError) as npe:
                    pt.log_crit(
                        f"Unable to run global NDV search on MRMS regridded precip: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = (
                    supplemental_precip.regridded_precip2[:]
                )
                supplemental_precip.regridded_rqi1[:] = (
                    supplemental_precip.regridded_rqi2[:]
                )
        # mpi_config.comm.barrier()

    finally:
        # Close whichever file handles got opened
        if id_mrms is not None:
            try:
                id_mrms.close()
            except OSError:
                pt.log_crit(f"Unable to close NetCDF file: {mrms_tmp_nc}")

        if id_mrms_rqi is not None:
            try:
                id_mrms_rqi.close()
            except OSError:
                pt.log_crit(f"Unable to close NetCDF file: {mrms_tmp_rqi_nc}")

        for f in (mrms_tmp_grib2, mrms_tmp_nc, mrms_tmp_rqi_grib2, mrms_tmp_rqi_nc):
            pt.os_remove_rank_0_partial(f)
        mpi_config.comm.barrier()
        err_handler.check_program_status(config_options, mpi_config)


def regrid_mrms_precip_flag(
    supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Rebgrid MRMS hourly precipitation flag.

    Function for handling regridding of SBCv2 Liquid Water Precip forcing files.
    :param supplemental_precip:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.exists(supplemental_precip.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if supplemental_precip.regridComplete:
        return

    # Unzip MRMS precip flag file to temporary location.
    file_name = f"_MRMS_PCP_FLAG_TMP_{mkfilename()}"
    file_uuid = str(mpi_config.uid64)
    mrms_tmp_grib2 = str(
        Path(config_options.scratch_dir) / f"{file_uuid}{file_name}.grib2"
    )
    mrms_tmp_nc = str(Path(config_options.scratch_dir) / f"{file_uuid}{file_name}.nc")

    ioMod.unzip_file(
        supplemental_precip.file_in2, mrms_tmp_grib2, config_options, mpi_config
    )
    err_handler.check_program_status(config_options, mpi_config)

    # Perform a GRIB dump to NetCDF for the MRMS precip and RQI data.
    cmd1 = f"$WGRIB2 {mrms_tmp_grib2} -netcdf {mrms_tmp_nc}"
    id_tmp = ioMod.open_grib2(
        mrms_tmp_grib2,
        mrms_tmp_nc,
        cmd1,
        config_options,
        mpi_config,
        supplemental_precip.netcdf_var_names[0],
    )
    err_handler.check_program_status(config_options, mpi_config)

    # Check to see if we need to calculate regridding weights.
    calc_regrid_flag = check_supp_pcp_regrid_status(
        id_tmp, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
    )
    err_handler.check_program_status(config_options, mpi_config)

    if calc_regrid_flag:
        if mpi_config.rank == 0:
            pt.log_info("Calculating MRMS PrecipFlag regridding weights.")
        calculate_supp_pcp_weights(
            supplemental_precip,
            id_tmp,
            supplemental_precip.file_in2,
            config_options,
            mpi_config,
        )
        err_handler.check_program_status(config_options, mpi_config)

    # Regrid the input variable
    var_tmp = None
    if mpi_config.rank == 0:
        if mpi_config.rank == 0:
            pt.log_info("Regridding MRMS PrecipFlag Fraction.")
        try:
            var_tmp = id_tmp.variables[supplemental_precip.netcdf_var_names[0]][0, :, :]
        except (ValueError, KeyError, AttributeError) as err:
            pt.log_crit(
                f"Unable to extract PrecipFlag from file: {supplemental_precip.file_in2} ({err})"
            )
    err_handler.check_program_status(config_options, mpi_config)

    var_sub_tmp = mpi_config.scatter_array(supplemental_precip, var_tmp, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        var_sub_tmp[var_sub_tmp <= 0] = 1.0  # all liquid if no other category
        var_sub_tmp[var_sub_tmp == 3] = 0.0  # snow
        var_sub_tmp[var_sub_tmp == 7] = 0.0  # hail
        var_sub_tmp[var_sub_tmp > 0] = 1.0  # all other liquid categories

        supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
    except (ValueError, KeyError, AttributeError) as err:
        pt.log_crit(f"Unable to place MRMS PrecipFlag into local ESMF field: {err}")
    err_handler.check_program_status(config_options, mpi_config)

    try:
        supplemental_precip.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
            supplemental_precip.regridObj,
            supplemental_precip.esmf_field_in,
            supplemental_precip.esmf_field_out,
        )
    except ValueError as ve:
        pt.log_crit(f"Unable to regrid MRMS PrecipFlag: {ve}")
    err_handler.check_program_status(config_options, mpi_config)

    # Set any missing data or pixel cells outside the input domain to a default of 100%
    try:
        supplemental_precip.esmf_field_out.data[
            np.where(supplemental_precip.regridded_mask == 0)
        ] = 1.0
        supplemental_precip.esmf_field_out.data[
            np.where(supplemental_precip.esmf_field_out.data < 0)
        ] = 1.0
    except (ValueError, ArithmeticError) as npe:
        pt.log_crit(f"Unable to run mask search on MRMS PrecipFlag: {npe}")
    err_handler.check_program_status(config_options, mpi_config)

    supplemental_precip.regridded_precip2[:] = supplemental_precip.esmf_field_out.data
    err_handler.check_program_status(config_options, mpi_config)

    # If we are on the first timestep, set the previous regridded field to be
    # the latest as there are no states for time 0.
    if config_options.current_output_step == 1:
        supplemental_precip.regridded_precip1[:] = (
            supplemental_precip.regridded_precip2[:]
        )
    err_handler.check_program_status(config_options, mpi_config)

    if config_options.grid_type == "unstructured":
        # Regrid the input variable
        var_tmp_elem = None
        if mpi_config.rank == 0:
            if mpi_config.rank == 0:
                pt.log_info("Regridding MRMS PrecipFlag Fraction.")
            try:
                var_tmp_elem = id_tmp.variables[
                    supplemental_precip.netcdf_var_names[0]
                ][0, :, :]
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to extract PrecipFlag from file: {supplemental_precip.file_in2} ({err})"
                )
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp_elem = mpi_config.scatter_array(
            supplemental_precip, var_tmp_elem, config_options
        )
        err_handler.check_program_status(config_options, mpi_config)

        try:
            var_sub_tmp_elem[var_sub_tmp_elem <= 0] = (
                1.0  # all liquid if no other category
            )
            var_sub_tmp_elem[var_sub_tmp_elem == 3] = 0.0  # snow
            var_sub_tmp_elem[var_sub_tmp_elem == 7] = 0.0  # hail
            var_sub_tmp_elem[var_sub_tmp_elem > 0] = 1.0  # all other liquid categories

            supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
        except (ValueError, KeyError, AttributeError) as err:
            pt.log_crit(f"Unable to place MRMS PrecipFlag into local ESMF field: {err}")
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_precip.esmf_field_out_elem = (
                pt.esmf_regridobj_call_retry_partial(
                    supplemental_precip.regridObj_elem,
                    supplemental_precip.esmf_field_in_elem,
                    supplemental_precip.esmf_field_out_elem,
                )
            )
        except ValueError as ve:
            pt.log_crit(f"Unable to regrid MRMS PrecipFlag: {ve}")
        err_handler.check_program_status(config_options, mpi_config)

        # Set any missing data or pixel cells outside the input domain to a default of 100%
        try:
            supplemental_precip.esmf_field_out_elem.data[
                np.where(supplemental_precip.regridded_mask_elem == 0)
            ] = 1.0
            supplemental_precip.esmf_field_out_elem.data[
                np.where(supplemental_precip.esmf_field_out_elem.data < 0)
            ] = 1.0
        except (ValueError, ArithmeticError) as npe:
            pt.log_crit(f"Unable to run mask search on MRMS PrecipFlag: {npe}")
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_precip.regridded_precip2_elem[:] = (
            supplemental_precip.esmf_field_out_elem.data
        )
        err_handler.check_program_status(config_options, mpi_config)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if config_options.current_output_step == 1:
            supplemental_precip.regridded_precip1_elem[:] = (
                supplemental_precip.regridded_precip2_elem[:]
            )
        err_handler.check_program_status(config_options, mpi_config)

    pt.close_rank_0_partial(id_tmp)
    pt.os_remove_rank_0_partial(mrms_tmp_nc)


def regrid_hourly_wrf_arw(
    input_forcings, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Regrid NAM Nest Data.

    Function for handing regridding of input NAM nest data
    fro GRIB2 files.
    :param mpi_config:
    :param wrf_hydro_geo_meta:
    :param input_forcings:
    :param config_options:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        pt.log_debug(
            "No regridding of WRF-ARW nest data necessary for this timestep - already completed."
        )
        return

    # Create a path for a temporary NetCDF file

    file_name = f"ARW_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    input_forcings.tmpFile = str(
        Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}"
    )

    err_handler.check_program_status(config_options, mpi_config)

    id_tmp = None
    try:
        pt.log_info("Regrid WRF-ARW nest data")

        if input_forcings.input_force_types != NETCDF:
            pt.os_remove_rank_0_partial(input_forcings.tmpFile)

            fields = []
            for force_count, grib_var in enumerate(input_forcings.grib_vars):
                if mpi_config.rank == 0:
                    pt.log_debug(f"Converting WRF-ARW Variable: {grib_var}")
                time_str = (
                    f"{input_forcings.fcst_hour1}-{input_forcings.fcst_hour2} hour acc fcst"
                    if grib_var == "APCP"
                    else f"{input_forcings.fcst_hour2} hour fcst"
                )
                fields.append(
                    f":{grib_var}:{input_forcings.grib_levels[force_count]}:{time_str}:"
                )
            fields.append(":(HGT):(surface):")

            # Create a temporary NetCDF file from the GRIB2 file.
            if WGRIB2_env:
                cmd = f'$WGRIB2 -match "({"|".join(fields)})" {input_forcings.file_in2} -netcdf {input_forcings.tmpFile}'
            else:
                cmd = f"({'|'.join(fields)})"

            id_tmp = ioMod.open_grib2(
                input_forcings.file_in2,
                input_forcings.tmpFile,
                cmd,
                config_options,
                mpi_config,
                inputVar=None,
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link(
                "WRF-ARW",
                input_forcings.file_in2,
                input_forcings.tmpFile,
                config_options,
                mpi_config,
            )
            id_tmp = ioMod.open_netcdf_forcing(
                input_forcings.tmpFile, config_options, mpi_config
            )

        # Loop through all of the input forcings in NAM nest data. Convert the GRIB2 files
        # to NetCDF, read in the data, regrid it, then map it to the appropriate
        # array slice in the output arrays.
        for force_count, grib_var in enumerate(input_forcings.grib_vars):
            if mpi_config.rank == 0:
                pt.log_debug(f"Processing WRF-ARW Variable: {grib_var}")

            calc_regrid_flag = check_regrid_status(
                id_tmp,
                force_count,
                input_forcings,
                config_options,
                wrf_hydro_geo_meta,
                mpi_config,
            )
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    pt.log_debug("Calculating WRF-ARW regridding weights....")
                calculate_weights(
                    id_tmp,
                    force_count,
                    input_forcings,
                    config_options,
                    mpi_config,
                    wrf_hydro_geo_meta,
                )
                err_handler.check_program_status(config_options, mpi_config)

                # Read in the RAP height field, which is used for downscaling purposes.
                # if mpi_config.rank == 0:
                #     config_options.statusMsg = "Reading in WRF-ARW elevation data from GRIB2."
                #     err_handler.log_msg(config_options, mpi_config, True)
                # cmd = "$WGRIB2 " + input_forcings.file_in2 + " -match " + \
                #       "\":(HGT):(surface):\" " + \
                #       " -netcdf " + input_forcings.tmpFileHeight
                # id_tmp_height = ioMod.open_grib2(input_forcings.file_in2, input_forcings.tmpFileHeight,
                #                                  cmd, config_options, mpi_config, 'HGT_surface')
                # err_handler.check_program_status(config_options, mpi_config)
                if config_options.grid_type == "gridded":
                    # Regrid the height variable.
                    if mpi_config.rank == 0:
                        var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place NetCDF WRF-ARW elevation data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding WRF-ARW elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid WRF-ARW elevation data to the WRF-Hydro domain using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to compute mask on WRF-ARW elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract ESMF regridded WRF-ARW elevation data to a local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)
                elif config_options.grid_type == "unstructured":
                    # Regrid the height variable.
                    if mpi_config.rank == 0:
                        var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place NetCDF WRF-ARW elevation data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding WRF-ARW elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid WRF-ARW elevation data to the WRF-Hydro domain using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to compute mask on WRF-ARW elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract ESMF regridded WRF-ARW elevation data to a local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Regrid the height variable.
                    if mpi_config.rank == 0:
                        var_tmp_elem = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp_elem = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp_elem = mpi_config.scatter_array(
                        input_forcings, var_tmp_elem, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place NetCDF WRF-ARW elevation data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding WRF-ARW elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out_elem = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj_elem,
                                input_forcings.esmf_field_in_elem,
                                input_forcings.esmf_field_out_elem,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid WRF-ARW elevation data to the WRF-Hydro domain using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out_elem.data[
                            np.where(input_forcings.regridded_mask_elem == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to compute mask on WRF-ARW elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height_elem[:] = (
                            input_forcings.esmf_field_out_elem.data
                        )
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract ESMF regridded WRF-ARW elevation data to a local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                elif config_options.grid_type == "hydrofabric":
                    # Regrid the height variable.
                    if mpi_config.rank == 0:
                        var_tmp = id_tmp.variables["HGT_surface"][0, :, :]
                    else:
                        var_tmp = None
                    err_handler.check_program_status(config_options, mpi_config)

                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place NetCDF WRF-ARW elevation data into the ESMF field object: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    if mpi_config.rank == 0:
                        pt.log_debug(
                            "Regridding WRF-ARW elevation data to the WRF-Hydro domain."
                        )
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid WRF-ARW elevation data to the WRF-Hydro domain using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = config_options.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to compute mask on WRF-ARW elevation data: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    try:
                        input_forcings.height[:] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract ESMF regridded WRF-ARW elevation data to a local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                # Close the temporary NetCDF file and remove it.
                # if mpi_config.rank == 0:
                #     try:
                #         id_tmp_height.close()
                #     except OSError:
                #         config_options.errMsg = "Unable to close temporary file: " + input_forcings.tmpFileHeight
                #         err_handler.log_critical(config_options, mpi_config)
                #
                #     try:
                #         os_utils.os_remove_retry(input_forcings.tmpFileHeight)
                #     except OSError:
                #         config_options.errMsg = "Unable to remove temporary file: " + input_forcings.tmpFileHeight
                #         err_handler.log_critical(config_options, mpi_config)
                # err_handler.check_program_status(config_options, mpi_config)

            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "gridded":
                # Regrid the input variables.
                var_tmp = None
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding WRF-ARW input variable: {input_forcings.netcdf_var_names[force_count]}"
                    )
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local array into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid input WRF-ARW forcing variables using ESMF: {ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to calculate mask from input WRF-ARW regridded forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Convert the hourly precipitation total to a rate of mm/s
                if grib_var == "APCP":
                    try:
                        ind_valid = np.where(
                            input_forcings.esmf_field_out.data
                            != config_options.globalNdv
                        )
                        input_forcings.esmf_field_out.data[ind_valid] = (
                            input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        )
                        del ind_valid
                    except (
                        ValueError,
                        ArithmeticError,
                        AttributeError,
                        KeyError,
                    ) as npe:
                        pt.log_crit(
                            f"Unable to run NDV search on WRF ARW precipitation: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local ESMF regridded data into local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                # Regrid the input variables.
                var_tmp = None
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding WRF-ARW input variable: {input_forcings.netcdf_var_names[force_count]}"
                    )
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local array into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid input WRF-ARW forcing variables using ESMF: {ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to calculate mask from input WRF-ARW regridded forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Convert the hourly precipitation total to a rate of mm/s
                if grib_var == "APCP":
                    try:
                        ind_valid = np.where(
                            input_forcings.esmf_field_out.data
                            != config_options.globalNdv
                        )
                        input_forcings.esmf_field_out.data[ind_valid] = (
                            input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        )
                        del ind_valid
                    except (
                        ValueError,
                        ArithmeticError,
                        AttributeError,
                        KeyError,
                    ) as npe:
                        pt.log_crit(
                            f"Unable to run NDV search on WRF ARW precipitation: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local ESMF regridded data into local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

                # Regrid the input variables.
                var_tmp_elem = None
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding WRF-ARW input variable: {input_forcings.netcdf_var_names[force_count]}"
                    )
                    try:
                        var_tmp_elem = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(
                    input_forcings, var_tmp_elem, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local array into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj_elem,
                            input_forcings.esmf_field_in_elem,
                            input_forcings.esmf_field_out_elem,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid input WRF-ARW forcing variables using ESMF: {ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out_elem.data[
                        np.where(input_forcings.regridded_mask_elem == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to calculate mask from input WRF-ARW regridded forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Convert the hourly precipitation total to a rate of mm/s
                if grib_var == "APCP":
                    try:
                        ind_valid_elem = np.where(
                            input_forcings.esmf_field_out_elem.data
                            != config_options.globalNdv
                        )
                        input_forcings.esmf_field_out_elem.data[ind_valid_elem] = (
                            input_forcings.esmf_field_out_elem.data[ind_valid_elem]
                            / 3600.0
                        )
                        del ind_valid_elem
                    except (
                        ValueError,
                        ArithmeticError,
                        AttributeError,
                        KeyError,
                    ) as npe:
                        pt.log_crit(
                            f"Unable to run NDV search on WRF ARW precipitation: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local ESMF regridded data into local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "hydrofabric":
                # Regrid the input variables.
                var_tmp = None
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"Regridding WRF-ARW input variable: {input_forcings.netcdf_var_names[force_count]}"
                    )
                    try:
                        var_tmp = id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to extract {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local array into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid input WRF-ARW forcing variables using ESMF: {ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to calculate mask from input WRF-ARW regridded forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Convert the hourly precipitation total to a rate of mm/s
                if grib_var == "APCP":
                    try:
                        ind_valid = np.where(
                            input_forcings.esmf_field_out.data
                            != config_options.globalNdv
                        )
                        input_forcings.esmf_field_out.data[ind_valid] = (
                            input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        )
                        del ind_valid
                    except (
                        ValueError,
                        ArithmeticError,
                        AttributeError,
                        KeyError,
                    ) as npe:
                        pt.log_crit(
                            f"Unable to run NDV search on WRF ARW precipitation: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local ESMF regridded data into local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

    finally:
        pt.close_rank_0_partial(id_tmp)
        pt.os_remove_rank_0_partial(input_forcings.tmpFile)
    err_handler.check_program_status(config_options, mpi_config)


def regrid_hourly_wrf_arw_hi_res_pcp(
    supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Rebgrid hourly WRF-ARW hi-res nest precipitation.

    Function for handling regridding hourly forecasted ARW precipitation for hi-res nests.
    :param supplemental_precip:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.exists(supplemental_precip.file_in1):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if supplemental_precip.regridComplete:
        if mpi_config.rank == 0:
            pt.log_debug("No ARW regridding required for this timestep.")
        return

    # Create a path for a temporary NetCDF files that will
    # be created through the wgrib2 process.
    file_name = f"ARW_PCP_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    arw_tmp_nc = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    id_tmp = None
    try:
        pt.log_info("Regrid ARW")

        if supplemental_precip.file_type != NETCDF:
            pt.os_remove_rank_0_partial(arw_tmp_nc)

            # If the input paths have been set to None, this means input is missing. We will
            # alert the user, and set the final output grids to be the global NDV and return.
            # if not supplemental_precip.file_in1 or not supplemental_precip.file_in1:
            #    if Mpi6366:18Config.rank == 0:
            #        "NO ARW PRECIP AVAILABLE. SETTING FINAL SUPP GRIDS TO NDV"
            #    supplemental_precip.regridded_precip2 = None
            #    supplemental_precip.regridded_precip1 = None
            #    return
            # errMod.check_program_status(ConfigOptions, MpiConfig)

            # Create a temporary NetCDF file from the GRIB2 file.
            if WGRIB2_env:
                cmd = f'$WGRIB2 {supplemental_precip.file_in1} -match ":(APCP):(surface):({supplemental_precip.fcst_hour1 - 1}-{supplemental_precip.fcst_hour1} hour acc fcst):" -netcdf {arw_tmp_nc}'
            else:
                cmd = f"(APCP):(surface):({supplemental_precip.fcst_hour1 - 1}-{supplemental_precip.fcst_hour1} hour acc fcst)"

            id_tmp = ioMod.open_grib2(
                supplemental_precip.file_in1,
                arw_tmp_nc,
                cmd,
                config_options,
                mpi_config,
                "APCP_surface",
                special_case=False,
            )
            err_handler.check_program_status(config_options, mpi_config)
        else:
            create_link(
                "ARW-PCP",
                supplemental_precip.file_in1,
                arw_tmp_nc,
                config_options,
                mpi_config,
            )
            id_tmp = ioMod.open_netcdf_forcing(arw_tmp_nc, config_options, mpi_config)

        # Check to see if we need to calculate regridding weights.
        calc_regrid_flag = check_supp_pcp_regrid_status(
            id_tmp, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
        )
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if mpi_config.rank == 0:
                pt.log_debug("Calculating WRF ARW regridding weights.")
            calculate_supp_pcp_weights(
                supplemental_precip, id_tmp, arw_tmp_nc, config_options, mpi_config
            )
            err_handler.check_program_status(config_options, mpi_config)

        if config_options.grid_type == "gridded":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    pt.log_debug("Regridding WRF ARW APCP Precipitation.")
                try:
                    var_tmp = id_tmp.variables["APCP_surface"][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract precipitation from WRF ARW file: {supplemental_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place WRF ARW precipitation into local ESMF field: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj,
                        supplemental_precip.esmf_field_in,
                        supplemental_precip.esmf_field_out,
                    )
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid WRF ARW supplemental precipitation: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on WRF ARW supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:, :] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total to a rate of mm/s
            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2 != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2[ind_valid] = (
                    supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                pt.log_crit(
                    f"Unable to run NDV search on WRF ARW supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:, :] = (
                    supplemental_precip.regridded_precip2[:, :]
                )
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "unstructured":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    pt.log_debug("Regridding WRF ARW APCP Precipitation.")
                try:
                    var_tmp = id_tmp.variables["APCP_surface"][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract precipitation from WRF ARW file: {supplemental_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place WRF ARW precipitation into local ESMF field: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj,
                        supplemental_precip.esmf_field_in,
                        supplemental_precip.esmf_field_out,
                    )
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid WRF ARW supplemental precipitation: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on WRF ARW supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total to a rate of mm/s
            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2 != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2[ind_valid] = (
                    supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                pt.log_crit(
                    f"Unable to run NDV search on WRF ARW supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = (
                    supplemental_precip.regridded_precip2[:]
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the input variables.
            var_tmp_elem = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    pt.log_debug("Regridding WRF ARW APCP Precipitation.")
                try:
                    var_tmp_elem = id_tmp.variables["APCP_surface"][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract precipitation from WRF ARW file: {supplemental_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(
                supplemental_precip, var_tmp_elem, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place WRF ARW precipitation into local ESMF field: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out_elem = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj_elem,
                        supplemental_precip.esmf_field_in_elem,
                        supplemental_precip.esmf_field_out_elem,
                    )
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid WRF ARW supplemental precipitation: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                supplemental_precip.esmf_field_out_elem.data[
                    np.where(supplemental_precip.regridded_mask_elem == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on WRF ARW supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2_elem[:] = (
                supplemental_precip.esmf_field_out_elem.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total to a rate of mm/s
            try:
                ind_valid_elem = np.where(
                    supplemental_precip.regridded_precip2_elem
                    != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2_elem[ind_valid_elem] = (
                    supplemental_precip.regridded_precip2_elem[ind_valid_elem] / 3600.0
                )
                del ind_valid_elem
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                pt.log_crit(
                    f"Unable to run NDV search on WRF ARW supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1_elem[:] = (
                    supplemental_precip.regridded_precip2_elem[:]
                )
            err_handler.check_program_status(config_options, mpi_config)

        elif config_options.grid_type == "hydrofabric":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                if mpi_config.rank == 0:
                    pt.log_debug("Regridding WRF ARW APCP Precipitation.")
                try:
                    var_tmp = id_tmp.variables["APCP_surface"][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract precipitation from WRF ARW file: {supplemental_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                supplemental_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place WRF ARW precipitation into local ESMF field: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                supplemental_precip.esmf_field_out = (
                    pt.esmf_regridobj_call_retry_partial(
                        supplemental_precip.regridObj,
                        supplemental_precip.esmf_field_in,
                        supplemental_precip.esmf_field_out,
                    )
                )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid WRF ARW supplemental precipitation: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                supplemental_precip.esmf_field_out.data[
                    np.where(supplemental_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on WRF ARW supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            supplemental_precip.regridded_precip2[:] = (
                supplemental_precip.esmf_field_out.data
            )
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total to a rate of mm/s
            try:
                ind_valid = np.where(
                    supplemental_precip.regridded_precip2 != config_options.globalNdv
                )
                supplemental_precip.regridded_precip2[ind_valid] = (
                    supplemental_precip.regridded_precip2[ind_valid] / 3600.0
                )
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                pt.log_crit(
                    f"Unable to run NDV search on WRF ARW supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                supplemental_precip.regridded_precip1[:] = (
                    supplemental_precip.regridded_precip2[:]
                )
            err_handler.check_program_status(config_options, mpi_config)

    finally:
        pt.close_rank_0_partial(id_tmp)
        pt.os_remove_rank_0_partial(arw_tmp_nc)


def regrid_sbcv2_liquid_water_fraction(
    supplemental_forcings, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Rebgrid SBCv2 Liquid Water Fraction.

    Function for handling regridding of SBCv2 Liquid Water Precip forcing files.
    :param supplemental_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.exists(supplemental_forcings.file_in1):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if supplemental_forcings.regridComplete:
        return

    id_tmp = ioMod.open_netcdf_forcing(
        supplemental_forcings.file_in1, config_options, mpi_config
    )

    # Check to see if we need to calculate regridding weights.
    calc_regrid_flag = check_supp_pcp_regrid_status(
        id_tmp, supplemental_forcings, config_options, wrf_hydro_geo_meta, mpi_config
    )
    err_handler.check_program_status(config_options, mpi_config)

    if calc_regrid_flag:
        if mpi_config.rank == 0:
            pt.log_debug("Calculating SBCv2 Liquid Water Fraction regridding weights.")
        calculate_supp_pcp_weights(
            supplemental_forcings,
            id_tmp,
            supplemental_forcings.file_in1,
            config_options,
            mpi_config,
            lat_var="Lat",
            lon_var="Lon",
        )
        err_handler.check_program_status(config_options, mpi_config)

    if config_options.grid_type == "gridded":
        # Regrid the input variable
        var_tmp = None
        if mpi_config.rank == 0:
            if mpi_config.rank == 0:
                pt.log_info("Regridding SBCv2 Liquid Water Fraction.")
            try:
                var_tmp = id_tmp.variables[supplemental_forcings.netcdf_var_names[0]][:]
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to extract Liquid Water Fraction from SBCv2 file: {supplemental_forcings.file_in1} ({err})"
                )
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(
            supplemental_forcings, var_tmp, config_options
        )
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_in.data[:, :] = (
                var_sub_tmp / 100.0
            )  # convert from 0-100 to 0-1.0
        except (ValueError, KeyError, AttributeError) as err:
            pt.log_crit(
                f"Unable to place SBCv2 Liquid Water Fraction into local ESMF field: {err}"
            )
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
                supplemental_forcings.regridObj,
                supplemental_forcings.esmf_field_in,
                supplemental_forcings.esmf_field_out,
            )
        except ValueError as ve:
            pt.log_crit(f"Unable to regrid SBCv2 Liquid Water Fraction: {ve}")
        err_handler.check_program_status(config_options, mpi_config)

        # Set any missing data or pixel cells outside the input domain to a default of 100%
        try:
            supplemental_forcings.esmf_field_out.data[
                np.where(supplemental_forcings.regridded_mask == 0)
            ] = 1.0
            supplemental_forcings.esmf_field_out.data[
                np.where(supplemental_forcings.esmf_field_out.data < 0)
            ] = 1.0
        except (ValueError, ArithmeticError) as npe:
            pt.log_crit(
                f"Unable to run mask search on SBCv2 Liquid Water Fraction: {npe}"
            )
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_forcings.regridded_precip2[:] = (
            supplemental_forcings.esmf_field_out.data
        )
        err_handler.check_program_status(config_options, mpi_config)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if config_options.current_output_step == 1:
            supplemental_forcings.regridded_precip1[:] = (
                supplemental_forcings.regridded_precip2[:]
            )
        err_handler.check_program_status(config_options, mpi_config)

    elif config_options.grid_type == "unstructured":
        # Regrid the input variable
        var_tmp = None
        if mpi_config.rank == 0:
            if mpi_config.rank == 0:
                pt.log_debug("Regridding SBCv2 Liquid Water Fraction - unstructured")
            try:
                var_tmp = id_tmp.variables[supplemental_forcings.netcdf_var_names[0]][:]
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to extract Liquid Water Fraction from SBCv2 file: {supplemental_forcings.file_in1} ({err})"
                )
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(
            supplemental_forcings, var_tmp, config_options
        )
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_in.data[:, :] = (
                var_sub_tmp / 100.0
            )  # convert from 0-100 to 0-1.0
        except (ValueError, KeyError, AttributeError) as err:
            pt.log_crit(
                f"Unable to place SBCv2 Liquid Water Fraction into local ESMF field: {err}"
            )
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
                supplemental_forcings.regridObj,
                supplemental_forcings.esmf_field_in,
                supplemental_forcings.esmf_field_out,
            )
        except ValueError as ve:
            pt.log_crit(f"Unable to regrid SBCv2 Liquid Water Fraction: {ve}")
        err_handler.check_program_status(config_options, mpi_config)

        # Set any missing data or pixel cells outside the input domain to a default of 100%
        try:
            supplemental_forcings.esmf_field_out.data[
                np.where(supplemental_forcings.regridded_mask == 0)
            ] = 1.0
            supplemental_forcings.esmf_field_out.data[
                np.where(supplemental_forcings.esmf_field_out.data < 0)
            ] = 1.0
        except (ValueError, ArithmeticError) as npe:
            pt.log_crit(
                f"Unable to run mask search on SBCv2 Liquid Water Fraction: {npe}"
            )
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_forcings.regridded_precip2[:] = (
            supplemental_forcings.esmf_field_out.data
        )
        err_handler.check_program_status(config_options, mpi_config)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if config_options.current_output_step == 1:
            supplemental_forcings.regridded_precip1[:] = (
                supplemental_forcings.regridded_precip2[:]
            )
        err_handler.check_program_status(config_options, mpi_config)

        # Regrid the input variable
        var_tmp_elem = None
        if mpi_config.rank == 0:
            if mpi_config.rank == 0:
                pt.log_debug("Regridding SBCv2 Liquid Water Fraction input variable.")
            try:
                var_tmp_elem = id_tmp.variables[
                    supplemental_forcings.netcdf_var_names[0]
                ][:]
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to extract Liquid Water Fraction from SBCv2 file: {supplemental_forcings.file_in1} ({err})"
                )
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp_elem = mpi_config.scatter_array(
            supplemental_forcings, var_tmp_elem, config_options
        )
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_in_elem.data[:, :] = (
                var_sub_tmp_elem / 100.0
            )  # convert from 0-100 to 0-1.0
        except (ValueError, KeyError, AttributeError) as err:
            pt.log_crit(
                f"Unable to place SBCv2 Liquid Water Fraction into local ESMF field: {err}"
            )
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_out_elem = (
                pt.esmf_regridobj_call_retry_partial(
                    supplemental_forcings.regridObj_elem,
                    supplemental_forcings.esmf_field_in_elem,
                    supplemental_forcings.esmf_field_out_elem,
                )
            )
        except ValueError as ve:
            pt.log_crit(f"Unable to regrid SBCv2 Liquid Water Fraction: {ve}")
        err_handler.check_program_status(config_options, mpi_config)

        # Set any missing data or pixel cells outside the input domain to a default of 100%
        try:
            supplemental_forcings.esmf_field_out_elem.data[
                np.where(supplemental_forcings.regridded_mask_elem == 0)
            ] = 1.0
            supplemental_forcings.esmf_field_out_elem.data[
                np.where(supplemental_forcings.esmf_field_out_elem.data < 0)
            ] = 1.0
        except (ValueError, ArithmeticError) as npe:
            pt.log_crit(
                f"Unable to run mask search on SBCv2 Liquid Water Fraction: {npe}"
            )
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_forcings.regridded_precip2_elem[:] = (
            supplemental_forcings.esmf_field_out_elem.data
        )
        err_handler.check_program_status(config_options, mpi_config)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if config_options.current_output_step == 1:
            supplemental_forcings.regridded_precip1_elem[:] = (
                supplemental_forcings.regridded_precip2_elem[:]
            )
        err_handler.check_program_status(config_options, mpi_config)

    elif config_options.grid_type == "hydrofabric":
        # Regrid the input variable
        var_tmp = None
        if mpi_config.rank == 0:
            if mpi_config.rank == 0:
                pt.log_info("Regridding SBCv2 Liquid Water Fraction - hydrofabric")
            try:
                var_tmp = id_tmp.variables[supplemental_forcings.netcdf_var_names[0]][:]
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to extract Liquid Water Fraction from SBCv2 file: {supplemental_forcings.file_in1} ({err})"
                )
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(
            supplemental_forcings, var_tmp, config_options
        )
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_in.data[:, :] = (
                var_sub_tmp / 100.0
            )  # convert from 0-100 to 0-1.0
        except (ValueError, KeyError, AttributeError) as err:
            pt.log_crit(
                f"Unable to place SBCv2 Liquid Water Fraction into local ESMF field: {err}"
            )
        err_handler.check_program_status(config_options, mpi_config)

        try:
            supplemental_forcings.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
                supplemental_forcings.regridObj,
                supplemental_forcings.esmf_field_in,
                supplemental_forcings.esmf_field_out,
            )
        except ValueError as ve:
            pt.log_crit(f"Unable to regrid SBCv2 Liquid Water Fraction: {ve}")
        err_handler.check_program_status(config_options, mpi_config)

        # Set any missing data or pixel cells outside the input domain to a default of 100%
        try:
            supplemental_forcings.esmf_field_out.data[
                np.where(supplemental_forcings.regridded_mask == 0)
            ] = 1.0
            supplemental_forcings.esmf_field_out.data[
                np.where(supplemental_forcings.esmf_field_out.data < 0)
            ] = 1.0
        except (ValueError, ArithmeticError) as npe:
            pt.log_crit(
                f"Unable to run mask search on SBCv2 Liquid Water Fraction: {npe}"
            )
        err_handler.check_program_status(config_options, mpi_config)

        supplemental_forcings.regridded_precip2[:] = (
            supplemental_forcings.esmf_field_out.data
        )
        err_handler.check_program_status(config_options, mpi_config)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if config_options.current_output_step == 1:
            supplemental_forcings.regridded_precip1[:] = (
                supplemental_forcings.regridded_precip2[:]
            )
        err_handler.check_program_status(config_options, mpi_config)

    pt.close_rank_0_partial(id_tmp)


def regrid_hourly_nbm(
    forcings_or_precip:supplemental_precip|InputForcings, config_options:ConfigOptions, wrf_hydro_geo_meta:GeoMeta, mpi_config:MpiConfig
):
    """Regrid hourly NBM precipitation.

    Function for handling regridding hourly forecasted NBM precipitation.
    :param forcings_or_precip:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # Do we want to use NBM data at this timestep? If not, log and continue
    if not config_options.use_data_at_current_time:
        if mpi_config.rank == 0:
            pt.log_info(
                "Exceeded max hours for NBM data, will not use NBM in final layering."
            )
        return

    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.exists(forcings_or_precip.file_in1):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if forcings_or_precip.regridComplete:
        return

    file_name = f"NBM_PCP_TMP-{mkfilename()}.nc"
    file_uuid = str(mpi_config.uid64)
    nbm_tmp_nc = str(Path(config_options.scratch_dir) / f"{file_uuid}_{file_name}")

    pt.os_remove_rank_0_partial(nbm_tmp_nc)

    if forcings_or_precip.grib_vars is not None:
        fields = []
        for force_count, grib_var in enumerate(forcings_or_precip.grib_vars):
            if mpi_config.rank == 0:
                pt.log_debug(f"Converting NBM Variable: {grib_var}")
            time_str = (
                f"{forcings_or_precip.fcst_hour1}-{forcings_or_precip.fcst_hour2} hour acc fcst"
                if grib_var == "APCP"
                else f"{forcings_or_precip.fcst_hour2} hour fcst"
            )
            fields.append(
                f":{grib_var}:{forcings_or_precip.grib_levels[force_count]}:{time_str}:"
            )
        # fields.append(":(HGT):(surface):")
        # Create a temporary NetCDF file from the GRIB2 file.
        cmd = f'$WGRIB2 -match "({"|".join(fields)})" -not "prob" -not "ens" {forcings_or_precip.file_in1} -netcdf {nbm_tmp_nc}'
    else:
        # Perform a GRIB dump to NetCDF for the precip data.
        time_str=f"{forcings_or_precip.fcst_hour1}-{forcings_or_precip.fcst_hour2} hour acc fcst"
        fieldnbm_match1 = f'":APCP:surface:{time_str}:"'
        fieldnbm_match2 = (
            f'"{forcings_or_precip.fcst_hour1}-{forcings_or_precip.fcst_hour2}"'
        )
        fieldnbm_notmatch1 = '"prob"'  # We don't want the probabilistic QPF layers
        cmd = f"$WGRIB2 {forcings_or_precip.file_in1} -match {fieldnbm_match1} -match {fieldnbm_match2} -not {fieldnbm_notmatch1} -netcdf {nbm_tmp_nc}"

    id_tmp = ioMod.open_grib2(
        forcings_or_precip.file_in1,
        nbm_tmp_nc,
        cmd,
        config_options,
        mpi_config,
        forcings_or_precip.netcdf_var_names[0],
        special_case=True,
    )

    err_handler.check_program_status(config_options, mpi_config)

    pt.log_info("Processing NBM Variables")

    for force_count, nc_var in enumerate(forcings_or_precip.netcdf_var_names):
        if mpi_config.rank == 0:
            pt.log_debug(f"Processing NBM Variable: {nc_var}")

        # Check to see if we need to calculate regridding weights.
        is_supp = forcings_or_precip.grib_vars is None
        if is_supp:
            tag = "supplemental precip"
            calc_regrid_flag = check_supp_pcp_regrid_status(
                id_tmp,
                forcings_or_precip,
                config_options,
                wrf_hydro_geo_meta,
                mpi_config,
            )
        else:
            tag = "input"
            calc_regrid_flag = check_regrid_status(
                id_tmp,
                force_count,
                forcings_or_precip,
                config_options,
                wrf_hydro_geo_meta,
                mpi_config,
            )
        err_handler.check_program_status(config_options, mpi_config)

        if calc_regrid_flag:
            if is_supp:
                if mpi_config.rank == 0:
                    pt.log_debug(f"Calculating NBM {tag} regridding weights.")
                calculate_supp_pcp_weights(
                    forcings_or_precip,
                    id_tmp,
                    forcings_or_precip.file_in1,
                    config_options,
                    mpi_config,
                )
                err_handler.check_program_status(config_options, mpi_config)
            else:
                if mpi_config.rank == 0:
                    pt.log_debug(f"Calculating NBM {tag} regridding weights.")
                calculate_weights(
                    id_tmp,
                    force_count,
                    forcings_or_precip,
                    config_options,
                    mpi_config,
                    fill=True,
                )
                err_handler.check_program_status(config_options, mpi_config)

                # Regrid the height variable.
                if config_options.grid_meta is None:
                    pt.log_warn(
                        "No NBM height file supplied, downscaling will not be available"
                    )
                    err_handler.check_program_status(config_options, mpi_config)
                else:
                    if not os.path.exists(config_options.grid_meta):
                        pt.log_crit(
                            f'NBM height file "{config_options.grid_meta}" does not exist'
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                    terrain_tmp = os.path.join(
                        config_options.scratch_dir, "nbm_terrain_temp.nc"
                    )
                    cmd = f"$WGRIB2 {config_options.grid_meta} -netcdf {terrain_tmp}"
                    hgt_tmp = ioMod.open_grib2(
                        config_options.grid_meta,
                        terrain_tmp,
                        cmd,
                        config_options,
                        mpi_config,
                        "DIST_surface",
                    )
                    if config_options.grid_type == "gridded":
                        if mpi_config.rank == 0:
                            var_tmp = hgt_tmp.variables["DIST_surface"][0, :, :]
                        else:
                            var_tmp = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp = mpi_config.scatter_array(
                            forcings_or_precip, var_tmp, config_options
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            forcings_or_precip.esmf_field_in.data[:, :] = var_sub_tmp
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to place NBM elevation data into the ESMF field object: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            pt.log_debug(
                                "Regridding NBM elevation data to the WRF-Hydro domain."
                            )
                        try:
                            forcings_or_precip.esmf_field_out = (
                                pt.esmf_regridobj_call_retry_partial(
                                    forcings_or_precip.regridObj,
                                    forcings_or_precip.esmf_field_in,
                                    forcings_or_precip.esmf_field_out,
                                )
                            )
                        except ValueError as ve:
                            pt.log_crit(
                                f"Unable to regrid NBM elevation data to the WRF-Hydro domain using ESMF: {ve}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        # Set any pixel cells outside the input domain to the global missing value.
                        try:
                            forcings_or_precip.esmf_field_out.data[
                                np.where(forcings_or_precip.regridded_mask == 0)
                            ] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            pt.log_crit(
                                f"Unable to compute mask on NBM elevation data: {npe}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            forcings_or_precip.height[:, :] = (
                                forcings_or_precip.esmf_field_out.data
                            )
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract ESMF regridded NBM elevation data to a local array: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            hgt_tmp.close()
                    elif config_options.grid_type == "hydrofabric":
                        if mpi_config.rank == 0:
                            var_tmp = hgt_tmp.variables["DIST_surface"][0, :, :]
                        else:
                            var_tmp = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp = mpi_config.scatter_array(
                            forcings_or_precip, var_tmp, config_options
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            forcings_or_precip.esmf_field_in.data[:, :] = var_sub_tmp
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to place NBM elevation data into the ESMF field object: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            pt.log_debug(
                                "Regridding NBM elevation data to the WRF-Hydro domain."
                            )
                        try:
                            forcings_or_precip.esmf_field_out = (
                                pt.esmf_regridobj_call_retry_partial(
                                    forcings_or_precip.regridObj,
                                    forcings_or_precip.esmf_field_in,
                                    forcings_or_precip.esmf_field_out,
                                )
                            )
                        except ValueError as ve:
                            pt.log_crit(
                                f"Unable to regrid NBM elevation data to the WRF-Hydro domain using ESMF: {ve}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        # Set any pixel cells outside the input domain to the global missing value.
                        try:
                            forcings_or_precip.esmf_field_out.data[
                                np.where(forcings_or_precip.regridded_mask == 0)
                            ] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            pt.log_crit(
                                f"Unable to compute mask on NBM elevation data: {npe}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            forcings_or_precip.height[:] = (
                                forcings_or_precip.esmf_field_out.data
                            )
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract ESMF regridded NBM elevation data to a local array: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            hgt_tmp.close()
                    elif config_options.grid_type == "unstructured":
                        if mpi_config.rank == 0:
                            var_tmp = hgt_tmp.variables["DIST_surface"][0, :, :]
                        else:
                            var_tmp = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp = mpi_config.scatter_array(
                            forcings_or_precip, var_tmp, config_options
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            forcings_or_precip.esmf_field_in.data[:, :] = var_sub_tmp
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to place NBM elevation data into the ESMF field object: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            pt.log_debug(
                                "Regridding NBM elevation data to the WRF-Hydro domain."
                            )
                        try:
                            forcings_or_precip.esmf_field_out = (
                                pt.esmf_regridobj_call_retry_partial(
                                    forcings_or_precip.regridObj,
                                    forcings_or_precip.esmf_field_in,
                                    forcings_or_precip.esmf_field_out,
                                )
                            )
                        except ValueError as ve:
                            pt.log_crit(
                                f"Unable to regrid NBM elevation data to the WRF-Hydro domain using ESMF: {ve}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        # Set any pixel cells outside the input domain to the global missing value.
                        try:
                            forcings_or_precip.esmf_field_out.data[
                                np.where(forcings_or_precip.regridded_mask == 0)
                            ] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            pt.log_crit(
                                f"Unable to compute mask on NBM elevation data: {npe}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            forcings_or_precip.height[:] = (
                                forcings_or_precip.esmf_field_out.data
                            )
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract ESMF regridded NBM elevation data to a local array: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            var_tmp_elem = hgt_tmp.variables["DIST_surface"][0, :, :]
                        else:
                            var_tmp_elem = None
                        err_handler.check_program_status(config_options, mpi_config)

                        var_sub_tmp_elem = mpi_config.scatter_array(
                            forcings_or_precip, var_tmp_elem, config_options
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            forcings_or_precip.esmf_field_in_elem.data[:, :] = (
                                var_sub_tmp_elem
                            )
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to place NBM elevation data into the ESMF field object: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            pt.log_debug(
                                "Regridding NBM elevation data to the unstructured domain."
                            )
                        try:
                            forcings_or_precip.esmf_field_out_elem = (
                                pt.esmf_regridobj_call_retry_partial(
                                    forcings_or_precip.regridObj_elem,
                                    forcings_or_precip.esmf_field_in_elem,
                                    forcings_or_precip.esmf_field_out_elem,
                                )
                            )
                        except ValueError as ve:
                            pt.log_crit(
                                f"Unable to regrid NBM elevation data to the unstructured domain using ESMF: {ve}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        # Set any pixel cells outside the input domain to the global missing value.
                        try:
                            forcings_or_precip.esmf_field_out_elem.data[
                                np.where(forcings_or_precip.regridded_mask_elem == 0)
                            ] = config_options.globalNdv
                        except (ValueError, ArithmeticError) as npe:
                            pt.log_crit(
                                f"Unable to compute element mask on NBM elevation data: {npe}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        try:
                            forcings_or_precip.height_elem[:] = (
                                forcings_or_precip.esmf_field_out_elem.data
                            )
                        except (ValueError, KeyError, AttributeError) as err:
                            pt.log_crit(
                                f"Unable to extract ESMF regridded NBM elevation data to a local array: {err}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                        if mpi_config.rank == 0:
                            hgt_tmp.close()

        if config_options.grid_type == "gridded":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding NBM {nc_var}")
                try:
                    var_tmp = id_tmp.variables[nc_var][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract data from NBM file: {forcings_or_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                forcings_or_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                forcings_or_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place NBM {tag} into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                forcings_or_precip.esmf_field_out = (
                    pt.esmf_regridobj_call_retry_partial(
                        forcings_or_precip.regridObj,
                        forcings_or_precip.esmf_field_in,
                        forcings_or_precip.esmf_field_out,
                    )
                )
            except ValueError as ve:
                pt.log_crit(f"Unable to regrid NBM {tag}: {ve}")
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value.
            try:
                forcings_or_precip.esmf_field_out.data[
                    np.where(forcings_or_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on NBM supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            destination1 = (
                forcings_or_precip.regridded_precip1
                if is_supp
                else forcings_or_precip.regridded_forcings1[
                    forcings_or_precip.input_map_output[force_count]
                ]
            )

            destination2 = (
                forcings_or_precip.regridded_precip2
                if is_supp
                else forcings_or_precip.regridded_forcings2[
                    forcings_or_precip.input_map_output[force_count]
                ]
            )

            destination2[:, :] = forcings_or_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total from kg.m-2.hour-1 to a rate of mm.s-1
            if "APCP" in nc_var:
                try:
                    ind_valid = np.where(destination2 != config_options.globalNdv)
                    if forcings_or_precip.input_frequency == 60.0:
                        destination2[ind_valid] = destination2[ind_valid] / 3600.0
                    elif forcings_or_precip.input_frequency == 360.0:
                        destination2[ind_valid] = (
                            destination2[ind_valid] / 21600.0
                        )  # uniform disaggregation for 6-hourly nbm data
                    del ind_valid
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    pt.log_crit(f"Unable to run NDV search on NBM precipitation: {npe}")
                err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                destination1[:, :] = destination2[:, :]
            err_handler.check_program_status(config_options, mpi_config)
        elif config_options.grid_type == "hydrofabric":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding NBM {nc_var}")
                try:
                    var_tmp = id_tmp.variables[nc_var][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract data from NBM file: {forcings_or_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                forcings_or_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                forcings_or_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place NBM {tag} into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                forcings_or_precip.esmf_field_out = (
                    pt.esmf_regridobj_call_retry_partial(
                        forcings_or_precip.regridObj,
                        forcings_or_precip.esmf_field_in,
                        forcings_or_precip.esmf_field_out,
                    )
                )
            except ValueError as ve:
                pt.log_crit(f"Unable to regrid NBM {tag}: {ve}")
            err_handler.check_program_status(config_options, mpi_config)
            # Set any pixel cells outside the input domain to the global missing value.
            try:
                forcings_or_precip.esmf_field_out.data[
                    np.where(forcings_or_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on NBM supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            destination1 = (
                forcings_or_precip.regridded_precip1
                if is_supp
                else forcings_or_precip.regridded_forcings1[
                    forcings_or_precip.input_map_output[force_count]
                ]
            )

            destination2 = (
                forcings_or_precip.regridded_precip2
                if is_supp
                else forcings_or_precip.regridded_forcings2[
                    forcings_or_precip.input_map_output[force_count]
                ]
            )

            destination2[:] = forcings_or_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total from kg.m-2.hour-1 to a rate of mm.s-1
            if "APCP" in nc_var:
                try:
                    ind_valid = np.where(destination2 != config_options.globalNdv)
                    if forcings_or_precip.input_frequency == 60.0:
                        destination2[ind_valid] = destination2[ind_valid] / 3600.0
                    elif forcings_or_precip.input_frequency == 360.0:
                        destination2[ind_valid] = (
                            destination2[ind_valid] / 21600.0
                        )  # uniform disaggregation for 6-hourly nbm data
                    del ind_valid
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    pt.log_crit(f"Unable to run NDV search on NBM precipitation: {npe}")
                err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                destination1[:] = destination2[:]
            err_handler.check_program_status(config_options, mpi_config)
        elif config_options.grid_type == "unstructured":
            # Regrid the input variables.
            var_tmp = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding NBM {nc_var}")
                try:
                    var_tmp = id_tmp.variables[nc_var][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract data from NBM file: {forcings_or_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                forcings_or_precip, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                forcings_or_precip.esmf_field_in.data[:, :] = var_sub_tmp
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place NBM {tag} into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                forcings_or_precip.esmf_field_out = (
                    pt.esmf_regridobj_call_retry_partial(
                        forcings_or_precip.regridObj,
                        forcings_or_precip.esmf_field_in,
                        forcings_or_precip.esmf_field_out,
                    )
                )
            except ValueError as ve:
                pt.log_crit(f"Unable to regrid NBM {tag}: {ve}")
            err_handler.check_program_status(config_options, mpi_config)
            # Set any pixel cells outside the input domain to the global missing value.
            try:
                forcings_or_precip.esmf_field_out.data[
                    np.where(forcings_or_precip.regridded_mask == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on NBM supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            destination1 = (
                forcings_or_precip.regridded_precip1
                if is_supp
                else forcings_or_precip.regridded_forcings1[
                    forcings_or_precip.input_map_output[force_count]
                ]
            )

            destination2 = (
                forcings_or_precip.regridded_precip2
                if is_supp
                else forcings_or_precip.regridded_forcings2[
                    forcings_or_precip.input_map_output[force_count]
                ]
            )

            destination2[:] = forcings_or_precip.esmf_field_out.data
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total from kg.m-2.hour-1 to a rate of mm.s-1
            if "APCP" in nc_var:
                try:
                    ind_valid = np.where(destination2 != config_options.globalNdv)
                    if forcings_or_precip.input_frequency == 60.0:
                        destination2[ind_valid] = destination2[ind_valid] / 3600.0
                    elif forcings_or_precip.input_frequency == 360.0:
                        destination2[ind_valid] = (
                            destination2[ind_valid] / 21600.0
                        )  # uniform disaggregation for 6-hourly nbm data
                    del ind_valid
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    pt.log_crit(f"Unable to run NDV search on NBM precipitation: {npe}")
                err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                destination1[:] = destination2[:]
            err_handler.check_program_status(config_options, mpi_config)

            # Regrid the input variables.
            var_tmp_elem = None
            if mpi_config.rank == 0:
                pt.log_debug(f"Regridding NBM {nc_var}")
                try:
                    var_tmp_elem = id_tmp.variables[nc_var][0, :, :]
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract data from NBM file: {forcings_or_precip.file_in1} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp_elem = mpi_config.scatter_array(
                forcings_or_precip, var_tmp_elem, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            try:
                forcings_or_precip.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place NBM {tag} into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                forcings_or_precip.esmf_field_out_elem = (
                    pt.esmf_regridobj_call_retry_partial(
                        forcings_or_precip.regridObj_elem,
                        forcings_or_precip.esmf_field_in_elem,
                        forcings_or_precip.esmf_field_out_elem,
                    )
                )
            except ValueError as ve:
                pt.log_crit(f"Unable to regrid NBM {tag}: {ve}")
            err_handler.check_program_status(config_options, mpi_config)
            # Set any pixel cells outside the input domain to the global missing value.
            try:
                forcings_or_precip.esmf_field_out_elem.data[
                    np.where(forcings_or_precip.regridded_mask_elem == 0)
                ] = config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to run mask search on NBM supplemental precipitation: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            destination1_elem = (
                forcings_or_precip.regridded_precip1_elem
                if is_supp
                else forcings_or_precip.regridded_forcings1_elem[
                    forcings_or_precip.input_map_output[force_count]
                ]
            )

            destination2_elem = (
                forcings_or_precip.regridded_precip2_elem
                if is_supp
                else forcings_or_precip.regridded_forcings2_elem[
                    forcings_or_precip.input_map_output[force_count]
                ]
            )

            destination2_elem[:] = forcings_or_precip.esmf_field_out_elem.data
            err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total from kg.m-2.hour-1 to a rate of mm.s-1
            if "APCP" in nc_var:
                try:
                    ind_valid_elem = np.where(
                        destination2_elem != config_options.globalNdv
                    )
                    if forcings_or_precip.input_frequency == 60.0:
                        destination2_elem[ind_valid_elem] = (
                            destination2_elem[ind_valid_elem] / 3600.0
                        )
                    elif forcings_or_precip.input_frequency == 360.0:
                        destination2_elem[ind_valid_elem] = (
                            destination2_elem[ind_valid_elem] / 21600.0
                        )  # uniform disaggregation for 6-hourly nbm data
                    del ind_valid_elem
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    pt.log_crit(f"Unable to run NDV search on NBM precipitation: {npe}")
                err_handler.check_program_status(config_options, mpi_config)

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if config_options.current_output_step == 1:
                destination1_elem[:] = destination2_elem[:]
            err_handler.check_program_status(config_options, mpi_config)

    pt.close_rank_0_partial(id_tmp)
    pt.os_remove_rank_0_partial(nbm_tmp_nc)


@static_vars(last_file=None)
def regrid_ndfd(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid NDFD forcing data to the WRF-Hydro domain."""
    pt = Partials(mpi_config, config_options)

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            pt.log_debug("No NDFD regridding required for this timestep.")
        return

    pt.log_info("Regrid NDFD")

    hour = input_forcings.fcst_hour2
    current_cycle = config_options.current_fcst_cycle
    forecast_time = config_options.current_time
    # DEBUG if mpi_config.rank == 0: LOG.debug(f"NEXT FILE: {hour=}, {current_cycle=}, {forecast_time=}")

    ndfd_files = ("tmp", "wdir", "wspd", "qpf")
    fill_values = {"tmp": 288.0, "wdir": 45.0, "wspd": 0.71, "qpf": 0}

    # check / set previous file to see if we're going to reuse
    reuse_prev_file = (
        f"{input_forcings.file_in2}-{current_cycle}" == regrid_ndfd.last_file
    )
    regrid_ndfd.last_file = f"{input_forcings.file_in2}-{current_cycle}"

    for i, ndfd_var in enumerate(ndfd_files):
        # check if file exists for this time period
        grb_file = input_forcings.file_in2.replace("%FIELD%", ndfd_var)

        tmp_file = os.path.join(
            config_options.scratch_dir, f"temp_ndfd_conus_{ndfd_var}.nc"
        )
        # Temp file may exist. If it does, and we don't need it again, remove it.....
        if not reuse_prev_file:
            pt.os_remove_rank_0_partial(tmp_file)
        err_handler.check_program_status(config_options, mpi_config)

        id_tmp = None
        try:
            if reuse_prev_file:
                if mpi_config.rank == 0:
                    pt.log_debug(f"Cycle unchanged, reusing temporary file: {tmp_file}")
                id_tmp = ioMod.open_netcdf_forcing(tmp_file, config_options, mpi_config)
            else:
                if mpi_config.rank == 0:
                    pt.log_debug(
                        f"New forecast cycle, creating temporary file from: {grb_file}, cycle hour {current_cycle.hour}"
                    )
                if not WGRIB2_env:
                    cmd = [f":d={current_cycle.strftime('%Y%m%d%H')}:"]
                else:
                    cmd = f"$WGRIB2 -match :d={current_cycle.strftime('%Y%m%d%H')}: -vt {grb_file} | sort -t = -k 2 | $WGRIB2 -i -netcdf {tmp_file} {grb_file}"
                id_tmp = ioMod.open_grib2(
                    grb_file,
                    tmp_file,
                    cmd,
                    config_options,
                    mpi_config,
                    inputVar=None,
                    special_case=True,
                )

            # look to see if current time is in file:
            skip_file = np.ubyte(0)
            if mpi_config.rank == 0:
                times = [datetime.utcfromtimestamp(t) for t in id_tmp["time"][:]]
                if ndfd_var != "qpf":
                    if forecast_time > times[-1] + timedelta(hours=3):
                        pt.log_debug("Forecast time beyond NDFD range, skipping")
                        skip_file = 1
                    elif forecast_time in times:
                        time_index = times.index(forecast_time)
                        pt.log_debug(
                            f"Found exact time {forecast_time} in NDFD file at index {time_index} for variable {ndfd_var}"
                        )
                    else:
                        time_index = min(
                            range(len(times)),
                            key=lambda i: abs(times[i] - forecast_time),
                        )
                        pt.log_debug(
                            f"Exact time {forecast_time} not found in NDFD file, using closest time at {times[time_index]}"
                        )
                else:
                    # TODO: qpf special handling
                    if forecast_time > times[-1] - timedelta(hours=6):
                        pt.log_debug("Forecast time beyond NDFD precip range, skipping")
                        skip_file = 1
                    else:
                        time_index = int(hour // 6)
                        pt.log_debug(
                            f"Forecast hour {forecast_time} will use precip from {times[time_index] - timedelta(hours=6)} to {times[time_index]}"
                        )

            skip_file = mpi_config.broadcast_parameter(
                skip_file, config_options, param_type=np.ubyte
            )
            err_handler.check_program_status(config_options, mpi_config)

            if skip_file:
                if config_options.grid_type == "gridded":
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[i], :, :
                    ] = config_options.globalNdv
                elif config_options.grid_type == "hydrofabric":
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[i], :
                    ] = config_options.globalNdv
                elif config_options.grid_type == "unstructured":
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[i], :
                    ] = config_options.globalNdv
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[i], :
                    ] = config_options.globalNdv
                continue

            if mpi_config.rank == 0:
                pt.log_debug(f"Processing NDFD variable: {ndfd_var}")

            calc_regrid_flag = check_regrid_status(
                id_tmp,
                i,
                input_forcings,
                config_options,
                wrf_hydro_geo_meta,
                mpi_config,
            )
            err_handler.check_program_status(config_options, mpi_config)

            if calc_regrid_flag:
                if mpi_config.rank == 0:
                    pt.log_debug("Calculating NDFD regridding weights.")
                calculate_weights(
                    id_tmp,
                    i,
                    input_forcings,
                    config_options,
                    mpi_config,
                    wrf_hydro_geo_meta,
                )
                err_handler.check_program_status(config_options, mpi_config)

            var_tmp = None
            if mpi_config.rank == 0:
                try:
                    var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[i]][
                        time_index, :, :
                    ]
                    var_tmp = var_tmp.filled(
                        config_options.globalNdv
                    )  # fill_values[ndfd_var])
                    if config_options.grid_type == "unstructured":
                        var_tmp_elem = id_tmp.variables[
                            input_forcings.netcdf_var_names[i]
                        ][time_index, :, :]
                        var_tmp_elem = var_tmp_elem.filled(config_options.globalNdv)
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to extract NDFD variable: {input_forcings.netcdf_var_names[i]} from: {input_forcings.tmpFile} ({err})"
                    )
            err_handler.check_program_status(config_options, mpi_config)

            var_sub_tmp = mpi_config.scatter_array(
                input_forcings, var_tmp, config_options
            )
            err_handler.check_program_status(config_options, mpi_config)

            if config_options.grid_type == "unstructured":
                var_sub_tmp_elem = mpi_config.scatter_array(
                    input_forcings, var_tmp_elem, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)
            try:
                if config_options.grid_type == "gridded":
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                elif config_options.grid_type == "hydrofabric":
                    input_forcings.esmf_field_in.data[:] = var_sub_tmp
                elif config_options.grid_type == "unstructured":
                    input_forcings.esmf_field_in.data[:] = var_sub_tmp
                    input_forcings.esmf_field_in_elem.data[:] = var_sub_tmp_elem
                # DEBUG if mpi_config.rank == 1: LOG.debug(f"esmf_file_in has type: {type(input_forcings.esmf_field_in.data)}, var_sub_tmp has type: {type(var_sub_tmp)}")
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(f"Unable to place local array into local ESMF field: {err}")
            err_handler.check_program_status(config_options, mpi_config)

            try:
                input_forcings.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
                    input_forcings.regridObj,
                    input_forcings.esmf_field_in,
                    input_forcings.esmf_field_out,
                )
                if config_options.grid_type == "unstructured":
                    input_forcings.esmf_field_out_elem = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj_elem,
                            input_forcings.esmf_field_in_elem,
                            input_forcings.esmf_field_out_elem,
                        )
                    )
            except ValueError as ve:
                pt.log_crit(
                    f"Unable to regrid input NDFD forcing variable using ESMF: {ve}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # Set any pixel cells outside the input domain to the global missing value, and fix missings that were interpolated
            try:
                input_forcings.esmf_field_out.data[
                    np.where(input_forcings.regridded_mask == 0)
                ] = config_options.globalNdv
                if config_options.grid_type == "unstructured":
                    input_forcings.esmf_field_out_elem.data[
                        np.where(input_forcings.regridded_mask_elem == 0)
                    ] = config_options.globalNdv
                # input_forcings.esmf_field_out.data[np.where((input_forcings.esmf_field_out.data/config_options.globalNdv) > 0.75)] = \
                #     config_options.globalNdv
            except (ValueError, ArithmeticError) as npe:
                pt.log_crit(
                    f"Unable to calculate mask from input NDFD regridded forcings: {npe}"
                )
            err_handler.check_program_status(config_options, mpi_config)

            # If processing wind speed, calculate final U2 / V2 fields
            def u_v_comps(wspd, wdir):
                rads = wdir * (np.pi / 180.0)
                u = -1 * wspd * np.sin(rads)
                v = -1 * wspd * np.cos(rads)
                return u, v

            if ndfd_var == "wspd":  # we already have direction at this point
                try:
                    ind_valid = np.where(
                        input_forcings.esmf_field_out.data != config_options.globalNdv
                    )
                    wspd = input_forcings.esmf_field_out.data[ind_valid]
                    wdir = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[i - 1]
                    ][ind_valid]

                    u, v = u_v_comps(wspd, wdir)
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[i - 1]
                    ][ind_valid] = u
                    input_forcings.esmf_field_out.data[ind_valid] = v

                    if config_options.grid_type == "unstructured":
                        ind_valid = np.where(
                            input_forcings.esmf_field_out_elem.data
                            != config_options.globalNdv
                        )
                        wspd = input_forcings.esmf_field_out_elem.data[ind_valid]
                        wdir = input_forcings.regridded_forcings2_elem[
                            input_forcings.input_map_output[i - 1]
                        ][ind_valid]

                        u, v = u_v_comps(wspd, wdir)
                        input_forcings.regridded_forcings2_elem[
                            input_forcings.input_map_output[i - 1]
                        ][ind_valid] = u
                        input_forcings.esmf_field_out_elem.data[ind_valid] = v

                    del ind_valid, u, v

                except (
                    IndexError,
                    ValueError,
                    ArithmeticError,
                    AttributeError,
                    KeyError,
                ) as npe:
                    pt.log_crit(
                        f"Unable to convert NDFD wind speed/dir to U/V components: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

            # Convert the hourly precipitation total to a rate of mm/s
            if ndfd_var == "qpf":
                try:
                    ind_valid = np.where(
                        input_forcings.esmf_field_out.data != config_options.globalNdv
                    )
                    input_forcings.esmf_field_out.data[ind_valid] = (
                        input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                    )
                    if config_options.grid_type == "unstructured":
                        ind_valid = np.where(
                            input_forcings.esmf_field_out_elem.data
                            != config_options.globalNdv
                        )
                        input_forcings.esmf_field_out_elem.data[ind_valid] = (
                            input_forcings.esmf_field_out_elem.data[ind_valid] / 3600.0
                        )
                    del ind_valid
                except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                    pt.log_crit(
                        f"Unable to run NDV search on NDFD QPF precipitation: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

            try:
                if config_options.grid_type == "gridded":
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[i], :, :
                    ] = input_forcings.esmf_field_out.data
                elif config_options.grid_type == "hydrofabric":
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[i], :
                    ] = input_forcings.esmf_field_out.data
                elif config_options.grid_type == "unstructured":
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[i], :
                    ] = input_forcings.esmf_field_out.data
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[i], :
                    ] = input_forcings.esmf_field_out_elem.data
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to place local ESMF regridded data into local array: {err}"
                )
            err_handler.check_program_status(config_options, mpi_config)
        finally:
            pt.close_anyrank_partial(id_tmp)  # NOTE all ranks are calling this

            # only remove the scratch file if this was a new‐cycle run
            if not reuse_prev_file:
                pt.os_remove_rank_0_partial(tmp_file)
            err_handler.check_program_status(config_options, mpi_config)


def regrid_aorc_aws(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """Regrid AORC AWS forcing data to the WRF-Hydro domain."""
    pt = Partials(mpi_config, config_options)

    fill_values = {
        "TMP": 288.0,
        "SPFH": 0.005,
        "PRES": 101300.0,
        "APCP": 0,
        "UGRD": 1.0,
        "VGRD": 1.0,
        "DSWRF": 80.0,
        "DLWRF": 310.0,
    }
    with timing_block(f"regrid step 1 | {mpi_config.rank}: get dataset"):
        mpi_config.comm.barrier()
        id_tmp = config_options.aws_obj

        pt.log_debug("Processing Custom NetCDF Forcing Variables")

    with timing_block(f"regrid step 2 | {mpi_config.rank}: loop over variables"):
        for force_count, nc_var in enumerate(input_forcings.netcdf_var_names):
            if mpi_config.rank == 0:
                pt.log_debug(f"Processing Custom NetCDF Forcing Variable: {nc_var}")
            with timing_block(
                f"regrid step 2.1 | {mpi_config.rank}: check regrid status"
            ):
                calc_regrid_flag = check_regrid_status(
                    id_tmp,
                    force_count,
                    input_forcings,
                    config_options,
                    wrf_hydro_geo_meta,
                    mpi_config,
                )

            with timing_block(
                f"regrid step 2.2 | {mpi_config.rank}: calculate weights"
            ):
                if calc_regrid_flag:
                    calculate_weights(
                        id_tmp,
                        force_count,
                        input_forcings,
                        config_options,
                        mpi_config,
                        wrf_hydro_geo_meta,
                        lat_var="y",
                        lon_var="x",
                    )

            with timing_block(
                f"regrid step 2.3 | {mpi_config.rank}: ERA5-interim mask"
            ):
                # Flag to set regridded mask for AORC to overlay with ERA5-Interim blend
                if 23 in config_options.input_forcings:
                    input_forcings.regridded_mask_AORC = input_forcings.regridded_mask
                    if config_options.grid_type == "unstructured":
                        input_forcings.regridded_mask_elem_AORC = (
                            input_forcings.regridded_mask_elem
                        )

            if config_options.grid_type == "gridded":
                # Regrid the input variables.
                var_tmp = None
                fill = fill_values.get(
                    input_forcings.grib_vars[force_count], config_options.globalNdv
                )
                if mpi_config.rank == 0:
                    pt.log_debug(f"Regridding Custom netCDF input variable: {nc_var}")
                    try:
                        pt.log_debug(f"Using {fill} to replace missing values in input")
                        var_tmp = id_tmp[nc_var].to_masked_array().filled(fill)
                    except Exception as err:
                        pt.log_crit(
                            f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local array into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid input Custom netCDF forcing variables using ESMF: {ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = fill
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to calculate mask from input Custom netCDF regridded forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Convert the hourly precipitation total to a rate of mm/s
                if nc_var == "APCP_surface":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != fill)
                        # Flag to set regridded mask for AORC to overlay with ERA5-Interim blend
                        input_forcings.esmf_field_out.data[ind_valid] = (
                            input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        )
                        del ind_valid
                    except (
                        ValueError,
                        ArithmeticError,
                        AttributeError,
                        KeyError,
                    ) as npe:
                        pt.log_crit(
                            f"Unable to run NDV search on Custom netCDF precipitation: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local ESMF regridded data into local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :, :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :, :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "unstructured":
                # Regrid the input variables.
                var_tmp = None
                fill = fill_values.get(
                    input_forcings.grib_vars[force_count], config_options.globalNdv
                )
                if mpi_config.rank == 0:
                    pt.log_debug(f"Regridding Custom netCDF input variable: {nc_var}")
                    try:
                        pt.log_debug(f"Using {fill} to replace missing values in input")
                        var_tmp = id_tmp[nc_var].to_masked_array().filled(fill)
                    except Exception as err:
                        pt.log_crit(
                            f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp = mpi_config.scatter_array(
                    input_forcings, var_tmp, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local array into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj,
                            input_forcings.esmf_field_in,
                            input_forcings.esmf_field_out,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid input Custom netCDF forcing variables using ESMF: {ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[
                        np.where(input_forcings.regridded_mask == 0)
                    ] = fill
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to calculate mask from input Custom netCDF regridded forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Convert the hourly precipitation total to a rate of mm/s
                if nc_var == "APCP_surface":
                    try:
                        ind_valid = np.where(input_forcings.esmf_field_out.data != fill)
                        input_forcings.esmf_field_out.data[ind_valid] = (
                            input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                        )
                        del ind_valid
                    except (
                        ValueError,
                        ArithmeticError,
                        AttributeError,
                        KeyError,
                    ) as npe:
                        pt.log_crit(
                            f"Unable to run NDV search on Custom netCDF precipitation: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local ESMF regridded data into local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

                # Regrid the input variables.
                var_tmp_elem = None
                fill = fill_values.get(
                    input_forcings.grib_vars[force_count], config_options.globalNdv
                )
                if mpi_config.rank == 0:
                    pt.log_debug(f"Regridding Custom netCDF input variable: {nc_var}")
                    try:
                        pt.log_debug(f"Using {fill} to replace missing values in input")
                        var_tmp_elem = id_tmp[nc_var].to_masked_array().filled(fill)
                    except Exception as err:
                        pt.log_crit(
                            f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({err})"
                        )
                err_handler.check_program_status(config_options, mpi_config)

                var_sub_tmp_elem = mpi_config.scatter_array(
                    input_forcings, var_tmp_elem, config_options
                )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp_elem
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local array into local ESMF field: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_out_elem = (
                        pt.esmf_regridobj_call_retry_partial(
                            input_forcings.regridObj_elem,
                            input_forcings.esmf_field_in_elem,
                            input_forcings.esmf_field_out_elem,
                        )
                    )
                except ValueError as ve:
                    pt.log_crit(
                        f"Unable to regrid input Custom netCDF forcing variables using ESMF: {ve}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out_elem.data[
                        np.where(input_forcings.regridded_mask_elem == 0)
                    ] = fill
                except (ValueError, ArithmeticError) as npe:
                    pt.log_crit(
                        f"Unable to calculate mask from input Custom netCDF regridded forcings: {npe}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # Convert the hourly precipitation total to a rate of mm/s
                if nc_var == "APCP_surface":
                    try:
                        ind_valid_elem = np.where(
                            input_forcings.esmf_field_out_elem.data != fill
                        )
                        input_forcings.esmf_field_out_elem.data[ind_valid_elem] = (
                            input_forcings.esmf_field_out_elem.data[ind_valid_elem]
                            / 3600.0
                        )
                        del ind_valid_elem
                    except (
                        ValueError,
                        ArithmeticError,
                        AttributeError,
                        KeyError,
                    ) as npe:
                        pt.log_crit(
                            f"Unable to run NDV search on Custom netCDF precipitation: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.esmf_field_out_elem.data
                except (ValueError, KeyError, AttributeError) as err:
                    pt.log_crit(
                        f"Unable to place local ESMF regridded data into local array: {err}"
                    )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are on the first timestep, set the previous regridded field to be
                # the latest as there are no states for time 0.
                if config_options.current_output_step == 1:
                    input_forcings.regridded_forcings1_elem[
                        input_forcings.input_map_output[force_count], :
                    ] = input_forcings.regridded_forcings2_elem[
                        input_forcings.input_map_output[force_count], :
                    ]
                err_handler.check_program_status(config_options, mpi_config)

            elif config_options.grid_type == "hydrofabric":
                with timing_block(
                    f"regrid step 2.4 | {mpi_config.rank}: get fill value"
                ):
                    # Regrid the input variables.
                    var_tmp = None
                    fill = fill_values.get(
                        input_forcings.grib_vars[force_count], config_options.globalNdv
                    )

                if mpi_config.rank == 0:
                    with timing_block(
                        f"regrid step 2.5.1 | {mpi_config.rank}: regrid messages"
                    ):
                        pt.log_debug(
                            f"Regridding Custom netCDF input variable: {nc_var}"
                        )
                    try:
                        with timing_block(
                            f"regrid step 2.5.2 | {mpi_config.rank}: messages2"
                        ):
                            pt.log_debug(
                                f"Using {fill} to replace missing values in input"
                            )
                        with timing_block(
                            f"regrid step 2.5.3 | {mpi_config.rank}: fill missing"
                        ):
                            var_tmp = id_tmp[nc_var].to_masked_array().filled(fill)

                    except Exception as err:
                        with timing_block(
                            f"regrid step 2.5.4 | {mpi_config.rank}: extract error"
                        ):
                            pt.log_crit(
                                f"Unable to extract {nc_var} from: {input_forcings.file_in2} ({str(err)})"
                            )

                with timing_block(
                    f"regrid step 2.5.5 | {mpi_config.rank}: check program status"
                ):
                    err_handler.check_program_status(config_options, mpi_config)

                with timing_block(
                    f"regrid step 2.6 | {mpi_config.rank}: scatter input variable"
                ):
                    var_sub_tmp = mpi_config.scatter_array(
                        input_forcings, var_tmp, config_options
                    )
                    err_handler.check_program_status(config_options, mpi_config)
                with timing_block(
                    f"regrid step 2.7 | {mpi_config.rank}: place local array into local ESMF field"
                ):
                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place local array into local ESMF field: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                with timing_block(
                    f"regrid step 2.8 | {mpi_config.rank}: regrid input variable ESMF"
                ):
                    try:
                        input_forcings.esmf_field_out = (
                            pt.esmf_regridobj_call_retry_partial(
                                input_forcings.regridObj,
                                input_forcings.esmf_field_in,
                                input_forcings.esmf_field_out,
                            )
                        )
                    except ValueError as ve:
                        pt.log_crit(
                            f"Unable to regrid input Custom netCDF forcing variables using ESMF: {ve}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                with timing_block(
                    f"regrid step 2.9 | {mpi_config.rank}: set pixels outside input domain to missing value"
                ):
                    # Set any pixel cells outside the input domain to the global missing value.
                    try:
                        input_forcings.esmf_field_out.data[
                            np.where(input_forcings.regridded_mask == 0)
                        ] = fill
                    except (ValueError, ArithmeticError) as npe:
                        pt.log_crit(
                            f"Unable to calculate mask from input Custom netCDF regridded forcings: {npe}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                with timing_block(
                    f"regrid step 2.10 | {mpi_config.rank}: convert hourly precipitation to mm/s"
                ):
                    # Convert the hourly precipitation total to a rate of mm/s
                    if nc_var == "APCP_surface":
                        try:
                            ind_valid = np.where(
                                input_forcings.esmf_field_out.data != fill
                            )
                            input_forcings.esmf_field_out.data[ind_valid] = (
                                input_forcings.esmf_field_out.data[ind_valid] / 3600.0
                            )
                            ind_valid = np.where(
                                input_forcings.esmf_field_out.data < 0.0
                            )
                            input_forcings.esmf_field_out.data[ind_valid] = 0.0
                            del ind_valid
                        except (
                            ValueError,
                            ArithmeticError,
                            AttributeError,
                            KeyError,
                        ) as npe:
                            pt.log_crit(
                                f"Unable to run NDV search on Custom netCDF precipitation: {npe}"
                            )
                        err_handler.check_program_status(config_options, mpi_config)

                with timing_block(
                    f"regrid step 2.11 | {mpi_config.rank}: place regridded data into local array"
                ):
                    try:
                        input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        pt.log_crit(
                            f"Unable to place local ESMF regridded data into local array: {err}"
                        )
                    err_handler.check_program_status(config_options, mpi_config)

                with timing_block(
                    f"regrid step 2.12 | {mpi_config.rank}: update previous regridded field for first timestep"
                ):
                    # If we are on the first timestep, set the previous regridded field to be
                    # the latest as there are no states for time 0.
                    if config_options.current_output_step == 1:
                        input_forcings.regridded_forcings1[
                            input_forcings.input_map_output[force_count], :
                        ] = input_forcings.regridded_forcings2[
                            input_forcings.input_map_output[force_count], :
                        ]
                    err_handler.check_program_status(config_options, mpi_config)


def check_regrid_status(
    id_tmp, force_count, input_forcings, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Check regrid status function.

    Function for checking to see if regridding weights need to be
    calculated (or recalculated).
    :param wrf_hydro_geo_meta:
    :param force_count:
    :param id_tmp:
    :param input_forcings:
    :param config_options:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # If the destination ESMF field hasn't been created, create it here.
    if not input_forcings.esmf_field_out:
        if config_options.grid_type == "gridded":
            try:
                input_forcings.esmf_field_out = pt.esmf_field_retry_partial(
                    wrf_hydro_geo_meta.esmf_grid,
                    name=f"{input_forcings.product_name}FORCING_REGRIDDED",
                )
            except ESMF.ESMPyException as esmf_error:
                pt.log_crit(
                    f"Unable to create {input_forcings.product_name} destination ESMF field object: {esmf_error}"
                )

        elif config_options.grid_type == "unstructured":
            try:
                input_forcings.esmf_field_out = pt.esmf_field_retry_partial(
                    wrf_hydro_geo_meta.esmf_grid,
                    meshloc=ESMF.MeshLoc.NODE,
                    name=f"{input_forcings.product_name}FORCING_REGRIDDED",
                )
            except ESMF.ESMPyException as esmf_error:
                pt.log_crit(
                    f"Unable to create {input_forcings.product_name} destination ESMF field node mesh object: {esmf_error}"
                )
            try:
                input_forcings.esmf_field_out_elem = pt.esmf_field_retry_partial(
                    wrf_hydro_geo_meta.esmf_grid,
                    meshloc=ESMF.MeshLoc.ELEMENT,
                    name=f"{input_forcings.product_name}FORCING_REGRIDDED",
                )
            except ESMF.ESMPyException as esmf_error:
                pt.log_crit(
                    f"Unable to create {input_forcings.product_name} destination ESMF field element mesh object: {esmf_error}"
                )
        elif config_options.grid_type == "hydrofabric":
            try:
                input_forcings.esmf_field_out = pt.esmf_field_retry_partial(
                    wrf_hydro_geo_meta.esmf_grid,
                    meshloc=ESMF.MeshLoc.ELEMENT,
                    name=f"{input_forcings.product_name}FORCING_REGRIDDED",
                )
            except ESMF.ESMPyException as esmf_error:
                pt.log_crit(
                    f"Unable to create {input_forcings.product_name} destination ESMF field element mesh object: {esmf_error}"
                )

    err_handler.check_program_status(config_options, mpi_config)

    # Determine if we need to calculate a regridding object. The following situations warrant the calculation of
    # a new weight file:
    # 1.) This is the first output time step, so we need to calculate a weight file.
    # 2.) The input forcing grid has changed.
    calc_regrid_flag = False
    # mpi_config.comm.barrier()

    if input_forcings.nx_global is None or input_forcings.ny_global is None:
        # This is the first timestep.
        # Create out regridded numpy arrays to hold the regridded data.
        force_count = 9 if config_options.include_lqfrac else 8
        if config_options.grid_type == "gridded":
            input_forcings.regridded_forcings1 = np.empty(
                [force_count, wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local],
                np.float32,
            )
            input_forcings.regridded_forcings2 = np.empty(
                [force_count, wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local],
                np.float32,
            )
        elif config_options.grid_type == "unstructured":
            input_forcings.regridded_forcings1 = np.empty(
                [force_count, wrf_hydro_geo_meta.ny_local], np.float32
            )
            input_forcings.regridded_forcings2 = np.empty(
                [force_count, wrf_hydro_geo_meta.ny_local], np.float32
            )
            input_forcings.regridded_forcings1_elem = np.empty(
                [force_count, wrf_hydro_geo_meta.ny_local_elem], np.float32
            )
            input_forcings.regridded_forcings2_elem = np.empty(
                [force_count, wrf_hydro_geo_meta.ny_local_elem], np.float32
            )
        elif config_options.grid_type == "hydrofabric":
            input_forcings.regridded_forcings1 = np.full(
                [force_count, wrf_hydro_geo_meta.ny_local], np.nan,dtype=np.float32 #NOTE changed to np.full to be deterministic for unit tests.
            )
            input_forcings.regridded_forcings2 = np.full(
                [force_count, wrf_hydro_geo_meta.ny_local], np.nan,dtype=np.float32 #NOTE changed to np.full to be deterministic for unit tests.
            )

    if mpi_config.rank == 0:
        if input_forcings.nx_global is None or input_forcings.ny_global is None:
            # This is the first timestep.
            calc_regrid_flag = True
        else:
            if mpi_config.rank == 0:
                if config_options.aws:
                    if (
                        id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ].shape[0]
                        != input_forcings.ny_global
                        and id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ].shape[1]
                        != input_forcings.nx_global
                    ):
                        calc_regrid_flag = True
                else:
                    if (
                        id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ].shape[1]
                        != input_forcings.ny_global
                        and id_tmp.variables[
                            input_forcings.netcdf_var_names[force_count]
                        ].shape[2]
                        != input_forcings.nx_global
                    ):
                        calc_regrid_flag = True

    # mpi_config.comm.barrier()

    # Broadcast the flag to the other processors.
    calc_regrid_flag = mpi_config.broadcast_parameter(
        calc_regrid_flag, config_options, param_type=bool
    )
    err_handler.check_program_status(config_options, mpi_config)

    return calc_regrid_flag


def check_supp_pcp_regrid_status(
    id_tmp, supplemental_precip, config_options, wrf_hydro_geo_meta, mpi_config
):
    """Check supplemental precipitation regrid status function.

    Function for checking to see if regridding weights need to be
    calculated (or recalculated).
    :param supplemental_precip:
    :param id_tmp:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    # If the destination ESMF field hasn't been created, create it here.
    if not supplemental_precip.esmf_field_out:
        if config_options.grid_type == "gridded":
            try:
                supplemental_precip.esmf_field_out = pt.esmf_field_retry_partial(
                    wrf_hydro_geo_meta.esmf_grid,
                    name=f"{supplemental_precip.product_name}SUPP_PCP_REGRIDDED",
                )
            except ESMF.ESMPyException as esmf_error:
                config_options.errMsg = f"Unable to create {supplemental_precip.product_name} destination ESMF field object: {esmf_error}"
                err_handler.err_out(config_options)
        elif config_options.grid_type == "unstructured":
            try:
                supplemental_precip.esmf_field_out = pt.esmf_field_retry_partial(
                    wrf_hydro_geo_meta.esmf_grid,
                    meshloc=ESMF.MeshLoc.NODE,
                    name=f"{supplemental_precip.product_name}SUPP_PCP_REGRIDDED",
                )
            except ESMF.ESMPyException as esmf_error:
                config_options.errMsg = f"Unable to create {supplemental_precip.product_name} destination ESMF node field object: {esmf_error}"
                err_handler.err_out(config_options)

            try:
                supplemental_precip.esmf_field_out_elem = pt.esmf_field_retry_partial(
                    wrf_hydro_geo_meta.esmf_grid,
                    meshloc=ESMF.MeshLoc.ELEMENT,
                    name=f"{supplemental_precip.product_name}SUPP_PCP_REGRIDDED",
                )
            except ESMF.ESMPyException as esmf_error:
                config_options.errMsg = f"Unable to create {supplemental_precip.product_name} destination ESMF element field object: {esmf_error}"
                err_handler.err_out(config_options)

        elif config_options.grid_type == "hydrofabric":
            try:
                supplemental_precip.esmf_field_out = pt.esmf_field_retry_partial(
                    wrf_hydro_geo_meta.esmf_grid,
                    meshloc=ESMF.MeshLoc.ELEMENT,
                    name=f"{supplemental_precip.product_name}SUPP_PCP_REGRIDDED",
                )
            except ESMF.ESMPyException as esmf_error:
                config_options.errMsg = f"Unable to create {supplemental_precip.product_name} destination ESMF element field object: {esmf_error}"
                err_handler.err_out(config_options)

    # Determine if we need to calculate a regridding object. The following situations warrant the calculation of
    # a new weight file:
    # 1.) This is the first output time step, so we need to calculate a weight file.
    # 2.) The input forcing grid has changed.
    calc_regrid_flag = False

    # mpi_config.comm.barrier()

    if supplemental_precip.nx_global is None or supplemental_precip.ny_global is None:
        if config_options.grid_type == "gridded":
            # This is the first timestep.
            # Create out regridded numpy arrays to hold the regridded data.
            supplemental_precip.regridded_precip1 = np.empty(
                [wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32
            )
            supplemental_precip.regridded_precip2 = np.empty(
                [wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32
            )
            supplemental_precip.regridded_rqi1 = np.empty(
                [wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32
            )
            supplemental_precip.regridded_rqi2 = np.empty(
                [wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32
            )
            supplemental_precip.regridded_rqi1[:, :] = config_options.globalNdv
            supplemental_precip.regridded_rqi2[:, :] = config_options.globalNdv
        elif config_options.grid_type == "unstructured":
            # This is the first timestep.
            # Create out regridded numpy arrays to hold the regridded data.
            supplemental_precip.regridded_precip1 = np.empty(
                [wrf_hydro_geo_meta.ny_local], np.float32
            )
            supplemental_precip.regridded_precip2 = np.empty(
                [wrf_hydro_geo_meta.ny_local], np.float32
            )
            supplemental_precip.regridded_rqi1 = np.empty(
                [wrf_hydro_geo_meta.ny_local], np.float32
            )
            supplemental_precip.regridded_rqi2 = np.empty(
                [wrf_hydro_geo_meta.ny_local], np.float32
            )
            supplemental_precip.regridded_rqi1[:] = config_options.globalNdv
            supplemental_precip.regridded_rqi2[:] = config_options.globalNdv
            supplemental_precip.regridded_precip1_elem = np.empty(
                [wrf_hydro_geo_meta.ny_local_elem], np.float32
            )
            supplemental_precip.regridded_precip2_elem = np.empty(
                [wrf_hydro_geo_meta.ny_local_elem], np.float32
            )
            supplemental_precip.regridded_rqi1_elem = np.empty(
                [wrf_hydro_geo_meta.ny_local_elem], np.float32
            )
            supplemental_precip.regridded_rqi2_elem = np.empty(
                [wrf_hydro_geo_meta.ny_local_elem], np.float32
            )
            supplemental_precip.regridded_rqi1_elem[:] = config_options.globalNdv
            supplemental_precip.regridded_rqi2_elem[:] = config_options.globalNdv

        elif config_options.grid_type == "hydrofabric":
            # This is the first timestep.
            # Create out regridded numpy arrays to hold the regridded data.
            supplemental_precip.regridded_precip1 = np.empty(
                [wrf_hydro_geo_meta.ny_local], np.float32
            )
            supplemental_precip.regridded_precip2 = np.empty(
                [wrf_hydro_geo_meta.ny_local], np.float32
            )
            supplemental_precip.regridded_rqi1 = np.empty(
                [wrf_hydro_geo_meta.ny_local], np.float32
            )
            supplemental_precip.regridded_rqi2 = np.empty(
                [wrf_hydro_geo_meta.ny_local], np.float32
            )
            supplemental_precip.regridded_rqi1[:] = config_options.globalNdv
            supplemental_precip.regridded_rqi2[:] = config_options.globalNdv

    if mpi_config.rank == 0:
        if (
            supplemental_precip.nx_global is None
            or supplemental_precip.ny_global is None
        ):
            # This is the first timestep.
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
                if (
                    ncvar.shape[latdim] != supplemental_precip.ny_global
                    and ncvar.shape[londim] != supplemental_precip.nx_global
                ):
                    calc_regrid_flag = True

    if config_options.grid_type == "gridded":
        # We will now check to see if the regridded arrays are still None. This means the fields were set to None
        # earlier for missing data. We need to reset them to nx_global/ny_global where the calc_regrid_flag is False.
        if supplemental_precip.regridded_precip2 is None:
            supplemental_precip.regridded_precip2 = np.empty(
                [wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32
            )
        if supplemental_precip.regridded_precip1 is None:
            supplemental_precip.regridded_precip1 = np.empty(
                [wrf_hydro_geo_meta.ny_local, wrf_hydro_geo_meta.nx_local], np.float32
            )
    elif config_options.grid_type == "unstructured":
        # We will now check to see if the regridded arrays are still None. This means the fields were set to None
        # earlier for missing data. We need to reset them to nx_global/ny_global where the calc_regrid_flag is False.
        if supplemental_precip.regridded_precip2 is None:
            supplemental_precip.regridded_precip2 = np.empty(
                [wrf_hydro_geo_meta.ny_local], np.float32
            )
        if supplemental_precip.regridded_precip1 is None:
            supplemental_precip.regridded_precip1 = np.empty(
                [wrf_hydro_geo_meta.ny_local], np.float32
            )
        if supplemental_precip.regridded_precip2_elem is None:
            supplemental_precip.regridded_precip2_elem = np.empty(
                [wrf_hydro_geo_meta.ny_local_elem], np.float32
            )
        if supplemental_precip.regridded_precip1_elem is None:
            supplemental_precip.regridded_precip1_elem = np.empty(
                [wrf_hydro_geo_meta.ny_local_elem], np.float32
            )

    elif config_options.grid_type == "hydrofabric":
        # We will now check to see if the regridded arrays are still None. This means the fields were set to None
        # earlier for missing data. We need to reset them to nx_global/ny_global where the calc_regrid_flag is False.
        if supplemental_precip.regridded_precip2 is None:
            supplemental_precip.regridded_precip2 = np.empty(
                [wrf_hydro_geo_meta.ny_local], np.float32
            )
        if supplemental_precip.regridded_precip1 is None:
            supplemental_precip.regridded_precip1 = np.empty(
                [wrf_hydro_geo_meta.ny_local], np.float32
            )

    # mpi_config.comm.barrier()

    # Broadcast the flag to the other processors.
    calc_regrid_flag = mpi_config.broadcast_parameter(
        calc_regrid_flag, config_options, param_type=bool
    )

    mpi_config.comm.barrier()
    return calc_regrid_flag


def get_weight_file_names(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    input_forcings: GeoMeta,
) -> tuple[str | None, str | None]:
    """Get weight file names for regridding."""
    if not config_options.weightsDir:
        return None, None

    grid_key = input_forcings.product_name
    file_key = f"{grid_key}_{config_options.geogrid}"
    hash_key = hashlib.md5(file_key.encode()).hexdigest()[:8]
    hash_key += f"_{mpi_config.uid64}"

    weight_file = os.path.join(config_options.weightsDir, f"ESMF_weight_{hash_key}.nc4")

    if config_options.grid_type == "unstructured":
        weight_file_elem = os.path.join(
            config_options.weightsDir, f"ESMF_weight_{hash_key}_elem.nc4"
        )
    else:
        weight_file_elem = None

    return weight_file, weight_file_elem


def load_weight_file(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    input_forcings: GeoMeta,
    weight_file: str,
    element_mode: bool,
) -> None:
    """`input_forcings.regridObj` or `input_forcings.regridObj_elem` is modified in-place."""
    pt = Partials(mpi_config, config_options)
    os_utils.assert_path_exists_retry(
        mpi_config, config_options, err_handler, weight_file
    )

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

    pt.log_debug(
        f"RANK: {mpi_config.rank}: Loading cached ESMF{msg_augment}weight object for {input_forcings.product_name} from {weight_file}"
    )

    err_handler.check_program_status(config_options, mpi_config)
    try:
        begin = monotonic()
        regrid = esmf_regridfromfile_retry(
            mpi_config, config_options, err_handler, field_in, field_out, weight_file
        )
        setattr(input_forcings, target_object_attr_name, regrid)
        end = monotonic()
    except (IOError, ValueError, ESMF.ESMPyException) as esmf_error:
        pt.log_warn(f"Unable to load cached ESMF{msg_augment}weight file: {esmf_error}")
    else:
        pt.log_debug(
            f"RANK: {mpi_config.rank}: Finished loading{msg_augment}weight object with ESMF, took {end - begin} seconds"
        )

    err_handler.check_program_status(config_options, mpi_config)


def make_regrid(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    input_forcings: GeoMeta,
    weight_file: str | None,
    fill: bool,
    element_mode: bool,
) -> None:
    """Make ESMF regrid object.

    `input_forcings.regridObj` or `input_forcings.regridObj_elem` is modified in-place.
    Writes weight file to disk if weight_file is not None.
    Operates on element object if `element_mode` is True.
    """
    pt = Partials(mpi_config, config_options)
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
        pt.log_debug(start_msg)

    extrap_method = ESMF.ExtrapMethod.CREEP_FILL if fill else ESMF.ExtrapMethod.NONE
    regrid_method = (ESMF.RegridMethod.BILINEAR, ESMF.RegridMethod.NEAREST_STOD)[
        input_forcings.regrid_opt - 1
    ]

    err_handler.check_program_status(config_options, mpi_config)
    try:
        begin = monotonic()
        regrid = esmf_regrid_retry(
            mpi_config,
            config_options,
            err_handler,
            field_in,
            field_out,
            src_mask_values=np.array([0, config_options.globalNdv]),
            regrid_method=regrid_method,
            extrap_method=extrap_method,
            unmapped_action=ESMF.UnmappedAction.IGNORE,
            filename=weight_file,
        )
        setattr(input_forcings, target_object_attr_name, regrid)
        end = monotonic()
    except (RuntimeError, ImportError, ESMF.ESMPyException) as esmf_error:
        pt.log_crit(
            f"RANK: {mpi_config.rank}: Failed: {start_msg}. Unable to regrid input data from ESMF: {esmf_error}"
        )
        etype, value, tb = sys.exc_info()
        traceback.print_exception(etype, value, tb)
    else:
        if mpi_config.rank == 0:
            pt.log_debug(
                f"RANK: {mpi_config.rank}: Finished: {start_msg}, took {end - begin} seconds"
            )

    err_handler.check_program_status(config_options, mpi_config)


def execute_regrid(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    input_forcings: GeoMeta,
    weight_file: str,
    element_mode: bool,
) -> None:
    """Input_forcings.esmf_field_out` or `input_forcings.esmf_field_out_elem` is modified in-place.

    On error, weight file is deleted from disk.
    """
    pt = Partials(mpi_config, config_options)
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
        setattr(
            input_forcings, target_object_attr_name, regrid_object(field_in, field_out)
        )
    except ValueError as ve:
        pt.log_crit(f"Unable to extract regridded data from ESMF regridded field: {ve}")
        # TODO confirm intent for all ranks to call this
        pt.os_remove_anyrank_partial(weight_file)

    err_handler.check_program_status(config_options, mpi_config)


def calculate_weights(
    id_tmp,
    force_count,
    input_forcings,
    config_options,
    mpi_config,
    wrf_hydro_geo_meta,
    lat_var="latitude",
    lon_var="longitude",
    fill=False,
):
    """Calculate weights function.

    Function to calculate ESMF weights based on the output ESMF
    field previously calculated, along with input lat/lon grids,
    and a sample dataset.
    :param input_forcings:
    :param id_tmp:
    :param mpi_config:
    :param config_options:
    :param force_count:
    :return:
    """
    pt = Partials(mpi_config, config_options)

    pt.log_debug("Calculate Weights")

    if mpi_config.rank == 0:
        if config_options.aws:
            try:
                input_forcings.ny_global = id_tmp.variables[
                    input_forcings.netcdf_var_names[force_count]
                ].shape[0]
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to extract Y shape size from: {input_forcings.netcdf_var_names[force_count]} from aws object"
                )
            try:
                input_forcings.nx_global = id_tmp.variables[
                    input_forcings.netcdf_var_names[force_count]
                ].shape[1]
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to extract X shape size from: {input_forcings.netcdf_var_names[force_count]} from aws object"
                )
        else:
            try:
                input_forcings.ny_global = id_tmp.variables[
                    input_forcings.netcdf_var_names[force_count]
                ].shape[1]
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to extract Y shape size from: {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                )
            try:
                input_forcings.nx_global = id_tmp.variables[
                    input_forcings.netcdf_var_names[force_count]
                ].shape[2]
            except (ValueError, KeyError, AttributeError) as err:
                pt.log_crit(
                    f"Unable to extract X shape size from: {input_forcings.netcdf_var_names[force_count]} from: {input_forcings.tmpFile} ({err})"
                )
    err_handler.check_program_status(config_options, mpi_config)

    # Broadcast the forcing nx/ny values
    input_forcings.ny_global = mpi_config.broadcast_parameter(
        input_forcings.ny_global, config_options, param_type=int
    )
    err_handler.check_program_status(config_options, mpi_config)
    input_forcings.nx_global = mpi_config.broadcast_parameter(
        input_forcings.nx_global, config_options, param_type=int
    )
    err_handler.check_program_status(config_options, mpi_config)

    try:
        # noinspection PyTypeChecker
        input_forcings.esmf_grid_in = pt.esmf_grid_retry_partial(
            np.array([input_forcings.ny_global, input_forcings.nx_global]),
            staggerloc=ESMF.StaggerLoc.CENTER,
            coord_sys=ESMF.CoordSys.SPH_DEG,
        )
    except ESMF.ESMPyException as esmf_error:
        pt.log_crit(
            f"Unable to create source ESMF grid from temporary file: {input_forcings.tmpFile} ({esmf_error})"
        )
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.x_lower_bound = input_forcings.esmf_grid_in.lower_bounds[
            ESMF.StaggerLoc.CENTER
        ][1]
        input_forcings.x_upper_bound = input_forcings.esmf_grid_in.upper_bounds[
            ESMF.StaggerLoc.CENTER
        ][1]
        input_forcings.y_lower_bound = input_forcings.esmf_grid_in.lower_bounds[
            ESMF.StaggerLoc.CENTER
        ][0]
        input_forcings.y_upper_bound = input_forcings.esmf_grid_in.upper_bounds[
            ESMF.StaggerLoc.CENTER
        ][0]
        input_forcings.nx_local = (
            input_forcings.x_upper_bound - input_forcings.x_lower_bound
        )
        input_forcings.ny_local = (
            input_forcings.y_upper_bound - input_forcings.y_lower_bound
        )
    except (ValueError, KeyError, AttributeError) as err:
        pt.log_crit(
            f"Unable to extract local X/Y boundaries from global grid from temporary file: {input_forcings.tmpFile} ({err})"
        )
    err_handler.check_program_status(config_options, mpi_config)

    # Check to make sure we have enough dimensionality to run regridding. ESMF requires both grids
    # to have a size of at least 2.
    if input_forcings.nx_local < 2 or input_forcings.ny_local < 2:
        pt.log_crit(
            f"You have either specified too many cores for: {input_forcings.product_name}, or  your input forcing grid is too small to process. Local grid must have x/y dimension size of 2."
        )
    err_handler.check_program_status(config_options, mpi_config)

    # check if we're doing border trimming and set up mask
    border = input_forcings.ignored_border_widths  # // 5  # HRRR is a 3 km product
    if border > 0:
        try:
            mask = input_forcings.esmf_grid_in.add_item(
                ESMF.GridItem.MASK, ESMF.StaggerLoc.CENTER
            )
            if mpi_config.rank == 0:
                pt.log_debug(
                    f"Trimming input forcing `{input_forcings.product_name}` by {border} grid cells"
                )

            gmask = np.ones([input_forcings.ny_global, input_forcings.nx_global])
            gmask[:+border, :] = 0.0  # top edge
            gmask[-border:, :] = 0.0  # bottom edge
            gmask[:, :+border] = 0.0  # left edge
            gmask[:, -border:] = 0.0  # right edge

            mask[:, :] = mpi_config.scatter_array(input_forcings, gmask, config_options)
            err_handler.check_program_status(config_options, mpi_config)
        except Exception as e:
            LOG.error(f"{e}")

    lat_tmp = None
    lon_tmp = None
    if mpi_config.rank == 0:
        if input_forcings.product_name == "NWM":
            nwm_geogrid = nc.Dataset(config_options.nwm_geogrid)

            # Get spatial bounds from aws_obj if available
            if (
                hasattr(config_options, "aws_obj")
                and config_options.aws_obj is not None
            ):
                # Extract subset indices from the aws_obj coordinates
                aws_x = config_options.aws_obj.x.values
                aws_y = config_options.aws_obj.y.values

                # Get full geogrid coordinates
                full_lat = nwm_geogrid.variables["XLAT_M"][:][0, :, :]
                full_lon = nwm_geogrid.variables["XLONG_M"][:][0, :, :]

                # Find matching indices in geogrid that correspond to aws_obj bounds
                # TODO: may be able to get nwm_crs from the zarr metadata
                nwm_crs = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
                transformer = Transformer.from_crs(nwm_crs, "EPSG:4326", always_xy=True)

                x_min_proj, x_max_proj = aws_x.min(), aws_x.max()
                y_min_proj, y_max_proj = aws_y.min(), aws_y.max()

                # Convert aws projected bounds back to geographic to match geogrid
                lon_min, lat_min = transformer.transform(x_min_proj, y_min_proj)
                lon_max, lat_max = transformer.transform(x_max_proj, y_max_proj)

                buffer = 0.1  # degrees
                lat_min -= buffer
                lat_max += buffer
                lon_min -= buffer
                lon_max += buffer

                # Find subset indices in geogrid
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
            # Process lat/lon values from the GFS grid.
            if len(id_tmp.variables[lat_var].shape) == 3:
                # We have 2D grids already in place.
                lat_tmp = id_tmp.variables[lat_var][0, :, :]
                lon_tmp = id_tmp.variables[lon_var][0, :, :]
            elif len(id_tmp.variables[lon_var].shape) == 2:
                # We have 2D grids already in place.
                lat_tmp = id_tmp.variables[lat_var][:, :]
                lon_tmp = id_tmp.variables[lon_var][:, :]
            elif len(id_tmp.variables[lat_var].shape) == 1:
                # We have 1D lat/lons we need to translate into
                # 2D grids, one which would come from AORC AWS
                # s3 bucket data we need to flag here for
                if config_options.aws:
                    lat_tmp = id_tmp.variables[lat_var][:].values
                    lon_tmp = id_tmp.variables[lon_var][:].values
                else:
                    lat_tmp = id_tmp.variables[lat_var][:]
                    lon_tmp = id_tmp.variables[lon_var][:]
                lon_tmp, lat_tmp = np.meshgrid(lon_tmp, lat_tmp)

    err_handler.check_program_status(config_options, mpi_config)

    # Scatter global GFS latitude grid to processors..
    if mpi_config.rank == 0:
        var_tmp = lat_tmp
    else:
        var_tmp = None
    var_sub_lat_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        var_tmp = lon_tmp
    else:
        var_tmp = None
    var_sub_lon_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.esmf_lats = input_forcings.esmf_grid_in.get_coords(1)
    except ESMF.GridException as ge:
        pt.log_crit(
            f"Unable to locate latitude coordinate object within input ESMF grid: {ge}"
        )
    err_handler.check_program_status(config_options, mpi_config)

    try:
        input_forcings.esmf_lons = input_forcings.esmf_grid_in.get_coords(0)
    except ESMF.GridException as ge:
        pt.log_crit(
            f"Unable to locate longitude coordinate object within input ESMF grid: {ge}"
        )
    err_handler.check_program_status(config_options, mpi_config)

    input_forcings.esmf_lats[:, :] = var_sub_lat_tmp
    input_forcings.esmf_lons[:, :] = var_sub_lon_tmp
    del var_sub_lat_tmp
    del var_sub_lon_tmp
    del lat_tmp
    del lon_tmp

    if config_options.grid_type == "gridded":
        # Create a ESMF field to hold the incoming data.
        try:
            input_forcings.esmf_field_in = pt.esmf_field_retry_partial(
                input_forcings.esmf_grid_in,
                name=f"{input_forcings.product_name}_NATIVE",
            )
        except ESMF.ESMPyException as esmf_error:
            pt.log_crit(f"Unable to create ESMF field object: {esmf_error}")
        err_handler.check_program_status(config_options, mpi_config)

    if config_options.grid_type == "unstructured":
        # Create a ESMF field to hold the incoming data.
        try:
            input_forcings.esmf_field_in = pt.esmf_field_retry_partial(
                input_forcings.esmf_grid_in,
                name=f"{input_forcings.product_name}_NATIVE",
            )
        except ESMF.ESMPyException as esmf_error:
            pt.log_crit(f"Unable to create ESMF field object: {esmf_error}")
        err_handler.check_program_status(config_options, mpi_config)

        # Create a ESMF field to hold the incoming data.
        try:
            input_forcings.esmf_field_in_elem = pt.esmf_field_retry_partial(
                input_forcings.esmf_grid_in,
                name=f"{input_forcings.product_name}_NATIVE_ELEMENT",
            )
        except ESMF.ESMPyException as esmf_error:
            pt.log_crit(f"Unable to create ESMF field object: {esmf_error}")
        err_handler.check_program_status(config_options, mpi_config)

    elif config_options.grid_type == "hydrofabric":
        # Create a ESMF field to hold the incoming data.
        try:
            input_forcings.esmf_field_in = pt.esmf_field_retry_partial(
                input_forcings.esmf_grid_in,
                name=f"{input_forcings.product_name}_NATIVE",
            )
        except ESMF.ESMPyException as esmf_error:
            pt.log_crit(f"Unable to create ESMF field object: {esmf_error}")
        err_handler.check_program_status(config_options, mpi_config)

    # Scatter global grid to processors..
    if mpi_config.rank == 0:
        if config_options.aws:
            var_tmp = id_tmp[
                input_forcings.netcdf_var_names[force_count]
            ].to_masked_array()
        else:
            var_tmp = id_tmp[input_forcings.netcdf_var_names[force_count]][0, :, :]
        # Set all valid values to 1, and all missing values to 0. This will
        # be used to generate an output mask that is used later on in downscaling, layering, etc.
        var_tmp.fill(1)
        var_tmp = var_tmp.filled(0)

        if input_forcings.product_name == "NWM":
            var_tmp = np.asarray(var_tmp, dtype=np.float64)  # or np.float32
    else:
        var_tmp = None
    var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
    err_handler.check_program_status(config_options, mpi_config)

    if config_options.grid_type == "gridded":
        # Place temporary data into the field array for generating the regridding object.
        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp

    elif config_options.grid_type == "unstructured":
        # Place temporary data into the field array for generating the regridding object.
        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
        input_forcings.esmf_field_in_elem.data[:, :] = var_sub_tmp

    elif config_options.grid_type == "hydrofabric":
        # Place temporary data into the field array for generating the regridding object.
        input_forcings.esmf_field_in.data[:, :] = var_sub_tmp

    # mpi_config.comm.barrier()

    # ## CALCULATE WEIGHT ## #
    common_args = (mpi_config, config_options, input_forcings)
    weight_file, weight_file_elem = get_weight_file_names(*common_args)

    # If regrid object has not been initialized yet, initialize it.
    if input_forcings.regridObj is None:
        # make_regrid's call to Regrid() is an implicit MPI barrier, all ranks must call it.
        if config_options.weightsDir is not None:
            if not os.path.exists(weight_file):
                make_regrid(
                    *common_args, weight_file=weight_file, fill=fill, element_mode=False
                )

            if config_options.grid_type == "unstructured" and (
                not os.path.exists(weight_file_elem)
            ):
                make_regrid(
                    *common_args,
                    weight_file=weight_file_elem,
                    fill=fill,
                    element_mode=True,
                )

            load_weight_file(*common_args, weight_file, element_mode=False)
            if config_options.grid_type == "unstructured":
                load_weight_file(*common_args, weight_file, element_mode=True)
        else:
            # Make regrid object in memory without writing files
            make_regrid(*common_args, weight_file=None, fill=fill, element_mode=False)
            if config_options.grid_type == "unstructured":
                make_regrid(
                    *common_args, weight_file=None, fill=fill, element_mode=True
                )

    execute_regrid(*common_args, weight_file, element_mode=False)
    if config_options.grid_type == "gridded":
        input_forcings.regridded_mask[:, :] = np.round(
            input_forcings.esmf_field_out.data[:, :]
        )
    elif config_options.grid_type != "gridded":
        input_forcings.regridded_mask[:] = np.round(
            input_forcings.esmf_field_out.data[:]
        )

    if config_options.grid_type == "unstructured":
        execute_regrid(*common_args, weight_file, element_mode=True)
        input_forcings.regridded_mask_elem[:] = np.round(
            input_forcings.esmf_field_out_elem.data[:]
        )

    err_handler.check_program_status(config_options, mpi_config)


def calculate_supp_pcp_weights(
    supplemental_precip,
    id_tmp,
    tmp_file,
    config_options,
    mpi_config,
    lat_var="latitude",
    lon_var="longitude",
):
    """Calculate supplemental precip ESMF weights based on the output ESMF.

    Function to calculate ESMF weights based on the output ESMF
    field previously calculated, along with input lat/lon grids,
    and a sample dataset.
    :param tmp_file:
    :param id_tmp:
    :param supplemental_precip:
    :param mpi_config:
    :param config_options:
    :return:
    """
    pt = Partials(mpi_config, config_options)

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
            config_options.errMsg = (
                f"Unable to determine lat/lon grid size from {tmp_file}"
            )
            err_handler.err_out(config_options)

        try:
            supplemental_precip.ny_global = id_tmp.variables[
                supplemental_precip.netcdf_var_names[0]
            ].shape[latdim]
        except (ValueError, KeyError, AttributeError, Exception) as err:
            config_options.errMsg = f"Unable to extract Y shape size from: {supplemental_precip.netcdf_var_names[0]} from: {tmp_file} ({err}, {type(err)})"
            err_handler.err_out(config_options)
        try:
            supplemental_precip.nx_global = id_tmp.variables[
                supplemental_precip.netcdf_var_names[0]
            ].shape[londim]
        except (ValueError, KeyError, AttributeError, Exception) as err:
            config_options.errMsg = f"Unable to extract X shape size from: {supplemental_precip.netcdf_var_names[0]} from: {tmp_file} ({err}, {type(err)})"
            err_handler.err_out(config_options)

    # mpi_config.comm.barrier()

    # Broadcast the forcing nx/ny values
    supplemental_precip.ny_global = mpi_config.broadcast_parameter(
        supplemental_precip.ny_global, config_options, param_type=int
    )
    supplemental_precip.nx_global = mpi_config.broadcast_parameter(
        supplemental_precip.nx_global, config_options, param_type=int
    )
    # mpi_config.comm.barrier()

    try:
        # noinspection PyTypeChecker
        supplemental_precip.esmf_grid_in = pt.esmf_grid_retry_partial(
            np.array([supplemental_precip.ny_global, supplemental_precip.nx_global]),
            staggerloc=ESMF.StaggerLoc.CENTER,
            coord_sys=ESMF.CoordSys.SPH_DEG,
        )
    except ESMF.ESMPyException as esmf_error:
        config_options.errMsg = f"Unable to create source ESMF grid from temporary file: {tmp_file} ({esmf_error})"
        err_handler.err_out(config_options)
    # mpi_config.comm.barrier()

    try:
        supplemental_precip.x_lower_bound = (
            supplemental_precip.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][1]
        )
        supplemental_precip.x_upper_bound = (
            supplemental_precip.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][1]
        )
        supplemental_precip.y_lower_bound = (
            supplemental_precip.esmf_grid_in.lower_bounds[ESMF.StaggerLoc.CENTER][0]
        )
        supplemental_precip.y_upper_bound = (
            supplemental_precip.esmf_grid_in.upper_bounds[ESMF.StaggerLoc.CENTER][0]
        )
        supplemental_precip.nx_local = (
            supplemental_precip.x_upper_bound - supplemental_precip.x_lower_bound
        )
        supplemental_precip.ny_local = (
            supplemental_precip.y_upper_bound - supplemental_precip.y_lower_bound
        )
    except (ValueError, KeyError, AttributeError) as err:
        config_options.errMsg = f"Unable to extract local X/Y boundaries from global grid from temporary file: {tmp_file} ({err})"
        err_handler.err_out(config_options)
    # mpi_config.comm.barrier()

    # Check to make sure we have enough dimensionality to run regridding. ESMF requires both grids
    # to have a size of at least 2.
    if supplemental_precip.nx_local < 2 or supplemental_precip.ny_local < 2:
        pt.log_crit(
            f"You have either specified too many cores for: {supplemental_precip.product_name}, or  your input forcing grid is too small to process. Local grid must have x/y dimension size of 2."
        )
    err_handler.check_program_status(config_options, mpi_config)

    lat_tmp = lon_tmp = None
    if mpi_config.rank == 0:
        # Process lat/lon values from the GFS grid.
        if len(id_tmp.variables[lat_var].shape) == 3:
            # We have 2D grids already in place.
            lat_tmp = id_tmp.variables[lat_var][0, :]
            lon_tmp = id_tmp.variables[lon_var][0, :]
        elif len(id_tmp.variables[lon_var].shape) == 2:
            # We have 2D grids already in place.
            lat_tmp = id_tmp.variables[lat_var][:]
            lon_tmp = id_tmp.variables[lon_var][:]
        elif len(id_tmp.variables[lat_var].shape) == 1:
            # We have 1D lat/lons we need to translate into
            # 2D grids.
            lat_tmp = np.repeat(
                id_tmp.variables[lat_var][:][:, np.newaxis],
                supplemental_precip.nx_global,
                axis=1,
            )
            lon_tmp = np.tile(
                id_tmp.variables[lon_var][:], (supplemental_precip.ny_global, 1)
            )
    # mpi_config.comm.barrier()

    # Scatter global GFS latitude grid to processors..
    if mpi_config.rank == 0:
        var_tmp = lat_tmp
    else:
        var_tmp = None
    var_sub_lat_tmp = mpi_config.scatter_array(
        supplemental_precip, var_tmp, config_options
    )
    # mpi_config.comm.barrier()

    if mpi_config.rank == 0:
        var_tmp = lon_tmp
    else:
        var_tmp = None
    var_sub_lon_tmp = mpi_config.scatter_array(
        supplemental_precip, var_tmp, config_options
    )
    # mpi_config.comm.barrier()

    try:
        supplemental_precip.esmf_lats = supplemental_precip.esmf_grid_in.get_coords(1)
    except ESMF.GridException as ge:
        config_options.errMsg = f"Unable to locate latitude coordinate object within supplemental precip ESMF grid: {ge}"
        err_handler.err_out(config_options)
    # mpi_config.comm.barrier()

    try:
        supplemental_precip.esmf_lons = supplemental_precip.esmf_grid_in.get_coords(0)
    except ESMF.GridException as ge:
        config_options.errMsg = f"Unable to locate longitude coordinate object within supplemental precip ESMF grid: {ge}"
        err_handler.err_out(config_options)
    # mpi_config.comm.barrier()

    supplemental_precip.esmf_lats[:, :] = var_sub_lat_tmp
    supplemental_precip.esmf_lons[:, :] = var_sub_lon_tmp
    del var_sub_lat_tmp
    del var_sub_lon_tmp
    del lat_tmp
    del lon_tmp

    if config_options.grid_type == "gridded":
        # Create a ESMF field to hold the incoming data.
        supplemental_precip.esmf_field_in = pt.esmf_field_retry_partial(
            supplemental_precip.esmf_grid_in,
            name=f"{supplemental_precip.product_name}_NATIVE",
        )

        # mpi_config.comm.barrier()

        # Scatter global grid to processors..
        if mpi_config.rank == 0:
            if ndims == 3:
                var_tmp = id_tmp[supplemental_precip.netcdf_var_names[0]][0, :]
            elif ndims == 2:
                var_tmp = id_tmp[supplemental_precip.netcdf_var_names[0]][:]
            else:
                var_tmp = None
            # Set all valid values to 1.0, and all missing values to 0.0. This will
            # be used to generate an output mask that is used later on in downscaling, layering,
            # etc.
            var_tmp[:] = np.where(
                var_tmp == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                0.0,
                1.0,
            )
        else:
            var_tmp = None
        var_sub_tmp = mpi_config.scatter_array(
            supplemental_precip, var_tmp, config_options
        )
        mpi_config.comm.barrier()

        # Place temporary data into the field array for generating the regridding object.
        supplemental_precip.esmf_field_in.data[:] = var_sub_tmp
        # mpi_config.comm.barrier()

        supplemental_precip.regridObj = pt.esmf_regrid_retry_partial(
            supplemental_precip.esmf_field_in,
            supplemental_precip.esmf_field_out,
            src_mask_values=np.array([0]),
            regrid_method=ESMF.RegridMethod.BILINEAR,
            unmapped_action=ESMF.UnmappedAction.IGNORE,
        )

        # Run the regridding object on this test dataset. Check the output grid for
        # any 0 values.
        supplemental_precip.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
            supplemental_precip.regridObj,
            supplemental_precip.esmf_field_in,
            supplemental_precip.esmf_field_out,
        )
        supplemental_precip.regridded_mask[:] = supplemental_precip.esmf_field_out.data[
            :
        ]

    elif config_options.grid_type == "unstructured":
        # Create a ESMF field to hold the incoming data.
        supplemental_precip.esmf_field_in = pt.esmf_field_retry_partial(
            supplemental_precip.esmf_grid_in,
            name=f"{supplemental_precip.product_name}_NATIVE",
        )

        # mpi_config.comm.barrier()

        # Scatter global grid to processors..
        if mpi_config.rank == 0:
            if ndims == 3:
                var_tmp = id_tmp[supplemental_precip.netcdf_var_names[0]][0, :].data
            elif ndims == 2:
                var_tmp = id_tmp[supplemental_precip.netcdf_var_names[0]][:].data
            else:
                var_tmp = None
            # Set all valid values to 1.0, and all missing values to 0.0. This will
            # be used to generate an output mask that is used later on in downscaling, layering,
            # etc.
            var_tmp[:] = np.where(
                var_tmp == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                0.0,
                1.0,
            )
        else:
            var_tmp = None
        var_sub_tmp = mpi_config.scatter_array(
            supplemental_precip, var_tmp, config_options
        )
        mpi_config.comm.barrier()

        # Place temporary data into the field array for generating the regridding object.
        supplemental_precip.esmf_field_in.data[:] = var_sub_tmp
        # mpi_config.comm.barrier()

        if supplemental_precip.regridOpt == 1:
            supplemental_precip.regridObj = pt.esmf_regrid_retry_partial(
                supplemental_precip.esmf_field_in,
                supplemental_precip.esmf_field_out,
                src_mask_values=np.array([0]),
                regrid_method=ESMF.RegridMethod.BILINEAR,
                unmapped_action=ESMF.UnmappedAction.IGNORE,
            )
        elif supplemental_precip.regridOpt == 2:
            supplemental_precip.regridObj = pt.esmf_regrid_retry_partial(
                supplemental_precip.esmf_field_in,
                supplemental_precip.esmf_field_out,
                src_mask_values=np.array([0]),
                regrid_method=ESMF.RegridMethod.NEAREST_STOD,
                unmapped_action=ESMF.UnmappedAction.IGNORE,
            )

        # Run the regridding object on this test dataset. Check the output grid for
        # any 0 values.
        supplemental_precip.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
            supplemental_precip.regridObj,
            supplemental_precip.esmf_field_in,
            supplemental_precip.esmf_field_out,
        )
        supplemental_precip.regridded_mask[:] = supplemental_precip.esmf_field_out.data[
            :
        ]

        # Create a ESMF field to hold the incoming data.
        supplemental_precip.esmf_field_in_elem = pt.esmf_field_retry_partial(
            supplemental_precip.esmf_grid_in,
            name=f"{supplemental_precip.product_name}_NATIVE",
        )

        # mpi_config.comm.barrier()

        # Scatter global grid to processors..
        if mpi_config.rank == 0:
            if ndims == 3:
                var_tmp_elem = id_tmp[supplemental_precip.netcdf_var_names[0]][
                    0, :
                ].data
            elif ndims == 2:
                var_tmp_elem = id_tmp[supplemental_precip.netcdf_var_names[0]][:].data
            else:
                var_tmp_elem = None
            # Set all valid values to 1.0, and all missing values to 0.0. This will
            # be used to generate an output mask that is used later on in downscaling, layering,
            # etc.
            var_tmp[:] = np.where(
                var_tmp == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                0.0,
                1.0,
            )
        else:
            var_tmp_elem = None
        var_sub_tmp_elem = mpi_config.scatter_array(
            supplemental_precip, var_tmp_elem, config_options
        )
        mpi_config.comm.barrier()

        # Place temporary data into the field array for generating the regridding object.
        supplemental_precip.esmf_field_in_elem.data[:] = var_sub_tmp_elem
        # mpi_config.comm.barrier()

        supplemental_precip.regridObj_elem = pt.esmf_regrid_retry_partial(
            supplemental_precip.esmf_field_in_elem,
            supplemental_precip.esmf_field_out_elem,
            src_mask_values=np.array([0]),
            regrid_method=ESMF.RegridMethod.BILINEAR,
            unmapped_action=ESMF.UnmappedAction.IGNORE,
        )

        # Run the regridding object on this test dataset. Check the output grid for
        # any 0 values.
        supplemental_precip.esmf_field_out_elem = pt.esmf_regridobj_call_retry_partial(
            supplemental_precip.regridObj_elem,
            supplemental_precip.esmf_field_in_elem,
            supplemental_precip.esmf_field_out_elem,
        )
        supplemental_precip.regridded_mask_elem[:] = (
            supplemental_precip.esmf_field_out_elem.data[:]
        )

    elif config_options.grid_type == "hydrofabric":
        # Create a ESMF field to hold the incoming data.
        supplemental_precip.esmf_field_in = pt.esmf_field_retry_partial(
            supplemental_precip.esmf_grid_in,
            name=f"{supplemental_precip.product_name}_NATIVE",
        )

        # mpi_config.comm.barrier()

        # Scatter global grid to processors..
        if mpi_config.rank == 0:
            if ndims == 3:
                var_tmp = id_tmp[supplemental_precip.netcdf_var_names[0]][0, :]
            elif ndims == 2:
                var_tmp = id_tmp[supplemental_precip.netcdf_var_names[0]][:]
            else:
                var_tmp = None
            # Set all valid values to 1.0, and all missing values to 0.0. This will
            # be used to generate an output mask that is used later on in downscaling, layering,
            # etc.
            var_tmp[:] = np.where(
                var_tmp == id_tmp[supplemental_precip.netcdf_var_names[0]]._FillValue,
                0.0,
                1.0,
            )
            var_tmp[:] = np.where(var_tmp == 9.999e20, 0.0, 1.0)
        else:
            var_tmp = None
        var_sub_tmp = mpi_config.scatter_array(
            supplemental_precip, var_tmp, config_options
        )
        mpi_config.comm.barrier()

        # Place temporary data into the field array for generating the regridding object.
        supplemental_precip.esmf_field_in.data[:] = var_sub_tmp
        # mpi_config.comm.barrier()
        if supplemental_precip.regridOpt == 2:
            supplemental_precip.regridObj = pt.esmf_regrid_retry_partial(
                supplemental_precip.esmf_field_in,
                supplemental_precip.esmf_field_out,
                src_mask_values=np.array([0]),
                regrid_method=ESMF.RegridMethod.NEAREST_STOD,
                unmapped_action=ESMF.UnmappedAction.IGNORE,
                extrap_method=ESMF.ExtrapMethod.NEAREST_STOD,
            )
        elif supplemental_precip.regridOpt == 1:
            supplemental_precip.regridObj = pt.esmf_regrid_retry_partial(
                supplemental_precip.esmf_field_in,
                supplemental_precip.esmf_field_out,
                src_mask_values=np.array([0]),
                regrid_method=ESMF.RegridMethod.BILINEAR,
                unmapped_action=ESMF.UnmappedAction.IGNORE,
                extrap_method=ESMF.ExtrapMethod.NEAREST_STOD,
            )

        # Run the regridding object on this test dataset. Check the output grid for
        # any 0 values.
        supplemental_precip.esmf_field_out = pt.esmf_regridobj_call_retry_partial(
            supplemental_precip.regridObj,
            supplemental_precip.esmf_field_in,
            supplemental_precip.esmf_field_out,
        )
        supplemental_precip.regridded_mask[:] = supplemental_precip.esmf_field_out.data[
            :
        ]
