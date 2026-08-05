"""Module file for handling of downscaling input regridded (possibly bias-corrected) forcing fields.

Each input forcing product will loop through and determine if
downscaling is needed, based off options specified by the user.
"""

from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from netCDF4 import Dataset

from . import err_handler

if TYPE_CHECKING:
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.config import (
        ConfigOptions,
    )
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.forcingInputMod import (
        InputForcings,
    )
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.geoMod import (
        GeoMeta,
    )
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.parallel import (
        MpiConfig,
    )


def run_downscaling(
    input_forcings: InputForcings,
    config_options: ConfigOptions,
    geo_meta: GeoMeta,
    mpi_config: MpiConfig,
):
    """Top level module function that will downscale forcing variables for this particular input forcing product.

    :param geo_meta:
    :param mpi_config:
    :param input_forcings:
    :param config_options:
    :return:
    """
    # Dictionary mapping to temperature downscaling.
    downscale_temperature = {0: no_downscale, 1: simple_lapse, 2: param_lapse}
    downscale_temperature[input_forcings.t2dDownscaleOpt](
        input_forcings, config_options, geo_meta, mpi_config
    )
    err_handler.check_program_status(config_options, mpi_config)

    # Dictionary mapping to pressure downscaling.
    downscale_pressure = {0: no_downscale, 1: pressure_down_classic}
    downscale_pressure[input_forcings.psfcDownscaleOpt](
        input_forcings, config_options, geo_meta, mpi_config
    )
    err_handler.check_program_status(config_options, mpi_config)

    # Dictionary mapping to shortwave radiation downscaling
    downscale_sw = {0: no_downscale, 1: ncar_topo_adj}
    downscale_sw[input_forcings.swDownscaleOpt](
        input_forcings, config_options, geo_meta, mpi_config
    )
    err_handler.check_program_status(config_options, mpi_config)

    # Dictionary mapping to specific humidity downscaling
    downscale_q2 = {0: no_downscale, 1: q2_down_classic}
    downscale_q2[input_forcings.q2dDownscaleOpt](
        input_forcings, config_options, geo_meta, mpi_config
    )
    err_handler.check_program_status(config_options, mpi_config)

    # Dictionary mapping to precipitation downscaling.
    downscale_precip = {
        0: no_downscale,
        1: nwm_monthly_PRISM_downscale,
        # 1: precip_mtn_mapper
    }
    downscale_precip[input_forcings.precipDownscaleOpt](
        input_forcings, config_options, geo_meta, mpi_config
    )
    err_handler.check_program_status(config_options, mpi_config)


def no_downscale(
    input_forcings: InputForcings,
    config_options: ConfigOptions,
    geo_meta: GeoMeta,
    mpi_config: MpiConfig,
):
    """Pass states through without any downscaling.

    Generic function for passing states through without any downscaling.
    :param input_forcings:
    :param config_options:
    :return:

    """
    if config_options.grid_type == "gridded":
        input_forcings.final_forcings = input_forcings.final_forcings
    elif config_options.grid_type == "unstructured":
        input_forcings.final_forcings = input_forcings.final_forcings
        input_forcings.final_forcings_elem = input_forcings.final_forcings_elem
    elif config_options.grid_type == "hydrofabric":
        input_forcings.final_forcings = input_forcings.final_forcings


def simple_lapse(
    input_forcings: InputForcings,
    config_options: ConfigOptions,
    geo_meta: GeoMeta,
    mpi_config: MpiConfig,
):
    """Apply a single lapse rate adjustment to modeled 2-meter temperature.

    Function that applies a single lapse rate adjustment to modeled
    2-meter temperature by taking the difference of the native
    input elevation and the WRF-hydro elevation.
    :param inpute_forcings:
    :param config_options:
    :param geo_meta:
    :return:
    """
    if mpi_config.rank == 0:
        config_options.statusMsg = (
            "Applying simple lapse rate to temperature downscaling"
        )
        err_handler.log_msg(config_options, mpi_config)

    # Calculate the elevation difference.
    if input_forcings.height is None:
        config_options.errMsg = (
            "Unable to perform downscaling without terrain height input"
        )
        err_handler.log_critical(config_options, mpi_config)
        return

    # Initalize missing data vars
    indNdv = None
    indNdv_elem = None

    if config_options.grid_type == "gridded":
        elevDiff = input_forcings.height - geo_meta.height
    elif config_options.grid_type == "unstructured":
        elevDiff = input_forcings.height - geo_meta.height
        elevDiff_elem = input_forcings.height_elem - geo_meta.height_elem
    elif config_options.grid_type == "hydrofabric":
        elevDiff = input_forcings.height - geo_meta.height

    # Assign existing, un-downscaled temperatures to a temporary placeholder, which
    # will be used for specific humidity downscaling.
    if input_forcings.q2dDownscaleOpt > 0:
        if config_options.grid_type == "gridded":
            input_forcings.t2dTmp[:, :] = input_forcings.final_forcings[4, :, :]
        elif config_options.grid_type == "unstructured":
            input_forcings.t2dTmp[:] = input_forcings.final_forcings[4, :]
            input_forcings.t2dTmp_elem[:] = input_forcings.final_forcings_elem[4, :]
        elif config_options.grid_type == "hydrofabric":
            input_forcings.t2dTmp[:] = input_forcings.final_forcings[4, :]

    # Apply single lapse rate value to the input 2-meter
    # temperature values.
    if config_options.grid_type == "gridded":
        try:
            indNdv = np.where(input_forcings.final_forcings == config_options.globalNdv)
        except:
            config_options.errMsg = "Unable to perform NDV search on input forcings"
            err_handler.log_critical(config_options, mpi_config)
            return
        try:
            input_forcings.final_forcings[4, :, :] = (
                input_forcings.final_forcings[4, :, :] + (6.49 / 1000.0) * elevDiff
            )
        except:
            config_options.errMsg = (
                "Unable to apply lapse rate to input 2-meter temperatures."
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        input_forcings.final_forcings[indNdv] = config_options.globalNdv

    elif config_options.grid_type == "unstructured":
        try:
            indNdv = np.where(input_forcings.final_forcings == config_options.globalNdv)
        except:
            config_options.errMsg = "Unable to perform NDV search on input forcings"
            err_handler.log_critical(config_options, mpi_config)
            return
        try:
            indNdv_elem = np.where(
                input_forcings.final_forcings_elem == config_options.globalNdv
            )
        except:
            config_options.errMsg = "Unable to perform NDV search on input forcings"
            err_handler.log_critical(config_options, mpi_config)
            return
        try:
            input_forcings.final_forcings[4, :] = (
                input_forcings.final_forcings[4, :] + (6.49 / 1000.0) * elevDiff
            )
        except:
            config_options.errMsg = (
                "Unable to apply lapse rate to input 2-meter temperatures."
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        input_forcings.final_forcings[indNdv] = config_options.globalNdv
        try:
            input_forcings.final_forcings_elem[4, :] = (
                input_forcings.final_forcings_elem[4, :]
                + (6.49 / 1000.0) * elevDiff_elem
            )
        except:
            config_options.errMsg = (
                "Unable to apply lapse rate to input 2-meter temperatures."
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        input_forcings.final_forcings_elem[indNdv_elem] = config_options.globalNdv

    elif config_options.grid_type == "hydrofabric":
        try:
            indNdv = np.where(input_forcings.final_forcings == config_options.globalNdv)
        except:
            config_options.errMsg = "Unable to perform NDV search on input forcings"
            err_handler.log_critical(config_options, mpi_config)
            return
        try:
            input_forcings.final_forcings[4, :] = (
                input_forcings.final_forcings[4, :] + (6.49 / 1000.0) * elevDiff
            )
        except:
            config_options.errMsg = (
                "Unable to apply lapse rate to input 2-meter temperatures."
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        input_forcings.final_forcings[indNdv] = config_options.globalNdv

    # Reset for memory efficiency
    indNdv = None
    indNdv_elem = None


def param_lapse(
    input_forcings: InputForcings,
    config_options: ConfigOptions,
    geo_meta: GeoMeta,
    mpi_config: MpiConfig,
):
    """Apply a single lapse rate adjustment to modeled 2-meter temperature.

    Function that applies a apriori lapse rate adjustment to modeled
    2-meter temperature by taking the difference of the native
    input elevation and the WRF-hydro elevation. It's assumed this lapse
    rate grid has already been regridded to the final output WRF-Hydro
    grid.
    :param inpute_forcings:
    :param config_options:
    :param geo_meta:
    :return:
    """
    ###################### WRF-Hydro domain only functionality ######################
    if mpi_config.rank == 0:
        config_options.statusMsg = (
            "Applying apriori lapse rate grid to temperature downscaling"
        )
        err_handler.log_msg(config_options, mpi_config)

    # Calculate the elevation difference.
    if input_forcings.height is None:
        config_options.errMsg = (
            "Unable to perform downscaling without terrain height input"
        )
        err_handler.log_critical(config_options, mpi_config)
        return
    elevDiff = input_forcings.height - geo_meta.height

    if input_forcings.lapseGrid is None:
        # if not np.any(input_forcings.lapseGrid):
        # We have not read in our lapse rate file. Read it in, do extensive checks,
        # scatter the lapse rate grid out to individual processors, then apply the
        # lapse rate to the 2-meter temperature grid.
        if mpi_config.rank == 0:
            while True:
                # First ensure we have a parameter directory
                if input_forcings.paramDir == "NONE":
                    config_options.errMsg = (
                        "User has specified spatial temperature lapse rate "
                        "downscaling while no downscaling parameter directory "
                        "exists."
                    )
                    err_handler.log_critical(config_options, mpi_config)
                    break

                # Compose the path to the lapse rate grid file.
                lapsePath = f"{input_forcings.paramDir}/lapse_param.nc"
                if not os.path.isfile(lapsePath):
                    ConfigOptions.errMsg = f"Expected lapse rate parameter file: {lapsePath} does not exist."
                    err_handler.log_critical(config_options, mpi_config)
                    break

                # Open the lapse rate file. Check for the expected variable, along with
                # the dimension size to make sure everything matches up.
                try:
                    idTmp = Dataset(lapsePath, "r")
                except:
                    ConfigOptions.errMsg = f"Unable to open parameter file: {lapsePath}"
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break
                if not "lapse" in idTmp.variables.keys():
                    ConfigOptions.errMsg = f"Expected 'lapse' variable not located in parameter file: {lapsePath}"
                    err_handler.log_critical(config_options, mpi_config)
                    break
                try:
                    lapseTmp = idTmp.variables["lapse"][:, :]
                except:
                    ConfigOptions.errMsg = f"Unable to extracte 'lapse' variable from parameter: file: {lapsePath}"
                    err_handler.log_critical(config_options, mpi_config)
                    break

                # Check dimensions to ensure they match up to the output grid.
                if lapseTmp.shape[1] != geo_meta.nx_global:
                    ConfigOptions.errMsg = f"X-Dimension size mismatch between output grid and lapse rate from parameter file: {lapsePath}"
                    err_handler.log_critical(config_options, mpi_config)
                    break
                if lapseTmp.shape[0] != geo_meta.ny_global:
                    ConfigOptions.errMsg = f"Y-Dimension size mismatch between output grid and lapse rate from parameter file: {lapsePath}"
                    err_handler.log_critical(config_options, mpi_config)
                    break

                # Perform a quick search to ensure we don't have radical values.
                indTmp = np.where(lapseTmp < -10.0)
                if len(indTmp[0]) > 0:
                    ConfigOptions.errMsg = f"Found anomolous negative values in the lapse rate grid from parameter file: {lapsePath}"
                    err_handler.log_critical(config_options, mpi_config)
                    break
                indTmp = np.where(lapseTmp > 100.0)
                if len(indTmp[0]) > 0:
                    ConfigOptions.errMsg = f"Found excessively high values in the lapse rate grid from parameter file: {lapsePath}"
                    err_handler.log_critical(config_options, mpi_config)
                    break

                # Close the parameter lapse rate file.
                try:
                    idTmp.close()
                except:
                    ConfigOptions.errMsg = (
                        f"Unable to close parameter file: {lapsePath}"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                    break

                break
        else:
            lapseTmp = None
        err_handler.check_program_status(config_options, mpi_config)

        # Scatter the lapse rate grid to the other processors.
        input_forcings.lapseGrid = mpi_config.scatter_array(
            geo_meta, lapseTmp, config_options
        )
        err_handler.check_program_status(config_options, mpi_config)

    # Apply the local lapse rate grid to our local slab of 2-meter temperature data.
    temperature_grid_tmp = input_forcings.final_forcings[4, :, :]
    try:
        indNdv = np.where(input_forcings.final_forcings == config_options.globalNdv)
    except:
        ConfigOptions.errMsg = f"Unable to perform NDV search on input {input_forcings.product_name} regridded forcings."
        err_handler.log_critical(config_options, mpi_config)
        return
    try:
        indValid = np.where(temperature_grid_tmp != config_options.globalNdv)
    except:
        ConfigOptions.errMsg = f"Unable to perform search for valid values on input {input_forcings.product_name} regridded temperature forcings."
        err_handler.log_critical(config_options, mpi_config)
        return
    try:
        temperature_grid_tmp[indValid] = temperature_grid_tmp[indValid] + (
            (input_forcings.lapseGrid[indValid] / 1000.0) * elevDiff[indValid]
        )
    except:
        ConfigOptions.errMsg = f"Unable to apply spatial lapse rate values to input {input_forcings.product_name} regridded temperature forcings."
        err_handler.log_critical(config_options, mpi_config)
        return

    input_forcings.final_forcings[4, :, :] = temperature_grid_tmp
    input_forcings.final_forcings[indNdv] = config_options.globalNdv

    # Reset for memory efficiency
    indTmp = None
    indNdv = None
    indValid = None
    elevDiff = None
    temperature_grid_tmp = None


def pressure_down_classic(
    input_forcings: InputForcings,
    config_options: ConfigOptions,
    geo_meta: GeoMeta,
    mpi_config: MpiConfig,
):
    """Apply a single lapse rate adjustment to modeled surface pressure.

    Generic function to downscale surface pressure to the WRF-Hydro domain.
    :param input_forcings:
    :param config_options:
    :param geo_meta:
    :return:
    """
    if mpi_config.rank == 0:
        config_options.statusMsg = (
            "Performing topographic adjustment to surface pressure."
        )
        err_handler.log_msg(config_options, mpi_config)

    # Calculate the elevation difference.
    if input_forcings.height is None:
        config_options.errMsg = (
            "Unable to perform downscaling without terrain height input"
        )
        err_handler.log_critical(config_options, mpi_config)
        return

    # Initalize missing data vars
    indNdv = None
    indNdv_elem = None

    if config_options.grid_type == "gridded":
        elevDiff = input_forcings.height - geo_meta.height
    elif config_options.grid_type == "unstructured":
        elevDiff = input_forcings.height - geo_meta.height
        elevDiff_elem = input_forcings.height_elem - geo_meta.height_elem
    elif config_options.grid_type == "hydrofabric":
        elevDiff = input_forcings.height - geo_meta.height

    # Assign existing, un-downscaled pressure values to a temporary placeholder, which
    # will be used for specific humidity downscaling.
    if input_forcings.q2dDownscaleOpt > 0:
        if config_options.grid_type == "gridded":
            input_forcings.psfcTmp[:, :] = input_forcings.final_forcings[6, :, :]
        elif config_options.grid_type == "unstructured":
            input_forcings.psfcTmp[:] = input_forcings.final_forcings[6, :]
            input_forcings.psfcTmp_elem[:] = input_forcings.final_forcings_elem[6, :]
        elif config_options.grid_type == "hydrofabric":
            input_forcings.psfcTmp[:] = input_forcings.final_forcings[6, :]

    if config_options.grid_type == "gridded":
        try:
            indNdv = np.where(input_forcings.final_forcings == config_options.globalNdv)
        except:
            config_options.errMsg = "Unable to perform NDV search on input forcings"
            err_handler.log_critical(config_options, mpi_config)
            return
        try:
            input_forcings.final_forcings[6, :, :] = input_forcings.final_forcings[
                6, :, :
            ] + (input_forcings.final_forcings[6, :, :] * elevDiff * 9.8) / (
                input_forcings.final_forcings[4, :, :] * 287.05
            )
        except:
            config_options.errMsg = (
                "Unable to downscale surface pressure to input forcings."
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        input_forcings.final_forcings[indNdv] = config_options.globalNdv

    elif config_options.grid_type == "unstructured":
        try:
            indNdv = np.where(input_forcings.final_forcings == config_options.globalNdv)
        except:
            config_options.errMsg = "Unable to perform NDV search on input forcings"
            err_handler.log_critical(config_options, mpi_config)
            return
        try:
            indNdv_elem = np.where(
                input_forcings.final_forcings_elem == config_options.globalNdv
            )
        except:
            config_options.errMsg = "Unable to perform NDV search on input forcings"
            err_handler.log_critical(config_options, mpi_config)
            return
        try:
            input_forcings.final_forcings[6, :] = input_forcings.final_forcings[
                6, :
            ] + (input_forcings.final_forcings[6, :] * elevDiff * 9.8) / (
                input_forcings.final_forcings[4, :] * 287.05
            )
        except:
            config_options.errMsg = (
                "Unable to downscale surface pressure to input forcings."
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        input_forcings.final_forcings[indNdv] = config_options.globalNdv

        try:
            input_forcings.final_forcings_elem[6, :] = (
                input_forcings.final_forcings_elem[6, :]
                + (input_forcings.final_forcings_elem[6, :] * elevDiff_elem * 9.8)
                / (input_forcings.final_forcings_elem[4, :] * 287.05)
            )
        except:
            config_options.errMsg = (
                "Unable to downscale surface pressure to input forcings."
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        input_forcings.final_forcings_elem[indNdv_elem] = config_options.globalNdv
    elif config_options.grid_type == "hydrofabric":
        try:
            indNdv = np.where(input_forcings.final_forcings == config_options.globalNdv)
        except:
            config_options.errMsg = "Unable to perform NDV search on input forcings"
            err_handler.log_critical(config_options, mpi_config)
            return
        try:
            input_forcings.final_forcings[6, :] = input_forcings.final_forcings[
                6, :
            ] + (input_forcings.final_forcings[6, :] * elevDiff * 9.8) / (
                input_forcings.final_forcings[4, :] * 287.05
            )
        except:
            config_options.errMsg = (
                "Unable to downscale surface pressure to input forcings."
            )
            err_handler.log_critical(config_options, mpi_config)
            return
        input_forcings.final_forcings[indNdv] = config_options.globalNdv

    # Reset for memory efficiency
    indNdv = None
    indNdv_elem = None


def q2_down_classic(
    input_forcings: InputForcings,
    config_options: ConfigOptions,
    geo_meta: GeoMeta,
    mpi_config: MpiConfig,
):
    """Apply a single lapse rate adjustment to modeled 2-meter specific humidity.

    NCAR function for downscaling 2-meter specific humidity using already downscaled
    2-meter temperature, unadjusted surface pressure, and downscaled surface
    pressure.
    :param input_forcings:
    :param config_options:
    :param geo_meta:
    :return:
    """
    if mpi_config.rank == 0:
        config_options.statusMsg = (
            "Performing topographic adjustment to specific humidity."
        )
        err_handler.log_msg(config_options, mpi_config)

    if config_options.grid_type != "unstructured":
        # Establish where we have missing values.
        try:
            indNdv = np.where(input_forcings.final_forcings == config_options.globalNdv)
        except:
            config_options.errMsg = "Unable to perform NDV search on input forcings"
            err_handler.log_critical(config_options, mpi_config)
            return

        # First calculate relative humidity given original surface pressure and 2-meter
        # temperature
        try:
            relHum = rel_hum(input_forcings, config_options)
        except:
            config_options.errMsg = (
                "Unable to perform topographic downscaling of incoming "
                "specific humidity to relative humidity"
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        # Downscale 2-meter specific humidity
        try:
            q2Tmp = mixhum_ptrh(input_forcings, relHum, 2, config_options)
        except:
            config_options.errMsg = (
                "Unable to perform topographic downscaling of "
                "incoming specific humidity"
            )
            err_handler.log_critical(config_options, mpi_config)
            return
        if config_options.grid_type == "gridded":
            input_forcings.final_forcings[5, :, :] = q2Tmp
        else:
            input_forcings.final_forcings[5, :] = q2Tmp

        input_forcings.final_forcings[indNdv] = config_options.globalNdv
        q2Tmp = None

    elif config_options.grid_type == "unstructured":
        # Establish where we have missing values.
        try:
            indNdv = np.where(input_forcings.final_forcings == config_options.globalNdv)
        except:
            config_options.errMsg = "Unable to perform NDV search on input forcings"
            err_handler.log_critical(config_options, mpi_config)
            return
        # Establish where we have missing values.
        try:
            indNdv_elem = np.where(
                input_forcings.final_forcings_elem == config_options.globalNdv
            )
        except:
            config_options.errMsg = "Unable to perform NDV search on input forcings"
            err_handler.log_critical(config_options, mpi_config)
            return
        # First calculate relative humidity given original surface pressure and 2-meter
        # temperature
        try:
            relHum, relHum_elem = rel_hum(input_forcings, config_options)
        except:
            config_options.errMsg = (
                "Unable to perform topographic downscaling of incoming "
                "specific humidity to relative humidity"
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        # Downscale 2-meter specific humidity
        try:
            q2Tmp, q2Tmp_elem = mixhum_ptrh_unstructured(
                input_forcings, relHum, relHum_elem, 2, config_options
            )
        except:
            config_options.errMsg = (
                "Unable to perform topographic downscaling of "
                "incoming specific humidity"
            )
            err_handler.log_critical(config_options, mpi_config)
            return
        input_forcings.final_forcings[5, :] = q2Tmp
        input_forcings.final_forcings_elem[5, :] = q2Tmp_elem
        input_forcings.final_forcings[indNdv] = config_options.globalNdv
        input_forcings.final_forcings_elem[indNdv_elem] = config_options.globalNdv
        q2Tmp = None
        indNdv = None
        q2Tmp_elem = None
        indNdv_elem = None


def nwm_monthly_PRISM_downscale(
    input_forcings: InputForcings,
    config_options: ConfigOptions,
    geo_meta: GeoMeta,
    mpi_config: MpiConfig,
):
    """Apply a single lapse rate adjustment to modeled precipitation.

    NCAR/OWP function for downscaling precipitation using monthly PRISM climatology in a
    mountain-mapper like fashion.
    :param input_forcings:
    :param config_options:
    :param geo_meta:
    :return:
    """
    ############################### WRF-Hydro domain only method ################################

    if mpi_config.rank == 0:
        config_options.statusMsg = (
            "Performing NWM Monthly PRISM Mountain Mapper Downscaling of Precipitation"
        )
        err_handler.log_msg(config_options, mpi_config)

    # Establish whether or not we need to read in new PRISM monthly climatology:
    # 1.) This is the first output timestep, and no grids have been initialized.
    # 2.) We have switched months from the last timestep. In this case, we need
    #     to re-initialize the grids for the current month.
    initialize_flag = False

    mmVersion = 2
    if input_forcings.keyValue == 3:
        keyValueStr = "GFS"
    if mmVersion == None:
        config_options.errMsg = "Invalid Mountain Mapper Precip Downscaling option\n"
        err_handler.log_critical(config_options, mpi_config)

    if (
        input_forcings.nwmPRISM_denGrid is None
        and input_forcings.nwmPRISM_numGrid is None
    ):
        # We are on situation 1 - This is the first output step.
        initialize_flag = True
        # LOG.debug('WE NEED TO READ IN PRISM GRIDS')
    if (
        config_options.current_output_date.month
        != config_options.prev_output_date.month
    ):
        # We are on situation #2 - The month has changed so we need to reinitialize the
        # PRISM grids.
        initialize_flag = True
        # LOG.debug('MONTH CHANGE.... NEED TO READ IN NEW PRISM GRIDS.')
    if initialize_flag is True:
        while True:
            # First reset the local PRISM grids to be safe.
            input_forcings.nwmPRISM_numGrid = None
            input_forcings.nwmPRISM_denGrid = None

            if mmVersion == 1:
                # Compose paths to the expected files.
                numeratorPath = f"{input_forcings.paramDir}/PRISM_Precip_Clim_{ConfigOptions.current_output_date.strftime('%b')}_NWM_Grid.nc"
                denominatorPath = f"{input_forcings.paramDir}/PRISM_Precip_Clim_{ConfigOptions.current_output_date.strftime('%b')}_NWM_to_{keyValueStr!s}_Grid.nc"

            elif mmVersion == 2:
                # Compose paths to the expected files.
                numeratorPath = f"{input_forcings.paramDir}/PRISM_Precip_Clim_{ConfigOptions.current_output_date.strftime('%b')}_NWM_Grid.nc"
                denominatorPath = f"{input_forcings.paramDir}/PRISM_Precip_Clim_{ConfigOptions.current_output_date.strftime('%b')}_{keyValueStr!s}_to_NWM_Grid.nc"

            # Make sure files exist.
            if not os.path.isfile(numeratorPath):
                ConfigOptions.errMsg = f"Expected parameter file: {numeratorPath} for mountain mapper downscaling of precipitation not found."
                err_handler.log_critical(config_options, mpi_config)
                break

            if not os.path.isfile(denominatorPath):
                ConfigOptions.errMsg = f"Expected parameter file: {denominatorPath} for mountain mapper downscaling of precipitation not found."
                err_handler.log_critical(config_options, mpi_config)
                break

            if mpi_config.rank == 0:
                # Open the NetCDF parameter files. Check to make sure expected dimension
                # sizes are in place, along with variable names, etc.
                try:
                    idNum = Dataset(numeratorPath, "r")
                except:
                    ConfigOptions.errMsg = (
                        f"Unable to open parameter file: {numeratorPath}"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                    break
                try:
                    idDenom = Dataset(denominatorPath, "r")
                except:
                    ConfigOptions.errMsg = (
                        f"Unable to open parameter file: {denominatorPath}"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                    break

                # Check to make sure expected names, dimension sizes are present.
                if "x" not in idNum.variables.keys():
                    ConfigOptions.errMsg = f"Expected 'x' variable not found in parameter file: {numeratorPath}"
                    err_handler.log_critical(config_options, mpi_config)
                    break
                if "x" not in idDenom.variables.keys():
                    ConfigOptions.errMsg = f"Expected 'x' variable not found in parameter file: {denominatorPath}"
                    err_handler.log_critical(config_options, mpi_config)
                    break

                if "y" not in idNum.variables.keys():
                    ConfigOptions.errMsg = f"Expected 'y' variable not found in parameter file: {numeratorPath}"
                    err_handler.log_critical(config_options, mpi_config)
                    break
                if "y" not in idDenom.variables.keys():
                    ConfigOptions.errMsg = f"Expected 'y' variable not found in parameter file: {denominatorPath}"
                    err_handler.log_critical(config_options, mpi_config)
                    break

                if "Data" not in idNum.variables.keys():
                    ConfigOptions.errMsg = f"Expected 'Data' variable not found in parameter file: {numeratorPath}"
                    err_handler.log_critical(config_options, mpi_config)
                    break
                if "Data" not in idDenom.variables.keys():
                    ConfigOptions.errMsg = f"Expected 'Data' variable not found in parameter file: {denominatorPath}"
                    err_handler.log_critical(config_options, mpi_config)
                    break

                if idNum.variables["Data"].shape[0] != geo_meta.ny_global:
                    ConfigOptions.errMsg = f"Input Y dimension for: {numeratorPath} does not match the output WRF-Hydro Y dimension size."
                    err_handler.log_critical(config_options, mpi_config)
                    break
                if idDenom.variables["Data"].shape[0] != geo_meta.ny_global:
                    ConfigOptions.errMsg = f"Input Y dimension for: {denominatorPath} does not match the output WRF-Hydro Y dimension size."
                    err_handler.log_critical(config_options, mpi_config)
                    break

                if idNum.variables["Data"].shape[1] != geo_meta.nx_global:
                    ConfigOptions.errMsg = f"Input X dimension for: {numeratorPath} does not match the output WRF-Hydro X dimension size."
                    err_handler.log_critical(config_options, mpi_config)
                    break
                if idDenom.variables["Data"].shape[1] != geo_meta.nx_global:
                    ConfigOptions.errMsg = f"Input X dimension for: {denominatorPath} does not match the output WRF-Hydro X dimension size."
                    err_handler.log_critical(config_options, mpi_config)
                    break

                # Read in the PRISM grid on the output grid. Then scatter the array out to the processors.
                try:
                    numDataTmp = idNum.variables["Data"][:, :]
                except:
                    ConfigOptions.errMsg = (
                        f"Unable to extract 'Data' from parameter file: {numeratorPath}"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                    break
                try:
                    denDataTmp = idDenom.variables["Data"][:, :]
                except:
                    ConfigOptions.errMsg = f"Unable to extract 'Data' from parameter file: {denominatorPath}"
                    err_handler.log_critical(config_options, mpi_config)
                    break

                # Close the parameter files.
                try:
                    idNum.close()
                except:
                    ConfigOptions.errMsg = (
                        f"Unable to close parameter file: {numeratorPath}"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                    break
                try:
                    idDenom.close()
                except:
                    ConfigOptions.errMsg = (
                        f"Unable to close parameter file: {denominatorPath}"
                    )
                    err_handler.log_critical(config_options, mpi_config)
                    break
            else:
                numDataTmp = None
                denDataTmp = None

            break
        err_handler.check_program_status(config_options, mpi_config)

        # Scatter the array out to the local processors
        input_forcings.nwmPRISM_numGrid = mpi_config.scatter_array(
            geo_meta, numDataTmp, config_options
        )
        err_handler.check_program_status(config_options, mpi_config)

        input_forcings.nwmPRISM_denGrid = mpi_config.scatter_array(
            geo_meta, denDataTmp, config_options
        )
        err_handler.check_program_status(config_options, mpi_config)

    # Create temporary grids from the local slabs of params/precip forcings.
    hourlyGrid = input_forcings.final_forcings[3, :, :]
    tmpGrid = np.full([geo_meta.ny_local, geo_meta.nx_local], -9999.0, dtype=float)
    ratioRainGrid = np.full(
        [geo_meta.ny_local, geo_meta.nx_local], -9999.0, dtype=float
    )

    localRainRate = input_forcings.final_forcings[3, :, :]
    numLocal = input_forcings.nwmPRISM_numGrid
    denLocal = input_forcings.nwmPRISM_denGrid

    # Establish index of where we have valid data.
    try:
        indValid = np.where(
            (localRainRate != -9999.0) & (denLocal != -9999.0) & (denLocal > 1.0)
        )
    except:
        config_options.errMsg = (
            "Unable to run numpy search for valid values on precip and "
            "param grid in mountain mapper downscaling"
        )
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    try:
        tmpGrid[indValid] = localRainRate[indValid] / denLocal[indValid]
    except:
        config_options.errMsg = (
            "Unable to divide precip by denominator in mountain mapper downscaling"
        )
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Establish index of where we have valid data.
    try:
        indValid = np.where((tmpGrid != -9999.0) & (numLocal != -9999.0))
    except:
        config_options.errMsg = (
            "Unable to run numpy search for valid values on precip and "
            "param grid in mountain mapper downscaling"
        )
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)
    try:
        ratioRainGrid[indValid] = tmpGrid[indValid] * numLocal[indValid]
    except:
        config_options.errMsg = (
            "Unable to multiply precip by numerator in mountain mapper downscaling"
        )
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    count = 0

    try:
        indValid = np.where(
            (ratioRainGrid == -9999.0) & (numLocal != -9999.0) & (hourlyGrid != -9999.0)
        )
    except:
        config_options.errMsg = (
            "Unable to run numpy search for valid values on precip and "
            "param grid in mountain mapper downscaling"
        )
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    count = len(indValid[0])
    if count > 0:
        ratioRainGrid[indValid] = hourlyGrid[indValid] / 3600

    try:
        indValid = np.where(ratioRainGrid != -9999.0)

    except:
        config_options.errMsg = (
            "Unable to run numpy search for valid values on precip and "
            "param grid in mountain mapper downscaling"
        )
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    ## Convert local precip back to a rate (mm/s)
    try:
        ratioRainGrid[indValid] = ratioRainGrid[indValid] / 3600

    except:
        config_options.errMsg = (
            "Unable to convert temporary precip rate from mm to mm/s."
        )
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)
    input_forcings.final_forcings[3, :, :] = ratioRainGrid

    # Reset variables for memory efficiency
    idDenom = None
    idNum = None
    localRainRate = None
    numLocal = None
    denLocal = None


def ncar_topo_adj(
    input_forcings: InputForcings,
    config_options: ConfigOptions,
    geo_meta: GeoMeta,
    mpi_config: MpiConfig,
):
    """Topographic adjustment of incoming shortwave radiation fluxes, given input parameters.

    :param input_forcings:
    :param config_options:
    :return:

    """
    if mpi_config.rank == 0:
        config_options.statusMsg = (
            "Performing topographic adjustment to incoming shortwave radiation flux."
        )
        err_handler.log_msg(config_options, mpi_config)

    if config_options.grid_type == "gridded":
        # Establish where we have missing values.
        try:
            indNdv = np.where(input_forcings.final_forcings == config_options.globalNdv)
        except:
            config_options.errMsg = "Unable to perform NDV search on input forcings"
            err_handler.log_critical(config_options, mpi_config)
            return

        # By the time this function has been called, necessary input static grids (height, slope, etc),
        # should have been calculated for each local slab of data.
        DEGRAD = math.pi / 180.0
        DPD = 360.0 / 365.0
        try:
            DECLIN, SOLCON = radconst(config_options)
        except:
            config_options.errMsg = (
                "Unable to calculate solar constants based on datetime information."
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        try:
            coszen_loc, hrang_loc = calc_coszen(config_options, DECLIN, geo_meta)
        except:
            config_options.errMsg = (
                "Unable to calculate COSZEN or HRANG variables for topographic adjustment "
                "of incoming shortwave radiation"
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        try:
            TOPO_RAD_ADJ_DRVR(
                geo_meta,
                config_options,
                input_forcings,
                coszen_loc,
                DECLIN,
                SOLCON,
                hrang_loc,
            )
        except:
            config_options.errMsg = (
                "Unable to perform final topographic adjustment of incoming "
                "shortwave radiation fluxes."
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        # Assign missing values based on our mask.
        input_forcings.final_forcings[indNdv] = config_options.globalNdv

        # Reset variables to free up memory
        DECLIN = None
        SOLCON = None
        coszen_loc = None
        hrang_loc = None
        indNdv = None

    elif config_options.grid_type == "unstructured":
        # Establish where we have missing values.
        try:
            indNdv = np.where(input_forcings.final_forcings == config_options.globalNdv)
            indNdv_elem = np.where(
                input_forcings.final_forcings_elem == config_options.globalNdv
            )
        except:
            config_options.errMsg = "Unable to perform NDV search on input forcings"
            err_handler.log_critical(config_options, mpi_config)
            return

        # By the time this function has been called, necessary input static grids (height, slope, etc),
        # should have been calculated for each local slab of data.
        DEGRAD = math.pi / 180.0
        DPD = 360.0 / 365.0
        try:
            DECLIN, SOLCON = radconst(config_options)
        except:
            config_options.errMsg = (
                "Unable to calculate solar constants based on datetime information."
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        try:
            coszen_loc, coszen_loc_elem, hrang_loc, hrang_loc_elem = (
                calc_coszen_unstructured(config_options, DECLIN, geo_meta)
            )
        except:
            config_options.errMsg = (
                "Unable to calculate COSZEN or HRANG variables for topographic adjustment "
                "of incoming shortwave radiation"
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        try:
            TOPO_RAD_ADJ_DRVR_unstructured(
                geo_meta,
                input_forcings,
                coszen_loc,
                coszen_loc_elem,
                DECLIN,
                SOLCON,
                hrang_loc,
                hrang_loc_elem,
            )
        except:
            config_options.errMsg = (
                "Unable to perform final topographic adjustment of incoming "
                "shortwave radiation fluxes."
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        # Assign missing values based on our mask.
        input_forcings.final_forcings[indNdv] = config_options.globalNdv
        input_forcings.final_forcings_elem[indNdv_elem] = config_options.globalNdv

        # Reset variables to free up memory
        DECLIN = None
        SOLCON = None
        coszen_loc = None
        hrang_loc = None
        indNdv = None
        coszen_loc_elem = None
        hrang_loc_elem = None
        indNdv_elem = None
    elif config_options.grid_type == "hydrofabric":
        # Establish where we have missing values.
        try:
            indNdv = np.where(input_forcings.final_forcings == config_options.globalNdv)
        except:
            config_options.errMsg = "Unable to perform NDV search on input forcings"
            err_handler.log_critical(config_options, mpi_config)
            return

        # By the time this function has been called, necessary input static grids (height, slope, etc),
        # should have been calculated for each local slab of data.
        DEGRAD = math.pi / 180.0
        DPD = 360.0 / 365.0
        try:
            DECLIN, SOLCON = radconst(config_options)
        except:
            config_options.errMsg = (
                "Unable to calculate solar constants based on datetime information."
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        try:
            coszen_loc, hrang_loc = calc_coszen(config_options, DECLIN, geo_meta)
        except:
            config_options.errMsg = (
                "Unable to calculate COSZEN or HRANG variables for topographic adjustment "
                "of incoming shortwave radiation"
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        try:
            TOPO_RAD_ADJ_DRVR(
                geo_meta,
                config_options,
                input_forcings,
                coszen_loc,
                DECLIN,
                SOLCON,
                hrang_loc,
            )
        except:
            config_options.errMsg = (
                "Unable to perform final topographic adjustment of incoming "
                "shortwave radiation fluxes."
            )
            err_handler.log_critical(config_options, mpi_config)
            return

        # Assign missing values based on our mask.
        input_forcings.final_forcings[indNdv] = config_options.globalNdv

        # Reset variables to free up memory
        DECLIN = None
        SOLCON = None
        coszen_loc = None
        hrang_loc = None
        indNdv = None


def radconst(config_options: ConfigOptions):
    """Calculate the current incoming solar constant.

    Function to calculate the current incoming solar constant.
    :param config_options:
    :return:
    """
    dCurrent = config_options.current_output_date
    DEGRAD = math.pi / 180.0
    DPD = 360.0 / 365.0

    # For short wave radiation
    DECLIN = 0.0
    SOLCON = 0.0

    # Calculate the current julian day.
    JULIAN = time.strptime(dCurrent.strftime("%Y.%m.%d"), "%Y.%m.%d").tm_yday

    # OBECL : OBLIQUITY = 23.5 DEGREE
    OBECL = 23.5 * DEGRAD
    SINOB = math.sin(OBECL)

    # Calculate longitude of the sun from vernal equinox
    if JULIAN >= 80:
        SXLONG = DPD * (JULIAN - 80)
    if JULIAN < 80:
        SXLONG = DPD * (JULIAN + 285)
    SXLONG = SXLONG * DEGRAD
    ARG = SINOB * math.sin(SXLONG)
    DECLIN = math.asin(ARG)
    DECDEG = DECLIN / DEGRAD

    # Solar constant eccentricity factor (Paltridge and Platt 1976)
    DJUL = JULIAN * 360.0 / 365.0
    RJUL = DJUL * DEGRAD
    ECCFAC = (
        1.000110
        + (0.034221 * math.cos(RJUL))
        + (0.001280 * math.sin(RJUL))
        + (0.000719 * math.cos(2 * RJUL))
        + (0.000077 * math.sin(2 * RJUL))
    )
    SOLCON = 1370.0 * ECCFAC

    return DECLIN, SOLCON


def calc_coszen(config_options: ConfigOptions, declin, geo_meta: GeoMeta):
    """Calculate the cosine of the solar zenith angle and the hour angle.

    Downscaling function to compute radiation terms based on current datetime
    information and lat/lon grids.
    :param config_options:
    :param input_forcings:
    :param declin:
    :return:
    """
    degrad = math.pi / 180.0
    gmt = 0

    # Calculate the current julian day.
    dCurrent = config_options.current_output_date
    julian = time.strptime(dCurrent.strftime("%Y.%m.%d"), "%Y.%m.%d").tm_yday

    da = 6.2831853071795862 * ((julian - 1) / 365.0)
    eot = (
        (0.000075 + 0.001868 * math.cos(da))
        - (0.032077 * math.sin(da))
        - (0.014615 * math.cos(2 * da))
        - (0.04089 * math.sin(2 * da))
    ) * 229.18
    xtime = dCurrent.hour * 60.0  # Minutes of day
    xt24 = int(xtime) % 1440 + eot
    tloctm = geo_meta.longitude_grid / 15.0 + gmt + xt24 / 60.0
    hrang = ((tloctm - 12.0) * degrad) * 15.0
    xxlat = geo_meta.latitude_grid * degrad
    coszen = np.sin(xxlat) * math.sin(declin) + np.cos(xxlat) * math.cos(
        declin
    ) * np.cos(hrang)

    # Reset temporary variables to free up memory.
    tloctm = None
    xxlat = None

    return coszen, hrang


def calc_coszen_unstructured(config_options: ConfigOptions, declin, geo_meta: GeoMeta):
    """Calculate the cosine of the solar zenith angle and the hour angle for unstructured grids.

    Downscaling function to compute radiation terms based on current datetime
    information and lat/lon grids.
    :param config_options:
    :param input_forcings:
    :param declin:
    :return:
    """
    degrad = math.pi / 180.0
    gmt = 0

    # Calculate the current julian day.
    dCurrent = config_options.current_output_date
    julian = time.strptime(dCurrent.strftime("%Y.%m.%d"), "%Y.%m.%d").tm_yday

    da = 6.2831853071795862 * ((julian - 1) / 365.0)
    eot = (
        (0.000075 + 0.001868 * math.cos(da))
        - (0.032077 * math.sin(da))
        - (0.014615 * math.cos(2 * da))
        - (0.04089 * math.sin(2 * da))
    ) * 229.18
    xtime = dCurrent.hour * 60.0  # Minutes of day
    xt24 = int(xtime) % 1440 + eot
    tloctm = geo_meta.longitude_grid / 15.0 + gmt + xt24 / 60.0
    tloctm_elem = geo_meta.longitude_grid_elem / 15.0 + gmt + xt24 / 60.0
    hrang = ((tloctm - 12.0) * degrad) * 15.0
    hrang_elem = ((tloctm_elem - 12.0) * degrad) * 15.0
    xxlat = geo_meta.latitude_grid * degrad
    xxlat_elem = geo_meta.latitude_grid_elem * degrad
    coszen = np.sin(xxlat) * math.sin(declin) + np.cos(xxlat) * math.cos(
        declin
    ) * np.cos(hrang)
    coszen_elem = np.sin(xxlat_elem) * math.sin(declin) + np.cos(xxlat_elem) * math.cos(
        declin
    ) * np.cos(hrang_elem)
    # Reset temporary variables to free up memory.
    tloctm = None
    xxlat = None
    tloctm_elem = None
    xxlat_elem = None

    return coszen, coszen_elem, hrang, hrang_elem


def TOPO_RAD_ADJ_DRVR(
    geo_meta: GeoMeta,
    config_options: ConfigOptions,
    input_forcings: InputForcings,
    COSZEN,
    declin,
    solcon,
    hrang2d,
):
    """Topographic adjustment of incoming shortwave radiation fluxes, given input parameters.

    Downscaling driver for correcting incoming shortwave radiation fluxes from a low
    resolution to a a higher resolution.
    :param geo_meta:
    :param input_forcings:
    :param COSZEN:
    :param declin:
    :param solcon:
    :param hrang2d:
    :return:
    """
    degrad = math.pi / 180.0

    ny = geo_meta.ny_local
    nx = geo_meta.nx_local

    xxlat = geo_meta.latitude_grid * degrad

    # Sanity checking on incoming shortwave grid.
    if config_options.grid_type == "gridded":
        SWDOWN = input_forcings.final_forcings[7, :, :]
    else:
        SWDOWN = input_forcings.final_forcings[7, :]
    SWDOWN[np.where(SWDOWN < 0.0)] = 0.0
    SWDOWN[np.where(SWDOWN >= 1400.0)] = 1400.0

    COSZEN[np.where(COSZEN < 1e-4)] = 1e-4

    if config_options.grid_type == "gridded":
        corr_frac = np.empty([ny, nx], int)
        # shadow_mask = np.empty([ny,nx],int)
        diffuse_frac = np.empty([ny, nx], int)
        corr_frac[:, :] = 0
        diffuse_frac[:, :] = 0
        # shadow_mask[:,:] = 0
        indTmp = np.where((geo_meta.slope[:, :] == 0.0) & (SWDOWN <= 10.0))
    else:
        corr_frac = np.empty([ny], int)
        # shadow_mask = np.empty([ny],int)
        diffuse_frac = np.empty([ny], int)
        corr_frac[:] = 0
        diffuse_frac[:] = 0
        # shadow_mask[:] = 0
        indTmp = np.where((geo_meta.slope[:] == 0.0) & (SWDOWN <= 10.0))

    corr_frac[indTmp] = 1

    term1 = np.sin(xxlat) * np.cos(hrang2d)
    term2 = (0 - np.cos(geo_meta.slp_azi)) * np.sin(geo_meta.slope)
    term3 = np.sin(hrang2d) * (np.sin(geo_meta.slp_azi) * np.sin(geo_meta.slope))
    term4 = (np.cos(xxlat) * np.cos(hrang2d)) * np.cos(geo_meta.slope)
    term5 = np.cos(xxlat) * (np.cos(geo_meta.slp_azi) * np.sin(geo_meta.slope))
    term6 = np.sin(xxlat) * np.cos(geo_meta.slope)

    csza_slp = (term1 * term2 - term3 + term4) * math.cos(declin) + (
        term5 + term6
    ) * math.sin(declin)

    csza_slp[np.where(csza_slp <= 1e-4)] = 1e-4
    # Topographic shading
    # csza_slp[np.where(shadow == 1)] = 1E-4

    # Correction factor for sloping topographic: the diffuse fraction of solar
    # radiation is assumed to be unaffected by the slope.
    corr_fac = diffuse_frac + ((1 - diffuse_frac) * csza_slp) / COSZEN
    corr_fac[np.where(corr_fac > 1.3)] = 1.3

    # Peform downscaling
    SWDOWN_OUT = SWDOWN * corr_fac

    # Reset variables to free up memory
    # corr_frac = None
    diffuse_frac = None
    term1 = None
    term2 = None
    term3 = None
    term4 = None
    term5 = None
    term6 = None

    if config_options.grid_type == "gridded":
        input_forcings.final_forcings[7, :, :] = SWDOWN_OUT
    else:
        input_forcings.final_forcings[7, :] = SWDOWN_OUT

    # Reset variables to free up memory
    SWDOWN = None
    SWDOWN_OUT = None


def TOPO_RAD_ADJ_DRVR_unstructured(
    geo_meta: GeoMeta,
    input_forcings: InputForcings,
    COSZEN,
    COSZEN_elem,
    declin,
    solcon,
    hrang2d,
    hrang2d_elem,
):
    """Topographic adjustment of incoming shortwave radiation fluxes, given input parameters.

    Downscaling driver for correcting incoming shortwave radiation fluxes from a low
    resolution to a a higher resolution.
    :param geo_meta:
    :param input_forcings:
    :param COSZEN:
    :param declin:
    :param solcon:
    :param hrang2d:
    :return:
    """
    degrad = math.pi / 180.0

    ny = geo_meta.ny_local
    nx = geo_meta.nx_local

    ny_elem = geo_meta.ny_local_elem
    nx_elem = geo_meta.nx_local_elem

    xxlat = geo_meta.latitude_grid * degrad
    xxlat_elem = geo_meta.latitude_grid_elem * degrad

    # Sanity checking on incoming shortwave grid.
    SWDOWN = input_forcings.final_forcings[7, :]
    SWDOWN_elem = input_forcings.final_forcings_elem[7, :]
    SWDOWN[np.where(SWDOWN < 0.0)] = 0.0
    SWDOWN_elem[np.where(SWDOWN_elem < 0.0)] = 0.0
    SWDOWN[np.where(SWDOWN >= 1400.0)] = 1400.0
    SWDOWN_elem[np.where(SWDOWN_elem >= 1400.0)] = 1400.0
    COSZEN[np.where(COSZEN < 1e-4)] = 1e-4
    COSZEN_elem[np.where(COSZEN_elem < 1e-4)] = 1e-4

    corr_frac = np.empty([ny], int)
    corr_frac_elem = np.empty([ny_elem], int)
    diffuse_frac = np.empty([ny], int)
    diffuse_frac_elem = np.empty([ny_elem], int)
    corr_frac[:] = 0
    corr_frac_elem[:] = 0
    diffuse_frac[:] = 0
    diffuse_frac_elem[:] = 0

    indTmp = np.where((geo_meta.slope[:] == 0.0) & (SWDOWN <= 10.0))
    indTmp_elem = np.where((geo_meta.slope_elem[:] == 0.0) & (SWDOWN_elem <= 10.0))

    corr_frac[indTmp] = 1
    corr_frac_elem[indTmp_elem] = 1

    term1 = np.sin(xxlat) * np.cos(hrang2d)
    term1_elem = np.sin(xxlat_elem) * np.cos(hrang2d_elem)
    term2 = (0 - np.cos(geo_meta.slp_azi)) * np.sin(geo_meta.slope)
    term2_elem = (0 - np.cos(geo_meta.slp_azi_elem)) * np.sin(geo_meta.slope_elem)
    term3 = np.sin(hrang2d) * (np.sin(geo_meta.slp_azi) * np.sin(geo_meta.slope))
    term3_elem = np.sin(hrang2d_elem) * (
        np.sin(geo_meta.slp_azi_elem) * np.sin(geo_meta.slope_elem)
    )
    term4 = (np.cos(xxlat) * np.cos(hrang2d)) * np.cos(geo_meta.slope)
    term4_elem = (np.cos(xxlat_elem) * np.cos(hrang2d_elem)) * np.cos(
        geo_meta.slope_elem
    )
    term5 = np.cos(xxlat) * (np.cos(geo_meta.slp_azi) * np.sin(geo_meta.slope))
    term5_elem = np.cos(xxlat_elem) * (
        np.cos(geo_meta.slp_azi_elem) * np.sin(geo_meta.slope_elem)
    )
    term6 = np.sin(xxlat) * np.cos(geo_meta.slope)
    term6_elem = np.sin(xxlat_elem) * np.cos(geo_meta.slope_elem)

    csza_slp = (term1 * term2 - term3 + term4) * math.cos(declin) + (
        term5 + term6
    ) * math.sin(declin)
    csza_slp_elem = (term1_elem * term2_elem - term3_elem + term4_elem) * math.cos(
        declin
    ) + (term5_elem + term6_elem) * math.sin(declin)

    csza_slp[np.where(csza_slp <= 1e-4)] = 1e-4
    csza_slp_elem[np.where(csza_slp_elem <= 1e-4)] = 1e-4
    # Topographic shading
    # csza_slp[np.where(shadow == 1)] = 1E-4

    # Correction factor for sloping topographic: the diffuse fraction of solar
    # radiation is assumed to be unaffected by the slope.
    corr_fac = diffuse_frac + ((1 - diffuse_frac) * csza_slp) / COSZEN
    corr_fac_elem = (
        diffuse_frac_elem + ((1 - diffuse_frac_elem) * csza_slp_elem) / COSZEN_elem
    )
    corr_fac[np.where(corr_fac > 1.3)] = 1.3
    corr_fac_elem[np.where(corr_fac_elem > 1.3)] = 1.3

    # Peform downscaling
    SWDOWN_OUT = SWDOWN * corr_fac
    SWDOWN_OUT_elem = SWDOWN_elem * corr_fac_elem

    # Reset variables to free up memory
    # corr_frac = None
    diffuse_frac = None
    term1 = None
    term2 = None
    term3 = None
    term4 = None
    term5 = None
    term6 = None
    diffuse_frac_elem = None
    term1_elem = None
    term2_elem = None
    term3_elem = None
    term4_elem = None
    term5_elem = None
    term6_elem = None

    input_forcings.final_forcings[7, :] = SWDOWN_OUT
    input_forcings.final_forcings_elem[7, :] = SWDOWN_OUT_elem

    # Reset variables to free up memory
    SWDOWN = None
    SWDOWN_OUT = None
    SWDOWN_elem = None
    SWDOWN_OUT_elem = None


def rel_hum(input_forcings: InputForcings, config_options: ConfigOptions):
    """Calculate relative humidity given original, undownscaled surface pressure and 2-meter temperature.

    Function to calculate relative humidity given
    original, undownscaled surface pressure and 2-meter
    temperature.
    :param input_forcings:
    :param config_options:
    :return:
    """
    if config_options.grid_type == "gridded":
        tmpHumidity = input_forcings.final_forcings[5, :, :] / (
            1 - input_forcings.final_forcings[5, :, :]
        )
    elif config_options.grid_type == "unstructured":
        tmpHumidity = input_forcings.final_forcings[5, :] / (
            1 - input_forcings.final_forcings[5, :]
        )
        tmpHumidity_elem = input_forcings.final_forcings_elem[5, :] / (
            1 - input_forcings.final_forcings_elem[5, :]
        )
    elif config_options.grid_type == "hydrofabric":
        tmpHumidity = input_forcings.final_forcings[5, :] / (
            1 - input_forcings.final_forcings[5, :]
        )

    T0 = 273.15
    EP = 0.622
    ONEMEP = 0.378
    ES0 = 6.11
    A = 17.269
    B = 35.86

    if config_options.grid_type == "gridded":
        EST = ES0 * np.exp(
            (A * (input_forcings.t2dTmp - T0)) / (input_forcings.t2dTmp - B)
        )
        QST = (EP * EST) / ((input_forcings.psfcTmp * 0.01) - ONEMEP * EST)
        RH = 100 * (tmpHumidity / QST)

        # Reset variables to free up memory
        tmpHumidity = None

        return RH

    elif config_options.grid_type == "unstructured":
        EST = ES0 * np.exp(
            (A * (input_forcings.t2dTmp - T0)) / (input_forcings.t2dTmp - B)
        )
        EST_elem = ES0 * np.exp(
            (A * (input_forcings.t2dTmp_elem - T0)) / (input_forcings.t2dTmp_elem - B)
        )
        QST = (EP * EST) / ((input_forcings.psfcTmp * 0.01) - ONEMEP * EST)
        QST_elem = (EP * EST_elem) / (
            (input_forcings.psfcTmp_elem * 0.01) - ONEMEP * EST_elem
        )
        RH = 100 * (tmpHumidity / QST)
        RH_elem = 100 * (tmpHumidity_elem / QST_elem)
        # Reset variables to free up memory
        tmpHumidity = None
        tmpHumidity_elem = None

        return RH, RH_elem

    elif config_options.grid_type == "hydrofabric":
        EST = ES0 * np.exp(
            (A * (input_forcings.t2dTmp - T0)) / (input_forcings.t2dTmp - B)
        )
        QST = (EP * EST) / ((input_forcings.psfcTmp * 0.01) - ONEMEP * EST)
        RH = 100 * (tmpHumidity / QST)

        # Reset variables to free up memory
        tmpHumidity = None

        return RH


def mixhum_ptrh(
    input_forcings: InputForcings, relHum, iswit, config_options: ConfigOptions
):
    """Convert relative humidity back to a downscaled 2-meter specific humidity.

    Functionto convert relative humidity back to a downscaled
    2-meter specific humidity
    :param input_forcings:
    :param config_options:
    :return:
    """
    T0 = 273.15
    EP = 0.622
    ONEMEP = 0.378
    ES0 = 6.11
    A = 17.269
    B = 35.86

    if config_options.grid_type == "gridded":
        term1 = A * (input_forcings.final_forcings[4, :, :] - T0)
        term2 = input_forcings.final_forcings[4, :, :] - B
        EST = np.exp(term1 / term2) * ES0
        QST = (EP * EST) / (
            (input_forcings.final_forcings[6, :, :] / 100.0) - ONEMEP * EST
        )
    else:
        term1 = A * (input_forcings.final_forcings[4, :] - T0)
        term2 = input_forcings.final_forcings[4, :] - B
        EST = np.exp(term1 / term2) * ES0
        QST = (EP * EST) / (
            (input_forcings.final_forcings[6, :] / 100.0) - ONEMEP * EST
        )

    QW = QST * (relHum * 0.01)
    if iswit == 2:
        QW = QW / (1.0 + QW)
    if iswit < 0:
        QW = QW * 1000.0

    # Reset variables to free up memory
    term1 = None
    term2 = None
    EST = None
    QST = None
    psfcTmp = None

    return QW


def mixhum_ptrh_unstructured(
    input_forcings: InputForcings,
    relHum,
    relHum_elem,
    iswit,
    config_options: ConfigOptions,
):
    """Convert relative humidity back to a downscaled 2-meter specific humidity.

    Functionto convert relative humidity back to a downscaled
    2-meter specific humidity
    :param input_forcings:
    :param config_options:
    :return:
    """
    T0 = 273.15
    EP = 0.622
    ONEMEP = 0.378
    ES0 = 6.11
    A = 17.269
    B = 35.86

    term1 = A * (input_forcings.final_forcings[4, :] - T0)
    term2 = input_forcings.final_forcings[4, :] - B
    EST = np.exp(term1 / term2) * ES0
    QST = (EP * EST) / ((input_forcings.final_forcings[6, :] / 100.0) - ONEMEP * EST)
    term1_elem = A * (input_forcings.final_forcings_elem[4, :] - T0)
    term2_elem = input_forcings.final_forcings_elem[4, :] - B
    EST_elem = np.exp(term1_elem / term2_elem) * ES0
    QST_elem = (EP * EST_elem) / (
        (input_forcings.final_forcings_elem[6, :] / 100.0) - ONEMEP * EST_elem
    )

    QW = QST * (relHum * 0.01)
    QW_elem = QST_elem * (relHum_elem * 0.01)
    if iswit == 2:
        QW = QW / (1.0 + QW)
        QW_elem = QW_elem / (1.0 + QW_elem)
    if iswit < 0:
        QW = QW * 1000.0
        QW_elem = QW_elem * 1000.0

    # Reset variables to free up memory
    term1 = None
    term2 = None
    EST = None
    QST = None
    psfcTmp = None
    term1_elem = None
    term2_elem = None
    EST_elem = None
    QST_elem = None
    psfcTmp_elem = None

    return QW, QW_elem
