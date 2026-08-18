import inspect
import logging
import os
import sys
import traceback

import numpy as np
from mpi4py import MPI
from scipy import spatial

# Use the Error, Warning, and Trapping System Package for logging
import ewts
LOG = ewts.get_logger(ewts.FORCING_ID)


def in_exception_context() -> bool:
    if sys.exc_info()[0] is not None:
        return True
    return False


def err_out_screen(err_msg: str, exc: BaseException | None = None):
    """Print an error message to the screen and exit the program gracefully.

    Generic routine to exit the program gracefully. This specific error function does not log
    error messages to the log file, but simply prints them out to the screen. This function
    is designed specifically for early in the program execution where a log file hasn't been
    established yet.

    If an exception is provided, its text will be appended to the error message.

    Logan Karsten - National Center for Atmospheric Research, karsten@ucar.edu
    """
    if exc is not None:
        err_msg += f" - {exc}"
    err_msg_out = "ERROR: " + err_msg

    print(err_msg_out, flush=True)
    LOG.critical(err_msg_out)

    if in_exception_context():
        tb = traceback.format_exc()
        tb_msg = f"TRACEBACK: {tb}"
        print(tb_msg, flush=True, file=sys.stderr)
        LOG.critical(tb_msg)

    final_msg = f"Calling sys.exit(1) from object {repr(inspect.currentframe().f_code.co_name)}"
    print(final_msg, flush=True, file=sys.stderr)
    LOG.critical(final_msg)

    sys.exit(1)


def err_out_screen_para(err_msg: str, MpiConfig, exc: BaseException | None = None):
    """Print an error message to the screen and abort MPI.

    Generic function for printing an error message to the screen and aborting MPI.
    This should only be called if logging cannot occur and an abrupt end the program
    is needed.

    If an exception is provided, its text will be appended to the error message.

    :param err_msg: The base error message string.
    :param MpiConfig: The MPI configuration object (must include 'rank').
    :param exc: Optional exception object to append to the error message.
    :return: None
    """
    if exc is not None:
        err_msg += f" - {exc}"
    err_msg_out = f"ERROR: RANK - {MpiConfig.rank} : {err_msg}"
    print(err_msg_out, flush=True)
    traceback.print_exc()  # Only prints if an exception is currently being handled
    # MpiConfig.comm.Abort()
    sys.exit(1)


def check_program_status(
    ConfigOptions, MpiConfig, rank_0_reduce: bool = True, any_rank_abort: bool = False
):
    """Check the err statuses for each processor in the program.

    If any flags come back, gracefully exit the program.
    :param ConfigOptions:
    :param MpiConfig:
    :param rank_0_reduce: (default = True)
        If True, then rank 0 will call MPI reduce(), causing it to block waiting for other ranks to send their exceptions.
        This should be set to False only for special cases that may experience deadlocks without it, such as certain ESMF calls.
    :param any_rank_abort: (default = False)
        If True, then any rank calling this may log its own error message and call MPI Abort() itself, not only rank 0.
        This should be set to True only for special cases that may experience deadlocks without it, such as certain ESMF calls.
    :return:

    NOTE: when rank_0_reduce is set to False, rank 0 will not receive error flags from other ranks.
    So in this case, any_rank_abort must be set to True, so that each rank can be responsible for
    checking its own flag and calling Abort() as needed.
    """
    # Sync up processors to ensure everyone is on the same page.
    # MpiConfig.comm.barrier()

    # Collect values from each processor.
    # data = MpiConfig.comm.gather(ConfigOptions.errFlag, root=0)
    # if MpiConfig.rank == 0:
    #     for i in range(MpiConfig.size):
    #         if data[i] != 0:
    #             MpiConfig.comm.Abort()
    #             sys.exit(1)
    # else:
    #     assert data is None

    # Reduce version:

    # Non-0 ranks should always call reduce since they send information and do not block.
    # By default, rank 0 should call reduce to collect the exceptions from the other ranks,
    # but there are special known conditions that can cause rank 0 to deadlock here,
    # so, optionally, rank 0 can skip the reduce call, and check only its own message.
    # Whenever that option is used, the any_rank_abort option must also be used.

    if (not rank_0_reduce) and (not any_rank_abort):
        raise ValueError(
            "When rank_0_reduce is Falsy, any_rank_abort must be Truthy, but both are Falsy."
        )

    if MpiConfig.rank != 0 or rank_0_reduce:
        any_error = MpiConfig.comm.reduce(ConfigOptions.errFlag)
    else:
        any_error = None

    if MpiConfig.rank == 0 or any_rank_abort:
        if ConfigOptions.errFlag or any_error:
            # print("any_error: ", any_error, type(any_error), flush=True)
            stack = traceback.format_stack()[:-1]
            for frame in stack:
                LOG.error(frame)
            MpiConfig.abort_with_cleanup(1)

    # Sync up processors.
    # When this is enabled, then all ranks wait for rank 0 to evaluate the
    # collected error flags and call Abort() as appropriate, before continuing.
    # When this is not enabled, then non-0 ranks may continue execution after sending
    # their error flag, since non-0 ranks do not block on reduce().
    # MpiConfig.comm.barrier()

def err_out(ConfigOptions):
    """Error out after an error message has been logged for a forecast cycle.

    Function to error out after an error message has been logged for a
    forecast cycle. We will exit with a non-zero exit status.
    :param ConfigOptions:
    :return:
    """

    try:
        LOG.error(ConfigOptions.errMsg)
    except Exception:
        ConfigOptions.errMsg = (
            "Unable to write error message"
        )
        raise Exception()
    MPI.Finalize()
    sys.exit(1)


def log_error(ConfigOptions, MpiConfig, msg: str = None):
    """Log an error message to the log file.

    :param ConfigOptions:
    :param MpiConfig:
    :param msg: Optional error message string, overrides current value for ConfigOptions.errMsg in-place before sending log call.
    :return:
    """
    if msg is not None:
        if not isinstance(msg, str):
            raise TypeError(
                f"Expected type str or NoneType for msg, got type: {type(msg)}"
            )
        ConfigOptions.errMsg = msg

    try:
        LOG.error("RANK: " + str(MpiConfig.rank) + " - " + ConfigOptions.errMsg)
    except Exception:
        err_out_screen_para(
            (
                "Unable to write ERROR message on RANK: "
                + str(MpiConfig.rank)
            ),
            MpiConfig,
        )
    ConfigOptions.errFlag = 1


def log_critical(ConfigOptions, MpiConfig, msg: str = None):
    """Log an error message without exiting with a non-zero exit status.

    :param ConfigOptions:
    :param msg: Optional error message string, overrides current value for ConfigOptions.errMsg in-place before sending log call.
    :return:
    """
    if msg is not None:
        if not isinstance(msg, str):
            raise TypeError(
                f"Expected type str or NoneType for msg, got type: {type(msg)}"
            )
        ConfigOptions.errMsg = msg

    try:
        LOG.critical("RANK: " + str(MpiConfig.rank) + " - " + ConfigOptions.errMsg)
    except Exception:
        err_out_screen_para(
            (
                "Unable to write CRITICAL message on RANK: "
                + str(MpiConfig.rank)
            ),
            MpiConfig,
        )

    # Add this for debugging:
    LOG.debug(f"log_critical called on RANK {MpiConfig.rank}: {ConfigOptions.errMsg}")

    ConfigOptions.errFlag = 1


def log_warning(ConfigOptions, MpiConfig, msg: str = None):
    """Log a warning message to the log file.

    :param ConfigOptions:
    :param msg: Optional error message string, overrides current value for ConfigOptions.statusMsg in-place before sending log call.
    :return:
    """
    if msg is not None:
        if not isinstance(msg, str):
            raise TypeError(
                f"Expected type str or NoneType for msg, got type: {type(msg)}"
            )
        ConfigOptions.statusMsg = msg

    try:
        LOG.warning("RANK: " + str(MpiConfig.rank) + " - " + ConfigOptions.statusMsg)
    except Exception:
        err_out_screen_para(
            (
                "Unable to write WARNING message on RANK: "
                + str(MpiConfig.rank)
            ),
            MpiConfig,
        )


def log_msg(ConfigOptions, MpiConfig, debug: bool = False, msg: str = None):
    """Log INFO messages to a specified log file.

    :param ConfigOptions:
    :param msg: Optional error message string, overrides current value for ConfigOptions.statusMsg in-place before sending log call.
    :return:
    """
    if not isinstance(debug, bool):
        raise TypeError(f"Expected type bool for debug, got type: {type(debug)}")
    
    if msg is not None:
        if not isinstance(msg, str):
            raise TypeError(
                f"Expected type str or NoneType for msg, got type: {type(msg)}"
            )
        ConfigOptions.statusMsg = msg

    try:
        if debug:
            LOG.debug("RANK: " + str(MpiConfig.rank) + " - " + ConfigOptions.statusMsg)
        else:
            LOG.info("RANK: " + str(MpiConfig.rank) + " - " + ConfigOptions.statusMsg)
    except Exception:
        err_out_screen_para(
            (
                "Unable to write log_msg message on RANK: "
                + str(MpiConfig.rank)
            ),
            MpiConfig,
        )


def check_forcing_bounds(ConfigOptions, input_forcings, MpiConfig):
    """Check the bounds of forcing variables for reasonable values.

    Function for running a reasonable value check for individual forcing
    variables. This one check type with the other checking for final missing
    values in the final output grid.
    :param ConfigOptions:
    :param input_forcings:
    :param MpiConfig:
    :return:
    """
    # Establish a range of values for each output variable.
    variable_range = {
        "U2D": [0, -500.0, 500.0],
        "V2D": [1, -500.0, 500.0],
        "LWDOWN": [2, -1000.0, 10000.0],
        "RAINRATE": [3, 0.0, 100.0],
        "T2D": [4, 0.0, 400.0],
        "Q2D": [5, -100.0, 100.0],
        "PSFC": [6, 0.0, 2000000.0],
        "SWDOWN": [7, 0.0, 5000.0],
        "LQFRAC": [8, 0, 1],
    }
    fvars = [
        "U2D",
        "V2D",
        "LWDOWN",
        "RAINRATE",
        "T2D",
        "Q2D",
        "PSFC",
        "SWDOWN",
        "LQFRAC",
    ]

    # If the regridded field is None type, return to the main program as this means no forcings
    # were found for this timestep.
    if input_forcings.regridded_forcings2 is None:
        return

    # Loop over all the variables. Check for reasonable ranges. If any values are
    # exceeded, shut the forcing engine down.
    for varTmp in variable_range:
        if fvars.index(varTmp) not in input_forcings.input_map_output:
            continue

        if varTmp == "LQFRAC" and not ConfigOptions.include_lqfrac:
            continue

        # First check to see if we have any data that is not missing.
        # indCheck = np.where(input_forcings.regridded_forcings2[variable_range[varTmp][0]]
        #                    != ConfigOptions.globalNdv)

        # if len(indCheck[0]) == 0:
        #    ConfigOptions.errMsg = "No valid data found for " + varTmp + " in " + input_forcings.file_in2
        #    log_critical(ConfigOptions, MpiConfig)
        #    indCheck = None
        #    return
        if not ConfigOptions.aws:
            src_file = input_forcings.file_in2
            if "%FIELD%" in src_file:
                src_file = src_file.replace(
                    "%FIELD%",
                    {
                        "U2D": "[wdir|wspd]",
                        "V2D": "[wdir|wspd]",
                        "RAINRATE": "qpf",
                        "T2D": "tmp",
                    }[varTmp],
                )

        # Check to see if any pixel cells are below the minimum value.
        indCheck = np.where(
            (
                input_forcings.regridded_forcings2[variable_range[varTmp][0]]
                != ConfigOptions.globalNdv
            )
            & (
                input_forcings.regridded_forcings2[variable_range[varTmp][0]]
                < variable_range[varTmp][1]
            )
        )
        numCells = len(indCheck[0])
        if numCells > 0:
            min = input_forcings.regridded_forcings2[variable_range[varTmp][0]][
                indCheck
            ].min()
            if input_forcings.product_name == "NWM":
                ConfigOptions.errMsg = (
                    f"Data (min = {min}) below minimum threshold for: {varTmp} in "
                    f"NWM data for {numCells} regridded pixel cells."
                )
            else:
                ConfigOptions.errMsg = (
                    f"Data (min = {min}) below minimum threshold for: {varTmp} in "
                    f"{input_forcings.file_in2} for {numCells} regridded pixel cells."
                )
            log_critical(ConfigOptions, MpiConfig)
            indCheck = None
            return

        # Check to see if any pixel cells are above the maximum value.
        indCheck = np.where(
            (
                input_forcings.regridded_forcings2[variable_range[varTmp][0]]
                != ConfigOptions.globalNdv
            )
            & (
                input_forcings.regridded_forcings2[variable_range[varTmp][0]]
                > variable_range[varTmp][2]
            )
        )
        numCells = len(indCheck[0])
        if numCells > 0:
            max = input_forcings.regridded_forcings2[variable_range[varTmp][0]][
                indCheck
            ].max()
            if input_forcings.product_name == "NWM":
                ConfigOptions.errMsg = (
                    f"Data (max = {max}) above maximum threshold for: {varTmp} in "
                    f"NWM data for {numCells} regridded pixel cells."
                )
            else:
                ConfigOptions.errMsg = (
                    f"Data (max = {max}) above maximum threshold for: {varTmp} in "
                    f"{src_file} for {numCells} regridded pixel cells."
                )
            log_critical(ConfigOptions, MpiConfig)
            indCheck = None
            return

    indCheck = None
    return


def check_supp_pcp_bounds(ConfigOptions, supplemental_precip, MpiConfig, GeoMeta):
    """Check the bounds of supplemental precipitation values for reasonable values.

    Function for running a reasonable value check on supplemental precipitation
    values. This is one check type with the other checking for final missing
    values in the final output grid.
    :param ConfigOptions:
    :param supplemental_precip:
    :param MpiConfig:
    :return:
    """
    # If the regridded field is None type, return to the main program as this means no forcings
    # were found for this timestep.
    if supplemental_precip.regridded_precip2 is None:
        return

    # Check to see if any pixel cells are below the minimum value.
    indCheck = np.where(
        (supplemental_precip.regridded_precip2 != ConfigOptions.globalNdv)
        & (supplemental_precip.regridded_precip2 < 0.0)
    )
    indCheck_valid = np.where(
        (supplemental_precip.regridded_precip2 > 0.0)
        & (supplemental_precip.regridded_precip2 < 100.0)
    )
    numCells = len(indCheck[0])
    if numCells > 0:
        ConfigOptions.errMsg = (
            "Supplemental precip data below minimum threshold for in "
            + supplemental_precip.file_in2
            + " for "
            + str(numCells)
            + " regridded pixel cells."
        )
        valid_coords = np.empty(
            (len(GeoMeta.latitude_grid[indCheck_valid[0]]), 2), dtype=float
        )
        invalid_coords = np.empty(
            (len(GeoMeta.latitude_grid[indCheck[0]]), 2), dtype=float
        )
        valid_coords[:, 0] = GeoMeta.longitude_grid[indCheck_valid[0]]
        valid_coords[:, 0] = GeoMeta.latitude_grid[indCheck_valid[0]]
        invalid_coords[:, 0] = GeoMeta.longitude_grid[indCheck[0]]
        invalid_coords[:, 0] = GeoMeta.latitude_grid[indCheck[0]]
        distance, pet_inds = spatial.KDTree(valid_coords).query(invalid_coords)
        supplemental_precip.regridded_precip2[indCheck[0]] = (
            supplemental_precip.regridded_precip2[pet_inds]
        )
        del valid_coords
        del invalid_coords
        del distance
        del pet_inds
        indCheck = None
        indCheck_valid = None
        return

    # Check to see if any pixel cells are above the maximum value.
    indCheck = np.where(
        (supplemental_precip.regridded_precip2 != ConfigOptions.globalNdv)
        & (supplemental_precip.regridded_precip2 > 100.0)
    )
    indCheck_valid = np.where(
        (supplemental_precip.regridded_precip2 > 0.0)
        & (supplemental_precip.regridded_precip2 < 100.0)
    )
    numCells = len(indCheck[0])
    if numCells > 0:
        ConfigOptions.errMsg = (
            "Supplemental precip data above maximum threshold for in "
            + supplemental_precip.file_in2
            + " for "
            + str(numCells)
            + " regridded pixel cells."
        )
        valid_coords = np.empty(
            (len(GeoMeta.latitude_grid[indCheck_valid[0]]), 2), dtype=float
        )
        invalid_coords = np.empty(
            (len(GeoMeta.latitude_grid[indCheck[0]]), 2), dtype=float
        )
        valid_coords[:, 0] = GeoMeta.longitude_grid[indCheck_valid[0]]
        valid_coords[:, 0] = GeoMeta.latitude_grid[indCheck_valid[0]]
        invalid_coords[:, 0] = GeoMeta.longitude_grid[indCheck[0]]
        invalid_coords[:, 0] = GeoMeta.latitude_grid[indCheck[0]]
        distance, pet_inds = spatial.KDTree(valid_coords).query(invalid_coords)
        # log_critical(ConfigOptions, MpiConfig)
        supplemental_precip.regridded_precip2[indCheck[0]] = (
            supplemental_precip.regridded_precip2[pet_inds]
        )

        indCheck = np.where(
            (supplemental_precip.regridded_precip2 != ConfigOptions.globalNdv)
            & (supplemental_precip.regridded_precip2 > 100.0)
        )
        indCheck_valid = np.where(
            (supplemental_precip.regridded_precip2 > 0.0)
            & (supplemental_precip.regridded_precip2 < 100.0)
        )
        if len(indCheck[0]) > 0:
            valid_coords = np.empty(
                (len(GeoMeta.latitude_grid[indCheck_valid[0]]), 2), dtype=float
            )
            invalid_coords = np.empty(
                (len(GeoMeta.latitude_grid[indCheck[0]]), 2), dtype=float
            )
            valid_coords[:, 0] = GeoMeta.longitude_grid[indCheck_valid[0]]
            valid_coords[:, 0] = GeoMeta.latitude_grid[indCheck_valid[0]]
            invalid_coords[:, 0] = GeoMeta.longitude_grid[indCheck[0]]
            invalid_coords[:, 0] = GeoMeta.latitude_grid[indCheck[0]]
            distance, pet_inds = spatial.KDTree(valid_coords).query(invalid_coords)
            supplemental_precip.regridded_precip2[indCheck[0]] = (
                supplemental_precip.regridded_precip2[pet_inds]
            )
        del valid_coords
        del invalid_coords
        del distance
        del pet_inds
        indCheck = None
        indCheck_valid = None
        return

    # Check to see if any pixel cells are above the maximum value.
    indCheck = np.where(
        (supplemental_precip.regridded_precip2 != ConfigOptions.globalNdv)
        & (supplemental_precip.regridded_precip2 > 100.0)
    )
    numCells = len(indCheck[0])
    if numCells > 0:
        ConfigOptions.errMsg = (
            "Supplemental precip data above maximum threshold for in "
            + supplemental_precip.file_in2
            + " for "
            + str(numCells)
            + " regridded pixel cells."
        )
        log_critical(ConfigOptions, MpiConfig)

    return


def check_missing_final(outPath, ConfigOptions, output_grid, var_name, MpiConfig):
    """Check the final output grids for missing values.

    Function that checks the final output grids to ensure no missing values
    are in place. Final output grids cannot contain any missing values per
    WRF-Hydro requirements.
    ;:param outPath:
    :param ConfigOptions:
    :param output_grid:
    :param var_name:
    :param MpiConfig:
    :return:
    """
    # Run NDV check. If ANY values are found, throw an error and shut the
    # forcing engine down.
    indCheck = np.where(output_grid == ConfigOptions.globalNdv)

    if len(indCheck[0]) > 0:
        ConfigOptions.errMsg = (
            "Found "
            + str(len(indCheck[0]))
            + " NDV pixel cells in output grid for: "
            + var_name
        )
        log_critical(ConfigOptions, MpiConfig)
        indCheck = None
        # If the output file has been created, remove it as it will be empty.
        if os.path.isfile(outPath):
            os.remove(outPath)
        return
    else:
        indCheck = None
        return
