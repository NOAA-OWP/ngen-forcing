
raise NotImplementedError(f"This file, {__file__}, is deprecated.")

from __future__ import annotations

import datetime
import os
from typing import TYPE_CHECKING

from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.bias_correction import (
    run_bias_correction,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.disaggregateMod import (
    disaggregate_factory,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.downscale import (
    run_downscaling,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.err_handler import (
    check_forcing_bounds,
    check_program_status,
    check_supp_pcp_bounds,
    err_out_screen_para,
    log_critical,
    log_msg,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.layeringMod import (
    layer_final_forcings,
    layer_supplemental_forcing,
)

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
        MPIConfig,
    )
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.suppPrecipMod import (
        supplemental_precip,
    )


def process_forecasts(
    config_options: ConfigOptions,
    geo_meta: GeoMeta,
    input_forcing: InputForcings,
    supp_precip: supplemental_precip,
    mpi_config: MPIConfig,
    output_obj,
):
    """Process forecasts.

    Main calling module for running realtime forecasts and re-forecasts.
    :param jobMeta:
    :return:

    """
    # Loop through each WRF-Hydro forecast cycle being processed. Within
    # each cycle, perform the following tasks:
    # 1.) Loop over each output frequency
    # 2.) Determine the input forcing cycle dates (both before and after)
    #     for temporal interpolation, downscaling, and bias correction reasons.
    # 3.) If the input forcings haven't been opened and read into memory,
    #     open them.
    # 4.) Check to see if the ESMF objects for input forcings have been
    #     created. If not, create them, including the regridding object.
    # 5.) Regrid forcing grids for input cycle dates surrounding the
    #     current output timestep if they haven't been regridded.
    # 6.) Perform bias correction and/or downscaling.
    # 7.) Output final grids to LDASIN NetCDF files with associated
    #     WRF-Hydro geospatial metadata to the final output directories.
    # Throughout this entire process, log progress being made into LOG
    # files. Once a forecast cycle is complete, we will touch an empty
    # 'WrfHydroForcing.COMPLETE' flag in the directory. This will be
    # checked upon the beginning of this program to see if we
    # need to process any files.
    raise NotImplementedError(f"This file, {__file__}, is deprecated.")

    disaggregate_fun = disaggregate_factory(config_options)

    for fcst_cycle_num in range(config_options.nFcsts):
        config_options.current_fcst_cycle = (
            config_options.b_date_proc
            + datetime.timedelta(seconds=config_options.fcst_freq * 60 * fcst_cycle_num)
        )
        if config_options.first_fcst_cycle is None:
            config_options.first_fcst_cycle = config_options.current_fcst_cycle

        if config_options.ana_flag:
            fcst_cycle_out_dir = f"{config_options.output_dir}/{config_options.e_date_proc.strftime('%Y%m%d%H')}"
        else:
            fcst_cycle_out_dir = f"{config_options.output_dir}/{config_options.current_fcst_cycle.strftime('%Y%m%d%H')}"

        # reset skips if present
        for force_key in config_options.input_forcings:
            input_forcing[force_key].skip = False

        # put all AnA output in the same directory
        if config_options.ana_flag:
            if config_options.ana_out_dir is None:
                config_options.ana_out_dir = fcst_cycle_out_dir
            fcst_cycle_out_dir = config_options.ana_out_dir

        # completeFlag = config_options.scratch_dir + "/WrfHydroForcing.COMPLETE"
        complete_flag = f"{fcst_cycle_out_dir}/WrfHydroForcing.COMPLETE"
        if os.path.isfile(complete_flag):
            config_options.statusMsg = f"Forecast Cycle: {config_options.current_fcst_cycle.strftime('%Y-%m-%d %H:%M')} has already completed."
            log_msg(config_options, mpi_config)
            # We have already completed processing this cycle,
            # move on.
            continue

        if not config_options.ana_flag:
            if mpi_config.rank == 0:
                # If the cycle directory doesn't exist, create it.
                if not os.path.isdir(fcst_cycle_out_dir):
                    try:
                        os.mkdir(fcst_cycle_out_dir)
                    except Exception:
                        config_options.errMsg = (
                            f"Unable to create output directory: {fcst_cycle_out_dir}"
                        )
                        err_out_screen_para(config_options.errMsg, mpi_config)
            check_program_status(config_options, mpi_config)

        # Log information about this forecast cycle
        if mpi_config.rank == 0:
            config_options.statusMsg = "X" * 38
            log_msg(config_options, mpi_config)
            config_options.statusMsg = f"Processing Forecast Cycle: {config_options.current_fcst_cycle.strftime('%Y-%m-%d %H:%M')}"
            log_msg(config_options, mpi_config)
            config_options.statusMsg = f"Forecast Cycle Length is: {config_options.cycle_length_minutes!s} minutes"
            log_msg(config_options, mpi_config)
        # mpi_config.comm.barrier()

        # Loop through each output timestep. Perform the following functions:
        # 1.) Calculate all necessary input files per user options.
        # 2.) Read in input forcings from GRIB/NetCDF files.
        # 3.) Regrid the forcings, and temporally interpolate.
        # 4.) Downscale.
        # 5.) Layer, and output as necessary.
        ana_factor = 1 if config_options.ana_flag is False else 0
        show_message = True
        for out_step in range(1, config_options.num_output_steps + 1):
            # Reset out final grids to missing values.
            output_obj.output_local[:, :, :] = -9999.0

            config_options.current_output_step = out_step
            output_obj.outDate = config_options.current_fcst_cycle + datetime.timedelta(
                seconds=config_options.output_freq * 60 * out_step
            )
            config_options.current_output_date = output_obj.outDate

            # if AnA, adjust file date for analysis vs forecast
            if config_options.ana_flag:
                file_date = output_obj.outDate - datetime.timedelta(
                    seconds=config_options.output_freq * 60
                )
            else:
                file_date = output_obj.outDate

            # Calculate the previous output timestep. This is used in potential downscaling routines.
            if out_step == ana_factor:
                config_options.prev_output_date = config_options.current_output_date
            else:
                config_options.prev_output_date = (
                    config_options.current_output_date
                    - datetime.timedelta(seconds=config_options.output_freq * 60)
                )
            if mpi_config.rank == 0 and show_message:
                config_options.statusMsg = "========================================="
                log_msg(config_options, mpi_config, True)
                config_options.statusMsg = f"Processing for output timestep: {file_date.strftime('%Y-%m-%d %H:%M')}"
                log_msg(config_options, mpi_config, True)
            # mpi_config.comm.barrier()

            # Compose the expected path to the output file. Check to see if the file exists,
            # if so, continue to the next time step. Also initialize our output arrays if necessary.
            output_obj.outPath = f"{fcst_cycle_out_dir}/{file_date.strftime('%Y%m%d%H%M')}.LDASIN_DOMAIN1"
            # mpi_config.comm.barrier()

            if os.path.isfile(output_obj.outPath):
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Output file: {output_obj.outPath} exists. Moving to the next output timestep."
                    log_msg(config_options, mpi_config)
                check_program_status(config_options, mpi_config)
                continue
            else:
                config_options.currentForceNum = 0
                config_options.currentCustomForceNum = 0
                # Loop over each of the input forcings specifed.
                for force_key in config_options.input_forcings:
                    input_forcings = input_forcing[force_key]
                    # Calculate the previous and next input cycle files from the inputs.
                    input_forcings.calc_neighbor_files(
                        config_options, output_obj.outDate, mpi_config
                    )
                    check_program_status(config_options, mpi_config)

                    # break loop if done early
                    if input_forcings.skip is True:
                        show_message = False  # just to avoid confusion
                        break

                    # Regrid forcings.
                    input_forcings.regrid_inputs(config_options, geo_meta, mpi_config)
                    check_program_status(config_options, mpi_config)

                    # Run check on regridded fields for reasonable values that are not missing values.
                    check_forcing_bounds(config_options, input_forcings, mpi_config)
                    check_program_status(config_options, mpi_config)

                    # If we are restarting a forecast cycle, re-calculate the neighboring files, and regrid the
                    # next set of forcings as the previous step just regridded the previous forcing.
                    if input_forcings.rstFlag == 1:
                        if (
                            input_forcings.regridded_forcings1 is not None
                            and input_forcings.regridded_forcings2 is not None
                        ):
                            # Set the forcings back to reflect we just regridded the previous set of inputs, not the next.
                            input_forcings.regridded_forcings1[:, :, :] = (
                                input_forcings.regridded_forcings2[:, :, :]
                            )

                        # Re-calculate the neighbor files.
                        input_forcings.calc_neighbor_files(
                            config_options, output_obj.outDate, mpi_config
                        )
                        check_program_status(config_options, mpi_config)

                        # Regrid the forcings for the end of the window.
                        input_forcings.regrid_inputs(
                            config_options, geo_meta, mpi_config
                        )
                        check_program_status(config_options, mpi_config)

                        input_forcings.rstFlag = 0

                    # Run temporal interpolation on the grids.
                    input_forcings.temporal_interpolate_inputs(
                        config_options, mpi_config
                    )
                    check_program_status(config_options, mpi_config)

                    # Run bias correction.
                    run_bias_correction(
                        input_forcings, config_options, geo_meta, mpi_config
                    )
                    check_program_status(config_options, mpi_config)

                    # Run downscaling on grids for this output timestep.
                    run_downscaling(
                        input_forcings, config_options, geo_meta, mpi_config
                    )
                    check_program_status(config_options, mpi_config)

                    # Layer in forcings from this product.
                    layer_final_forcings(
                        output_obj, input_forcings, config_options, mpi_config
                    )
                    check_program_status(config_options, mpi_config)

                    config_options.currentForceNum = config_options.currentForceNum + 1

                    if force_key == 10:
                        config_options.currentCustomForceNum = (
                            config_options.currentCustomForceNum + 1
                        )

                else:
                    # Process supplemental precipitation if we specified in the configuration file.
                    if config_options.number_supp_pcp > 0:
                        for supp_pcp_key in config_options.supp_precip_forcings:
                            # Like with input forcings, calculate the neighboring files to use.
                            supp_precip[supp_pcp_key].calc_neighbor_files(
                                config_options, output_obj.outDate, mpi_config
                            )
                            check_program_status(config_options, mpi_config)

                            # Regrid the supplemental precipitation.
                            supp_precip[supp_pcp_key].regrid_inputs(
                                config_options, geo_meta, mpi_config
                            )
                            check_program_status(config_options, mpi_config)

                            if (
                                supp_precip[supp_pcp_key].regridded_precip1 is not None
                                and supp_precip[supp_pcp_key].regridded_precip2
                                is not None
                            ):
                                # if np.any(supp_precip[supp_pcp_key].regridded_precip1) and \
                                #        np.any(supp_precip[supp_pcp_key].regridded_precip2):
                                # Run check on regridded fields for reasonable values that are not missing values.
                                check_supp_pcp_bounds(
                                    config_options,
                                    supp_precip[supp_pcp_key],
                                    mpi_config,
                                )
                                check_program_status(config_options, mpi_config)

                                disaggregate_fun(
                                    input_forcings,
                                    supp_precip[supp_pcp_key],
                                    config_options,
                                    mpi_config,
                                )
                                check_program_status(config_options, mpi_config)

                                # Run temporal interpolation on the grids.
                                supp_precip[supp_pcp_key].temporal_interpolate_inputs(
                                    config_options, mpi_config
                                )
                                check_program_status(config_options, mpi_config)

                                # Layer in the supplemental precipitation into the current output object.
                                layer_supplemental_forcing(
                                    output_obj,
                                    supp_precip[supp_pcp_key],
                                    config_options,
                                    mpi_config,
                                )
                                check_program_status(config_options, mpi_config)

                    # Call the output routines
                    #   adjust date for AnA if necessary
                    if config_options.ana_flag:
                        output_obj.outDate = file_date

                    output_obj.output_final_ldasin(config_options, geo_meta, mpi_config)
                    check_program_status(config_options, mpi_config)

        if (not config_options.ana_flag) or (
            fcst_cycle_num == (config_options.nFcsts - 1)
        ):
            if mpi_config.rank == 0:
                config_options.statusMsg = f"Forcings complete for forecast cycle: {config_options.current_fcst_cycle.strftime('%Y-%m-%d %H:%M')}"
                log_msg(config_options, mpi_config)
            check_program_status(config_options, mpi_config)

            # Success.... Now touch an empty complete file for this forecast cycle to indicate
            # completion in case the code is re-ran.
            try:
                open(complete_flag, "a").close()
            except Exception:
                config_options.errMsg = (
                    f"Unable to create completion file: {complete_flag}"
                )
                log_critical(config_options, mpi_config)
            check_program_status(config_options, mpi_config)
