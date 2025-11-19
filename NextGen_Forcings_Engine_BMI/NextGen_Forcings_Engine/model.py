import datetime
import os

import dask
import dask.delayed
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
import time
import dask.array as da
# from mpi4py.futures import MPIPoolExecutor
from mpi4py.futures import MPICommExecutor

from .core import bias_correction
from .core import disaggregateMod
from .core import downscale
from .core import err_handler
from .core import layeringMod
from . import nwm_proc

class NWMv3_Forcing_Engine_model:
    # TODO: refactor the bmi_model.py file and this to have this type maintain its own state.
    # def __init__(self):
    #    super(ngen_model, self).__init__()
    #    #self._model = model

    # @dask.delayed
    # def aws_obj(files):
    #    return xr.open_mfdataset(files, engine="zarr", parallel=True, consolidated=True)

    def run(self, model: dict, future_time: float, ConfigOptions, wrfHydroGeoMeta, inputForcingMod, suppPcpMod, MpiConfig, OutputObj):
        """
        Executes the full forcings engine BMI pipeline for a given future timestep.

        This method updates the `model` state dictionary with atmospheric forcings computed from
        available input datasets. It handles initialization, AWS Zarr loading, regridding, temporal
        interpolation, bias correction, downscaling, supplemental precipitation processing, and output
        population into the model structure.

        The following steps are performed:

        1. Determine the current forecast and output times based on the future timestamp
           and analysis mode (AnA or forecast).
        2. Initialize or reset output grids and step counters.
        3. Loop over each input forcing product:
           a. Calculate neighboring input files.
           b. Load AWS-hosted Zarr datasets if needed.
           c. Regrid input forcings to the model grid.
           d. Perform temporal interpolation.
           e. Apply bias correction and downscaling.
           f. Layer final forcings into the output object.
        4. Optionally process supplemental precipitation forcings:
           a. Regrid and validate.
           b. Disaggregate and interpolate.
           c. Layer into the final output.
        5. Write output to NetCDF forcing files if requested.
        6. Update the model state dictionary with flattened arrays.
        7. Advance the BMI time index.

        :param model: The model state dictionary that will be updated with new forcing data.
        :param future_time: The number of seconds into the future to advance the model.
        :param ConfigOptions: Configuration object containing all model options, flags, and paths.
        :param wrfHydroGeoMeta: Geospatial metadata needed for regridding and interpolation.
        :param inputForcingMod: Dictionary of initialized input forcing modules indexed by forcing key.
        :param suppPcpMod: Dictionary of supplemental precipitation modules indexed by key.
        :param MpiConfig: Object containing MPI communication settings such as rank and communicator.
        :param OutputObj: Output object that stores the generated atmospheric forcing arrays.

        :raises RuntimeError: If the model fails to initialize or if required arguments are missing.
        """

        # Assign the future time to the configuration
        ConfigOptions.bmi_time = future_time
        disaggregate_fun = disaggregateMod.disaggregate_factory(ConfigOptions)

        # Calculate current time stamp based on operational configuration
        if ConfigOptions.ana_flag:
            # If we're in an AnA configuration, then must offset the BMI future
            # timestamp to account for the "lookback" period being properly iterated
            # over between 3-28 hour look back time period and operation configuration
            if ConfigOptions.input_forcings[0] in [20, 22]:
                ConfigOptions.current_fcst_cycle = ConfigOptions.b_date_proc + pd.TimedeltaIndex(
                    np.array([future_time - 7200.0], dtype=float), 's')[0]
                ConfigOptions.current_time = ConfigOptions.b_date_proc + pd.TimedeltaIndex(
                    np.array([future_time - 7200.0], dtype=float), 's')[0]
                ConfigOptions.future_time = future_time
            else:
                # Puerto Rico / Hawaii AnA: 1-hour lookback (based on 6-hourly forecast cycles)
                ConfigOptions.current_fcst_cycle = ConfigOptions.b_date_proc + pd.TimedeltaIndex(
                    np.array([future_time - 3600.0], dtype=float), 's')[0]
                ConfigOptions.current_time = ConfigOptions.b_date_proc + pd.TimedeltaIndex(
                    np.array([future_time - 3600.0], dtype=float), 's')[0]
        else:
            # Forecast-only mode — use BMI timestamp as-is
            ConfigOptions.current_fcst_cycle = ConfigOptions.b_date_proc
            ConfigOptions.current_time = pd.Timestamp(ConfigOptions.b_date_proc) + pd.to_timedelta(future_time, unit='s')

        print("NextGen Forcings Engine processing meteorological forcings for BMI timestamp")
        print(f"Model.py current time: {ConfigOptions.current_time}")
        print(f"Model.py current fcst cycle: {ConfigOptions.current_fcst_cycle}")

        if ConfigOptions.first_fcst_cycle is None:
            ConfigOptions.first_fcst_cycle = ConfigOptions.current_fcst_cycle

        if not ConfigOptions.precip_only_flag:
            # reset skips if present
            for forceKey in ConfigOptions.input_forcings:
                inputForcingMod[forceKey].skip = False

            # Determine log timestamp
            if ConfigOptions.ana_flag:
                log_time = ConfigOptions.b_date_proc
            else:
                log_time = ConfigOptions.current_fcst_cycle

            # Compose a path to a log file, which will contain information about this forecast cycle
            log_filename = (
                f"LOG_{ConfigOptions.nwmConfig}"
                f"{'_' if ConfigOptions.nwmConfig != 'long_range' else f'_mem{ConfigOptions.cfsv2EnsMember}_'}"
                f"{ConfigOptions.d_program_init.strftime('%Y%m%d%H%M')}_{log_time.strftime('%Y%m%d%H%M')}"
                ".log"
            )
            ConfigOptions.logFile = os.path.join(ConfigOptions.scratch_dir, log_filename)

            # Initialize logging
            try:
                err_handler.init_log(ConfigOptions, MpiConfig)
            except Exception:
                err_handler.err_out_screen_para(ConfigOptions.errMsg, MpiConfig)

            err_handler.check_program_status(ConfigOptions, MpiConfig)

        # Log information about this forecast cycle
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
            err_handler.log_msg(ConfigOptions, MpiConfig)
            ConfigOptions.statusMsg = 'Processing Forecast Cycle: ' + \
                                      ConfigOptions.current_fcst_cycle.strftime('%Y-%m-%d %H:%M')
            err_handler.log_msg(ConfigOptions, MpiConfig)
            ConfigOptions.statusMsg = 'Forecast Cycle Length is: ' + \
                                      str(ConfigOptions.cycle_length_minutes) + " minutes"
            err_handler.log_msg(ConfigOptions, MpiConfig)
        # MpiConfig.comm.barrier()

        # Loop through each output timestep. Perform the following functions:
        # 1.) Calculate all necessary input files per user options.
        # 2.) Read in input forcings from GRIB/NetCDF files.
        # 3.) Regrid the forcings, and temporally interpolate.
        # 4.) Downscale.
        # 5.) Layer, and output as necessary.
        ana_factor = 1 if ConfigOptions.ana_flag is False else 0
        show_message = True
        if not ConfigOptions.precip_only_flag:
            if ConfigOptions.grid_type == "gridded":
                # Reset out final grids to missing values.
                OutputObj.output_local[:, :, :] = ConfigOptions.globalNdv
            elif ConfigOptions.grid_type == "unstructured":
                # Reset out final grids to missing values.
                OutputObj.output_local[:, :] = ConfigOptions.globalNdv
                OutputObj.output_local_elem[:, :] = ConfigOptions.globalNdv
            elif ConfigOptions.grid_type == "hydrofabric":
                # Reset out final grids to missing values.
                OutputObj.output_local[:, :] = ConfigOptions.globalNdv

            # Increment or initialize output step count
            if ConfigOptions.current_output_step is None:
                ConfigOptions.current_output_step = 1
            else:
                ConfigOptions.current_output_step += 1

            # Optional sub-output timestamp
            if ConfigOptions.sub_output_hour is not None:
                # TODO This is not used
                subOutDate = ConfigOptions.first_fcst_cycle + datetime.timedelta(hours=ConfigOptions.sub_output_hour)

            # Compute the output timestamp for this step
            if ConfigOptions.ana_flag:
                OutputObj.outDate = ConfigOptions.current_fcst_cycle + datetime.timedelta(seconds=ConfigOptions.output_freq * 60)
            else:
                OutputObj.outDate = ConfigOptions.current_fcst_cycle + datetime.timedelta(seconds=future_time)

            ConfigOptions.current_output_date = OutputObj.outDate

            # Adjust file_date for AnA if needed
            file_date = OutputObj.outDate - datetime.timedelta(seconds=ConfigOptions.output_freq * 60) if ConfigOptions.ana_flag else OutputObj.outDate

            # Compute previous output date (used for downscaling logic)
            if ConfigOptions.current_output_step == ana_factor:
                ConfigOptions.prev_output_date = ConfigOptions.current_output_date
            else:
                ConfigOptions.prev_output_date = ConfigOptions.current_output_date - datetime.timedelta(seconds=future_time)

            # Print message on log file indicating the timestamp
            # we are currently processing for forcings
            if MpiConfig.rank == 0 and show_message:
                ConfigOptions.statusMsg = '========================================='
                err_handler.log_msg(ConfigOptions, MpiConfig)
                ConfigOptions.statusMsg = f"Processing for output timestep: {file_date.strftime('%Y-%m-%d %H:%M')}"
                err_handler.log_msg(ConfigOptions, MpiConfig)

            ConfigOptions.currentForceNum = 0
            ConfigOptions.currentCustomForceNum = 0
            print(f"ConfigOptions.input_forcings: {ConfigOptions.input_forcings}")
            # Loop over each of the input forcings specified.
            print(f"\n[INFO] Model.py forcing loop: {len(ConfigOptions.input_forcings)} forcings configured: {ConfigOptions.input_forcings}")

            for forceKey in ConfigOptions.input_forcings:
                print('forceKey', forceKey)
                print(f"ConfigOptions.aws: {ConfigOptions.aws}")
                # Pass these methods for AORC data is ERA5-Interim blend is requested
                # so we can finish filling in the missing gaps
                if forceKey == 23 and 12 in ConfigOptions.input_forcings and 21 in ConfigOptions.input_forcings:
                    input_forcings = inputForcingMod[forceKey]

                    # These are not used
                    # AORC_mask = input_forcings.regridded_mask_AORC
                    # AORC_elem_mask = input_forcings.regridded_mask_elem_AORC
                else:
                    input_forcings = inputForcingMod[forceKey]
                    input_forcings.calc_neighbor_files(ConfigOptions, OutputObj.outDate, MpiConfig)

                if forceKey in [12, 21, 27] and ConfigOptions.aws is None:
                    # Calculate the previous and next input cycle files from the inputs.
                    input_forcings.calc_neighbor_files(ConfigOptions, OutputObj.outDate, MpiConfig)
                    err_handler.check_program_status(ConfigOptions, MpiConfig)
                else:
                    # Flag to indicate the AWS .zarr AORC method
                    if forceKey == 12 or forceKey == 21:
                        if ConfigOptions.aws_time is None or ConfigOptions.current_time.year != ConfigOptions.aws_time.year:
                            ConfigOptions.aws_time = ConfigOptions.current_time
                            _s3 = s3fs.S3FileSystem(anon=True)
                            files = [s3fs.S3Map(root=ConfigOptions.aorc_year_url.format(source=ConfigOptions.aorc_source, year=year), s3=_s3,
                                                check=False, ) for year in [ConfigOptions.aws_time.year]]
                            with MPICommExecutor(comm=MpiConfig.comm, root=0) as executor:
                                with dask.config.set(scheduler=executor):
                                    if MpiConfig.rank == 0:
                                        # TODO This is using the wrong call for zarr format - should be a single file, not a list
                                        ConfigOptions.aws_obj = xr.open_mfdataset(files, engine="zarr", parallel=True, consolidated=True)
                            MpiConfig.comm.barrier()
                    # Flag to indicate the AWS .zarr NWMv3 Forcing file method
                    # Which grabs the entire timeseries based on s3 bucket organizations
                    
                    # Added separate processing path for CONUS NWM retrospective data
                    # TODO: Expand functionality for oCONUS domains (different zarr structure)
                    elif forceKey == 27:
                        ConfigOptions.aws_obj = nwm_proc.proc_nwm(ConfigOptions, MpiConfig)

                # If skipping this forcing, continue early
                if input_forcings.skip is True:
                    print(f"Breaking loop for forceKey {forceKey}")
                    break
                # Regrid forcings.
                input_forcings.regrid_inputs(ConfigOptions, wrfHydroGeoMeta, MpiConfig)
                err_handler.check_program_status(ConfigOptions, MpiConfig)

                # Run check on regridded fields for reasonable values that are not missing values.
                err_handler.check_forcing_bounds(ConfigOptions, input_forcings, MpiConfig)
                err_handler.check_program_status(ConfigOptions, MpiConfig)

                # If we are restarting a forecast cycle, re-calculate the neighboring files, and regrid the
                # next set of forcings as the previous step just regridded the previous forcing.
                if input_forcings.rstFlag == 1:
                    if input_forcings.regridded_forcings1 is not None and input_forcings.regridded_forcings2 is not None:
                        # Set the forcings back to reflect we just regridded the previous set of inputs, not the next.
                        if ConfigOptions.grid_type == 'gridded':
                            input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                        elif ConfigOptions.grid_type == 'unstructured':
                            input_forcings.regridded_forcings1[:, :] = input_forcings.regridded_forcings2[:, :]
                            input_forcings.regridded_forcings1_elem[:, :] = input_forcings.regridded_forcings2_elem[:, :]
                        elif ConfigOptions.grid_type == 'hydrofabric':
                            input_forcings.regridded_forcings1[:, :] = input_forcings.regridded_forcings2[:, :]
                    # Re-calculate the neighbor files.
                    input_forcings.calc_neighbor_files(ConfigOptions, OutputObj.outDate, MpiConfig)
                    err_handler.check_program_status(ConfigOptions, MpiConfig)

                    # Regrid the forcings for the end of the window.
                    input_forcings.regrid_inputs(ConfigOptions, wrfHydroGeoMeta, MpiConfig)
                    err_handler.check_program_status(ConfigOptions, MpiConfig)

                    input_forcings.rstFlag = 0

                # Run temporal interpolation on the grids.
                input_forcings.temporal_interpolate_inputs(ConfigOptions, MpiConfig)
                err_handler.check_program_status(ConfigOptions, MpiConfig)

                # Run bias correction.
                bias_correction.run_bias_correction(input_forcings, ConfigOptions, wrfHydroGeoMeta, MpiConfig)
                err_handler.check_program_status(ConfigOptions, MpiConfig)

                # Run downscaling on grids for this output timestep.
                downscale.run_downscaling(input_forcings, ConfigOptions, wrfHydroGeoMeta, MpiConfig)
                err_handler.check_program_status(ConfigOptions, MpiConfig)

                # Layer in forcings from this product.
                layeringMod.layer_final_forcings(OutputObj, input_forcings, ConfigOptions, MpiConfig)
                err_handler.check_program_status(ConfigOptions, MpiConfig)

                ConfigOptions.currentForceNum += 1

                if forceKey == 10:
                    ConfigOptions.currentCustomForceNum += 1

                print(f'End of loop for forceKey {forceKey}\n')

            # Process supplemental precipitation if we specified in the configuration file.
            if ConfigOptions.number_supp_pcp > 0:
                for suppPcpKey in ConfigOptions.supp_precip_forcings:
                    if suppPcpKey != 13:
                        # Like with input forcings, calculate the neighboring files to use.
                        suppPcpMod[suppPcpKey].calc_neighbor_files(ConfigOptions, OutputObj.outDate, MpiConfig)
                        err_handler.check_program_status(ConfigOptions, MpiConfig)

                        # Regrid the supplemental precipitation.
                        suppPcpMod[suppPcpKey].regrid_inputs(ConfigOptions, wrfHydroGeoMeta, MpiConfig)
                        err_handler.check_program_status(ConfigOptions, MpiConfig)

                        if suppPcpMod[suppPcpKey].regridded_precip1 is not None \
                                and suppPcpMod[suppPcpKey].regridded_precip2 is not None:
                            # Run check on regridded fields for reasonable values that are not missing values.
                            err_handler.check_supp_pcp_bounds(ConfigOptions, suppPcpMod[suppPcpKey], MpiConfig, wrfHydroGeoMeta)
                            err_handler.check_program_status(ConfigOptions, MpiConfig)

                            # TODO input_forcings has not yet been initialized, so this is a bug waiting to happen
                            disaggregate_fun(input_forcings, suppPcpMod[suppPcpKey], ConfigOptions, MpiConfig)
                            err_handler.check_program_status(ConfigOptions, MpiConfig)

                            # Run temporal interpolation on the grids.
                            suppPcpMod[suppPcpKey].temporal_interpolate_inputs(ConfigOptions, MpiConfig)
                            err_handler.check_program_status(ConfigOptions, MpiConfig)

                            # Layer in the supplemental precipitation into the current output object.
                            layeringMod.layer_supplemental_forcing(OutputObj, suppPcpMod[suppPcpKey],
                                                                   ConfigOptions, MpiConfig)
                            err_handler.check_program_status(ConfigOptions, MpiConfig)

            # Call the output routines
            #   adjust date for AnA if necessary
            if ConfigOptions.ana_flag:
                OutputObj.outDate = file_date

                ################ Commenting this out to bypass NWM forcing file output functionality #########
                # OutputObj.output_final_ldasin(ConfigOptions, wrfHydroGeoMeta, MpiConfig)
                # err_handler.check_program_status(ConfigOptions, MpiConfig)
                ##############################################################################################

        if ConfigOptions.customSuppPcpFreq is not None:
            # Process supplemental precipitation if we specified in the configuration file.
            if ConfigOptions.number_supp_pcp > 0:
                for suppPcpKey in ConfigOptions.supp_precip_forcings:
                    if suppPcpKey == 14:
                        # Like with input forcings, calculate the neighboring files to use.
                        suppPcpMod[suppPcpKey].calc_neighbor_files(ConfigOptions, OutputObj.outDate, MpiConfig)
                        err_handler.check_program_status(ConfigOptions, MpiConfig)

                        # Regrid the supplemental precipitation.
                        suppPcpMod[suppPcpKey].regrid_inputs(ConfigOptions, wrfHydroGeoMeta, MpiConfig)
                        err_handler.check_program_status(ConfigOptions, MpiConfig)

                        if suppPcpMod[suppPcpKey].regridded_precip1 is not None \
                                and suppPcpMod[suppPcpKey].regridded_precip2 is not None:
                            # Run check on regridded fields for reasonable values that are not missing values.
                            err_handler.check_supp_pcp_bounds(ConfigOptions, suppPcpMod[suppPcpKey], MpiConfig, wrfHydroGeoMeta)
                            err_handler.check_program_status(ConfigOptions, MpiConfig)

                            disaggregate_fun(input_forcings, suppPcpMod[suppPcpKey], ConfigOptions, MpiConfig)
                            err_handler.check_program_status(ConfigOptions, MpiConfig)

                            # Run temporal interpolation on the grids.
                            suppPcpMod[suppPcpKey].temporal_interpolate_inputs(ConfigOptions, MpiConfig)
                            err_handler.check_program_status(ConfigOptions, MpiConfig)

                            # Layer in the supplemental precipitation into the current output object.
                            layeringMod.layer_supplemental_forcing(OutputObj, suppPcpMod[suppPcpKey],
                                                                   ConfigOptions, MpiConfig)
                            err_handler.check_program_status(ConfigOptions, MpiConfig)

        # Now loop through Forcings Engine output object 
        # and flatten the 2D forcing array and append to 
        # the BMI object to advertise to BMIinterface
        # 0.) U-Wind (m/s)
        # 1.) V-Wind (m/s)
        # 2.) Surface incoming longwave radiation flux (W/m^2)
        # 3.) Precipitation rate (mm/s)
        # 4.) 2-meter temperature (K)
        # 5.) 2-meter specific humidity (kg/kg)
        # 6.) Surface pressure (Pa)
        # 7.) Surface incoming shortwave radiation flux (W/m^2)
        # 8.) Liquid Precipitation Fraction (%), Only available in certain operational configurations
        if ConfigOptions.include_lqfrac == 1:
            variables = ['U2D', 'V2D', 'LWDOWN', 'RAINRATE', 'T2D', 'Q2D', 'PSFC', 'SWDOWN', 'LQFRAC']
        else:
            variables = ['U2D', 'V2D', 'LWDOWN', 'RAINRATE', 'T2D', 'Q2D', 'PSFC', 'SWDOWN']

        # If user requests output for given domain, then call
        # the I/O module to update opened netcdf file with forcing fields
        if ConfigOptions.forcing_output == 1:
            OutputObj.update_forcing_file_output(ConfigOptions, wrfHydroGeoMeta, MpiConfig)
            print(f"[INFO] Writing output forcing file for timestamp: {OutputObj.outDate.strftime('%Y-%m-%d %H:%M')}")

        if ConfigOptions.grid_type == "gridded":
            for count, variable in enumerate(variables):
                model[variable + '_ELEMENT'] = OutputObj.output_local[count, :, :].flatten()
        elif ConfigOptions.grid_type == "unstructured":
            for count, variable in enumerate(variables):
                model[variable + '_ELEMENT'] = OutputObj.output_local_elem[count, :].flatten()
                model[variable + '_NODE'] = OutputObj.output_local[count, :].flatten()
        elif ConfigOptions.grid_type == "hydrofabric":
            for count, variable in enumerate(variables):
                model[variable + '_ELEMENT'] = OutputObj.output_local[count, :].flatten()

        ## Update BMI model time index to next iteration
        ConfigOptions.bmi_time_index += 1
