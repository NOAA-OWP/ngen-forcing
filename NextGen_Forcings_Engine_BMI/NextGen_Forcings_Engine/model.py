import datetime
import os
from contextlib import contextmanager
from time import time

import ewts
import numpy as np
import pandas as pd

from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core import (
    bias_correction,
    disaggregateMod,
    downscale,
    err_handler,
    layeringMod,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.config import (
    ConfigOptions,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.geoMod import (
    GeoMeta,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.ioMod import OutputObj
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.parallel import MpiConfig
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.historical_forcing import (
    AORCAlaskaProcessor,
    AORCConusProcessor,
    NWMV3AlaskaProcessor,
    NWMV3ConusProcessor,
    NWMV3OConusProcessor,
)

LOG = ewts.get_logger(ewts.FORCING_ID)


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


def time_function(func):
    """Measure the execution time of a function."""

    def wrapper(*args, **kwargs):
        with timing_block(f"Executing {func.__name__}"):
            result = func(*args, **kwargs)
            return result

    return wrapper


class NWMv3ForcingEngineModel:
    """NextGen Forcings Engine BMI model class for NWMv3 forcings."""

    def __init__(self):
        """Initialize the NWMv3 Forcing Engine Model."""
        self.source_data_processor = None

    # TODO: refactor the bmi_model.py file and this to have this type maintain its own state.
    # def __init__(self):
    #    super(ngen_model, self).__init__()
    #    #self._model = model

    # @dask.delayed
    # def aws_obj(files):
    #    return xr.open_mfdataset(files, engine="zarr", parallel=True, consolidated=True)

    def run(
        self,
        model: dict,
        future_time: float,
        config_options: ConfigOptions,
        wrf_hydro_geo_meta: GeoMeta,
        input_forcing_mod: dict,
        supp_pcp_mod: dict,
        mpi_config: MpiConfig,
        output_obj: OutputObj,
    ) -> None:
        """Execute the full forcings engine BMI pipeline for a given future timestep.

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
        :param config_options: Configuration object containing all model options, flags, and paths.
        :param wrf_hydro_geo_meta: Geospatial metadata needed for regridding and interpolation.
        :param input_forcing_mod: Dictionary of initialized input forcing modules indexed by forcing key.
        :param supp_pcp_mod: Dictionary of supplemental precipitation modules indexed by key.
        :param mpi_config: Object containing MPI communication settings such as rank and communicator.
        :param output_obj: Output object that stores the generated atmospheric forcing arrays.

        :raises RuntimeError: If the model fails to initialize or if required arguments are missing.
        """
        (
            future_time,
            config_options,
        ) = self.determine_forecast(
            future_time,
            config_options,
        )
        (
            config_options,
            input_forcing_mod,
            mpi_config,
        ) = self.adjust_precip(
            config_options,
            input_forcing_mod,
            mpi_config,
        )
        (
            config_options,
            mpi_config,
        ) = self.log_forecast(
            config_options,
            mpi_config,
        )
        (
            future_time,
            config_options,
            wrf_hydro_geo_meta,
            input_forcing_mod,
            supp_pcp_mod,
            mpi_config,
            output_obj,
            input_forcings,
        ) = self.loop_through_forcing_products(
            future_time,
            config_options,
            wrf_hydro_geo_meta,
            input_forcing_mod,
            supp_pcp_mod,
            mpi_config,
            output_obj,
        )
        (
            config_options,
            wrf_hydro_geo_meta,
            supp_pcp_mod,
            mpi_config,
            output_obj,
        ) = self.process_suplemental_precip(
            config_options,
            wrf_hydro_geo_meta,
            supp_pcp_mod,
            mpi_config,
            output_obj,
            input_forcings,
        )
        (
            config_options,
            wrf_hydro_geo_meta,
            mpi_config,
            output_obj,
        ) = self.write_output(
            config_options,
            wrf_hydro_geo_meta,
            mpi_config,
            output_obj,
        )
        (
            model,
            config_options,
            wrf_hydro_geo_meta,
            output_obj,
        ) = self.update_dict(
            model,
            config_options,
            wrf_hydro_geo_meta,
            output_obj,
        )

        ## Update BMI model time index to next iteration
        config_options.bmi_time_index += 1

    @time_function
    def determine_forecast(
        self,
        future_time: float,
        config_options: ConfigOptions,
    ):
        """Determine the forecast for the given future time and configuration."""
        # Assign the future time to the configuration
        config_options.bmi_time = future_time
        self.disaggregate_fun = disaggregateMod.disaggregate_factory(config_options)

        # Calculate current time stamp based on operational configuration
        if config_options.ana_flag:
            # If we're in an AnA configuration, then must offset the BMI future
            # timestamp to account for the "lookback" period being properly iterated
            # over between 3-28 hour look back time period and operation configuration
            if config_options.input_forcings[0] in [20, 22]:
                config_options.current_fcst_cycle = (
                    config_options.b_date_proc
                    + pd.TimedeltaIndex(
                        np.array([future_time - 7200.0], dtype=float), "s"
                    )[0]
                )
                config_options.current_time = (
                    config_options.b_date_proc
                    + pd.TimedeltaIndex(
                        np.array([future_time - 7200.0], dtype=float), "s"
                    )[0]
                )
                config_options.future_time = future_time
            else:
                # Puerto Rico / Hawaii AnA: 1-hour lookback (based on 6-hourly forecast cycles)
                config_options.current_fcst_cycle = (
                    config_options.b_date_proc
                    + pd.TimedeltaIndex(
                        np.array([future_time - 3600.0], dtype=float), "s"
                    )[0]
                )
                config_options.current_time = (
                    config_options.b_date_proc
                    + pd.TimedeltaIndex(
                        np.array([future_time - 3600.0], dtype=float), "s"
                    )[0]
                )
        else:
            # Forecast-only mode — use BMI timestamp as-is
            config_options.current_fcst_cycle = config_options.b_date_proc
            config_options.current_time = pd.Timestamp(
                config_options.b_date_proc
            ) + pd.to_timedelta(future_time, unit="s")

        LOG.debug(
            "NextGen Forcings Engine processing meteorological forcings for BMI timestamp"
        )
        LOG.debug(f"Model.py current time: {config_options.current_time}")
        LOG.debug(f"Model.py current fcst cycle: {config_options.current_fcst_cycle}")

        if config_options.first_fcst_cycle is None:
            config_options.first_fcst_cycle = config_options.current_fcst_cycle

        return (
            future_time,
            config_options,
        )

    @time_function
    def adjust_precip(
        self,
        config_options: ConfigOptions,
        input_forcing_mod: dict,
        mpi_config: MpiConfig,
    ):
        """Adjust precipitation for the given forecast cycle."""
        if not config_options.precip_only_flag:
            # reset skips if present
            for force_key in config_options.input_forcings:
                input_forcing_mod[force_key].skip = False

            err_handler.check_program_status(config_options, mpi_config)
        return (
            config_options,
            input_forcing_mod,
            mpi_config,
        )

    @time_function
    def log_forecast(
        self,
        config_options: ConfigOptions,
        mpi_config: MpiConfig,
    ):
        """Log information about the current forecast cycle."""
        # Log information about this forecast cycle
        if mpi_config.rank == 0:
            config_options.statusMsg = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
            err_handler.log_msg(config_options, mpi_config, True)
            config_options.statusMsg = (
                "Processing Forecast Cycle: "
                + config_options.current_fcst_cycle.strftime("%Y-%m-%d %H:%M")
            )
            err_handler.log_msg(config_options, mpi_config, True)
            config_options.statusMsg = (
                "Forecast Cycle Length is: "
                + str(config_options.cycle_length_minutes)
                + " minutes"
            )
            err_handler.log_msg(config_options, mpi_config, True)
        # mpi_config.comm.barrier()

        return (
            config_options,
            mpi_config,
        )

    @time_function
    def loop_through_forcing_products(
        self,
        future_time: float,
        config_options: ConfigOptions,
        wrf_hydro_geo_meta: GeoMeta,
        input_forcing_mod: dict,
        supp_pcp_mod: dict,
        mpi_config: MpiConfig,
        output_obj: OutputObj,
    ):
        """Loop through each forcing product and process it for the current forecast cycle."""
        # Loop through each output timestep. Perform the following functions:
        # 1.) Calculate all necessary input files per user options.
        # 2.) Read in input forcings from GRIB/NetCDF files.
        # 3.) Regrid the forcings, and temporally interpolate.
        # 4.) Downscale.
        # 5.) Layer, and output as necessary.
        ana_factor = 1 if config_options.ana_flag is False else 0
        show_message = True
        if not config_options.precip_only_flag:
            if config_options.grid_type == "gridded":
                # Reset out final grids to missing values.
                output_obj.output_local[:, :, :] = config_options.globalNdv
            elif config_options.grid_type == "unstructured":
                # Reset out final grids to missing values.
                output_obj.output_local[:, :] = config_options.globalNdv
                output_obj.output_local_elem[:, :] = config_options.globalNdv
            elif config_options.grid_type == "hydrofabric":
                # Reset out final grids to missing values.
                output_obj.output_local[:, :] = config_options.globalNdv

            # Increment or initialize output step count
            if config_options.current_output_step is None:
                config_options.current_output_step = 1
            else:
                config_options.current_output_step += 1

            # Optional sub-output timestamp
            if config_options.sub_output_hour is not None:
                # TODO This is not used
                subOutDate = config_options.first_fcst_cycle + datetime.timedelta(
                    hours=config_options.sub_output_hour
                )

            # Compute the output timestamp for this step
            if config_options.ana_flag:
                output_obj.outDate = (
                    config_options.current_fcst_cycle
                    + datetime.timedelta(seconds=config_options.output_freq * 60)
                )
            else:
                output_obj.outDate = (
                    config_options.current_fcst_cycle
                    + datetime.timedelta(seconds=future_time)
                )

            config_options.current_output_date = output_obj.outDate

            # Adjust file_date for AnA if needed
            file_date = (
                output_obj.outDate
                - datetime.timedelta(seconds=config_options.output_freq * 60)
                if config_options.ana_flag
                else output_obj.outDate
            )

            # Compute previous output date (used for downscaling logic)
            if config_options.current_output_step == ana_factor:
                config_options.prev_output_date = config_options.current_output_date
            else:
                config_options.prev_output_date = (
                    config_options.current_output_date
                    - datetime.timedelta(seconds=future_time)
                )

            # Print message on log file indicating the timestamp
            # we are currently processing for forcings
            if mpi_config.rank == 0 and show_message:
                config_options.statusMsg = "========================================="
                err_handler.log_msg(config_options, mpi_config, True)
                config_options.statusMsg = f"Processing for output timestep: {file_date.strftime('%Y-%m-%d %H:%M')}"
                err_handler.log_msg(config_options, mpi_config, True)

            config_options.currentForceNum = 0
            config_options.currentCustomForceNum = 0
            LOG.debug(f"config_options.input_forcings: {config_options.input_forcings}")
            # Loop over each of the input forcings specified.
            LOG.debug(
                f"Model.py forcing loop: {len(config_options.input_forcings)} forcings configured: {config_options.input_forcings}"
            )

            for force_key in config_options.input_forcings:
                LOG.debug(f"force_key: {force_key}")
                LOG.debug(f"config_options.aws: {config_options.aws}")
                # Pass these methods for AORC data is ERA5-Interim blend is requested
                # so we can finish filling in the missing gaps
                if (
                    force_key == 23
                    and 12 in config_options.input_forcings
                    and 21 in config_options.input_forcings
                ):
                    input_forcings = input_forcing_mod[force_key]

                    # These are not used
                    # AORC_mask = input_forcings.regridded_mask_AORC
                    # AORC_elem_mask = input_forcings.regridded_mask_elem_AORC
                else:
                    input_forcings = input_forcing_mod[force_key]
                    input_forcings.calc_neighbor_files(
                        config_options, output_obj.outDate, mpi_config
                    )

                if force_key in [12, 21, 27]:
                    if config_options.aws is None:
                        # Calculate the previous and next input cycle files from the inputs.
                        input_forcings.calc_neighbor_files(
                            config_options, output_obj.outDate, mpi_config
                        )
                        err_handler.check_program_status(config_options, mpi_config)
                    else:
                        # Flag to indicate the AWS .zarr AORC method
                        if force_key == 12:
                            if self.source_data_processor is None:
                                self.source_data_processor = AORCConusProcessor(
                                    config_options, mpi_config, wrf_hydro_geo_meta
                                )
                        elif force_key == 21:
                            if self.source_data_processor is None:
                                self.source_data_processor = AORCAlaskaProcessor(
                                    config_options, mpi_config, wrf_hydro_geo_meta
                                )

                        # Flag to indicate the AWS .zarr NWMv3 Forcing file method
                        elif force_key == 27:
                            if self.source_data_processor is None:
                                if config_options.nwm_domain == "CONUS":
                                    self.source_data_processor = NWMV3ConusProcessor(
                                        config_options, mpi_config, wrf_hydro_geo_meta
                                    )
                                elif config_options.nwm_domain in [
                                    "Hawaii",
                                    "PR",
                                ]:
                                    self.source_data_processor = NWMV3OConusProcessor(
                                        config_options, mpi_config, wrf_hydro_geo_meta
                                    )
                                elif config_options.nwm_domain == "Alaska":
                                    self.source_data_processor = NWMV3AlaskaProcessor(
                                        config_options, mpi_config, wrf_hydro_geo_meta
                                    )
                                else:
                                    raise ValueError(
                                        f"Unsupported domain type ({config_options.nwm_domain} for forcing type: {force_key} )"
                                    )

                        config_options.aws_obj = (
                            self.source_data_processor.process_historical_data(
                                config_options.current_time
                            )
                        )

                # If skipping this forcing, continue early
                if input_forcings.skip is True:
                    LOG.debug(f"Breaking loop for force_key {force_key}")
                    break
                # Regrid forcings.
                input_forcings.regrid_inputs(
                    config_options, wrf_hydro_geo_meta, mpi_config
                )
                err_handler.check_program_status(config_options, mpi_config)

                # Run check on regridded fields for reasonable values that are not missing values.
                err_handler.check_forcing_bounds(
                    config_options, input_forcings, mpi_config
                )
                err_handler.check_program_status(config_options, mpi_config)

                # If we are restarting a forecast cycle, re-calculate the neighboring files, and regrid the
                # next set of forcings as the previous step just regridded the previous forcing.
                if input_forcings.rstFlag == 1:
                    if (
                        input_forcings.regridded_forcings1 is not None
                        and input_forcings.regridded_forcings2 is not None
                    ):
                        # Set the forcings back to reflect we just regridded the previous set of inputs, not the next.
                        if config_options.grid_type == "gridded":
                            input_forcings.regridded_forcings1[:, :, :] = (
                                input_forcings.regridded_forcings2[:, :, :]
                            )
                        elif config_options.grid_type == "unstructured":
                            input_forcings.regridded_forcings1[:, :] = (
                                input_forcings.regridded_forcings2[:, :]
                            )
                            input_forcings.regridded_forcings1_elem[:, :] = (
                                input_forcings.regridded_forcings2_elem[:, :]
                            )
                        elif config_options.grid_type == "hydrofabric":
                            input_forcings.regridded_forcings1[:, :] = (
                                input_forcings.regridded_forcings2[:, :]
                            )
                    # Re-calculate the neighbor files.
                    input_forcings.calc_neighbor_files(
                        config_options, output_obj.outDate, mpi_config
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    # Regrid the forcings for the end of the window.
                    input_forcings.regrid_inputs(
                        config_options, wrf_hydro_geo_meta, mpi_config
                    )
                    err_handler.check_program_status(config_options, mpi_config)

                    input_forcings.rstFlag = 0

                # Run temporal interpolation on the grids.
                input_forcings.temporal_interpolate_inputs(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                # Run bias correction.
                bias_correction.run_bias_correction(
                    input_forcings, config_options, wrf_hydro_geo_meta, mpi_config
                )
                err_handler.check_program_status(config_options, mpi_config)

                # Run downscaling on grids for this output timestep.
                downscale.run_downscaling(
                    input_forcings, config_options, wrf_hydro_geo_meta, mpi_config
                )
                err_handler.check_program_status(config_options, mpi_config)

                # Layer in forcings from this product.
                layeringMod.layer_final_forcings(
                    output_obj, input_forcings, config_options, mpi_config
                )
                err_handler.check_program_status(config_options, mpi_config)

                config_options.currentForceNum += 1

                if force_key == 10:
                    config_options.currentCustomForceNum += 1

                LOG.debug(f"End of loop for force_key {force_key}")

            # Process supplemental precipitation if we specified in the configuration file.
            if config_options.number_supp_pcp > 0:
                for supp_pcp_key in config_options.supp_precip_forcings:
                    if supp_pcp_key != 13:
                        # Like with input forcings, calculate the neighboring files to use.
                        supp_pcp_mod[supp_pcp_key].calc_neighbor_files(
                            config_options, output_obj.outDate, mpi_config
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        # Regrid the supplemental precipitation.
                        supp_pcp_mod[supp_pcp_key].regrid_inputs(
                            config_options, wrf_hydro_geo_meta, mpi_config
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        if (
                            supp_pcp_mod[supp_pcp_key].regridded_precip1 is not None
                            and supp_pcp_mod[supp_pcp_key].regridded_precip2 is not None
                        ):
                            # Run check on regridded fields for reasonable values that are not missing values.
                            err_handler.check_supp_pcp_bounds(
                                config_options,
                                supp_pcp_mod[supp_pcp_key],
                                mpi_config,
                                wrf_hydro_geo_meta,
                            )
                            err_handler.check_program_status(config_options, mpi_config)

                            # TODO input_forcings has not yet been initialized, so this is a bug waiting to happen
                            self.disaggregate_fun(
                                input_forcings,
                                supp_pcp_mod[supp_pcp_key],
                                config_options,
                                mpi_config,
                            )
                            err_handler.check_program_status(config_options, mpi_config)

                            # Run temporal interpolation on the grids.
                            supp_pcp_mod[supp_pcp_key].temporal_interpolate_inputs(
                                config_options, mpi_config
                            )
                            err_handler.check_program_status(config_options, mpi_config)

                            # Layer in the supplemental precipitation into the current output object.
                            layeringMod.layer_supplemental_forcing(
                                output_obj,
                                supp_pcp_mod[supp_pcp_key],
                                config_options,
                                mpi_config,
                            )
                            err_handler.check_program_status(config_options, mpi_config)

            # Call the output routines
            #   adjust date for AnA if necessary
            if config_options.ana_flag:
                output_obj.outDate = file_date

                ################ Commenting this out to bypass NWM forcing file output functionality #########
                # output_obj.output_final_ldasin(config_options, wrf_hydro_geo_meta, mpi_config)
                # err_handler.check_program_status(config_options, mpi_config)
                ##############################################################################################

        return (
            future_time,
            config_options,
            wrf_hydro_geo_meta,
            input_forcing_mod,
            supp_pcp_mod,
            mpi_config,
            output_obj,
            input_forcings,
        )

    @time_function
    def process_suplemental_precip(
        self,
        config_options: ConfigOptions,
        wrf_hydro_geo_meta: GeoMeta,
        supp_pcp_mod: dict,
        mpi_config: MpiConfig,
        output_obj: OutputObj,
        input_forcings: dict,
    ):
        """Process supplemental precipitation for the current forecast cycle."""
        if config_options.customSuppPcpFreq is not None:
            # Process supplemental precipitation if we specified in the configuration file.
            if config_options.number_supp_pcp > 0:
                for supp_pcp_key in config_options.supp_precip_forcings:
                    if supp_pcp_key == 14:
                        # Like with input forcings, calculate the neighboring files to use.
                        supp_pcp_mod[supp_pcp_key].calc_neighbor_files(
                            config_options, output_obj.outDate, mpi_config
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        # Regrid the supplemental precipitation.
                        supp_pcp_mod[supp_pcp_key].regrid_inputs(
                            config_options, wrf_hydro_geo_meta, mpi_config
                        )
                        err_handler.check_program_status(config_options, mpi_config)

                        if (
                            supp_pcp_mod[supp_pcp_key].regridded_precip1 is not None
                            and supp_pcp_mod[supp_pcp_key].regridded_precip2 is not None
                        ):
                            # Run check on regridded fields for reasonable values that are not missing values.
                            err_handler.check_supp_pcp_bounds(
                                config_options,
                                supp_pcp_mod[supp_pcp_key],
                                mpi_config,
                                wrf_hydro_geo_meta,
                            )
                            err_handler.check_program_status(config_options, mpi_config)

                            self.disaggregate_fun(
                                input_forcings,
                                supp_pcp_mod[supp_pcp_key],
                                config_options,
                                mpi_config,
                            )
                            err_handler.check_program_status(config_options, mpi_config)

                            # Run temporal interpolation on the grids.
                            supp_pcp_mod[supp_pcp_key].temporal_interpolate_inputs(
                                config_options, mpi_config
                            )
                            err_handler.check_program_status(config_options, mpi_config)

                            # Layer in the supplemental precipitation into the current output object.
                            layeringMod.layer_supplemental_forcing(
                                output_obj,
                                supp_pcp_mod[supp_pcp_key],
                                config_options,
                                mpi_config,
                            )
                            err_handler.check_program_status(config_options, mpi_config)

        return (
            config_options,
            wrf_hydro_geo_meta,
            supp_pcp_mod,
            mpi_config,
            output_obj,
        )

    @time_function
    def write_output(
        self,
        config_options: ConfigOptions,
        wrf_hydro_geo_meta: GeoMeta,
        mpi_config: MpiConfig,
        output_obj: OutputObj,
    ):
        """Write the output for the current forecast cycle."""
        # If user requests output for given domain, then call
        # the I/O module to update opened netcdf file with forcing fields
        if (
            config_options.forcing_output == 1
            or config_options.grid_type == "hydrofabric"
        ):
            output_obj.gather_global_outputs(
                config_options, wrf_hydro_geo_meta, mpi_config
            )
        return (
            config_options,
            wrf_hydro_geo_meta,
            mpi_config,
            output_obj,
        )

        """##################Step 6: flatten and update dict##########################################################################"""

    @time_function
    def update_dict(
        self,
        model: dict,
        config_options: ConfigOptions,
        wrf_hydro_geo_meta: GeoMeta,
        output_obj: OutputObj,
    ):
        """Flatten the Forcings Engine output object and update the BMI dictionary."""
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

        if config_options.include_lqfrac == 1:
            variables = [
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
        else:
            variables = [
                "U2D",
                "V2D",
                "LWDOWN",
                "RAINRATE",
                "T2D",
                "Q2D",
                "PSFC",
                "SWDOWN",
            ]
        if config_options.grid_type == "gridded":
            for count, variable in enumerate(variables):
                model[variable + "_ELEMENT"] = output_obj.output_local[
                    count, :, :
                ].flatten()
        elif config_options.grid_type == "unstructured":
            for count, variable in enumerate(variables):
                model[variable + "_ELEMENT"] = output_obj.output_local_elem[
                    count, :
                ].flatten()
                model[variable + "_NODE"] = output_obj.output_local[count, :].flatten()
        elif config_options.grid_type == "hydrofabric":
            for count, variable in enumerate(variables):
                model[variable + "_ELEMENT"] = output_obj.output_global[
                    count, :
                ].flatten()
                model["CAT-ID"] = wrf_hydro_geo_meta.element_ids_global

        return (
            model,
            config_options,
            wrf_hydro_geo_meta,
            output_obj,
        )
