"""NWMv3ForcingEngineModel, to be constructed and managed by inheritors of NWMv3_Forcing_Engine_BMI_model_Base from bmi_model.py"""

from __future__ import annotations

import copy
import datetime
import logging
from contextlib import contextmanager
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ewts.modules import ModuleKey

from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core import (
    bias_correction,
    disaggregateMod,
    downscale,
    err_handler,
    forcingInputMod,
    layeringMod,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.consts import (
    MODEL as model_consts,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.historical_forcing import (
    AORCAlaskaProcessor,
    AORCConusProcessor,
    NWMV3AlaskaProcessor,
    NWMV3ConusProcessor,
    NWMV3HawaiiProcessor,
    NWMV3PuertoRicoProcessor,
)

if TYPE_CHECKING:
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.bmi_model import (
        NWMv3_Forcing_Engine_BMI_model_Base,
    )

LOG = logging.getLogger("FORCING")

MODNM = ModuleKey.FORCING.value


@contextmanager
def timing_block(step_str: str):
    """Context manager for timing code execution. Used by the decorator ``time_function``.

    :param str step_str: Description of the step being timed.
    """
    start = perf_counter()
    yield
    end = perf_counter()
    LOG.debug(msg=f"  Execution time for {step_str}: {round(end - start, 2)} seconds")


def time_function(func):
    """Decorator for measuring the execution time of a function."""

    def wrapper(*args, **kwargs):
        with timing_block(f"Executing {func.__name__}"):
            result = func(*args, **kwargs)
            return result

    return wrapper


class NWMv3ForcingEngineModel:
    """NextGen Forcings Engine BMI model class for NWMv3 forcings.

    To be constructed and managed by inheritors of NWMv3_Forcing_Engine_BMI_model_Base from bmi_model.py.
    """

    def __init__(self, bmi_model: NWMv3_Forcing_Engine_BMI_model_Base):
        """Initialize the NWMv3 Forcing Engine model.

        :param bmi_model NWMv3_Forcing_Engine_BMI_model_Base: BMI model instance to initialize.
        """
        self.source_data_processor = None
        self._bmi = bmi_model
        # Partials
        self.log_info = partial(
            err_handler.log_msg, self._bmi._job_meta, self._bmi._mpi_meta, False
        )
        self.log_debug = partial(
            err_handler.log_msg, self._bmi._job_meta, self._bmi._mpi_meta, True
        )

    def check_program_status(self) -> None:
        """Call err_handler.check_program_status"""
        err_handler.check_program_status(self._bmi._job_meta, self._bmi._mpi_meta)

    def run(self, future_time: float) -> None:
        """Execute the full forcings engine BMI pipeline for a given future timestep.

        This method updates the ``self._bmi._values`` state dictionary with atmospheric
        forcings computed from available input datasets. It handles initialization,
        AWS Zarr loading, regridding, temporal interpolation, bias correction,
        downscaling, supplemental precipitation processing, and output population into
        the ``self._bmi._values`` structure.

        ``self._bmi._job_meta``, an instance of ``ConfigOptions``, is also updated
        in-place, for example for forecast time handling.

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
        6. Update the ``self._bmi._values`` state dictionary with flattened arrays.
        7. Advance the BMI time index.

        :param float future_time: Timestamp, represented as *seconds relative to overall
            start time*, to advance to before returning. Since this value is relative
            to the overall start time, it is unaware of the actual UTC datetimestamp of
            the start. For example, since 1-hour timesteps are typical, the first value
            would typically be 3600, the second value 7200, etc.

        :raises RuntimeError: If the model fails to initialize or if required arguments
            are missing.
        """

        self.set_cycle_timing_attrs(future_time)
        self.set_skip_flags()
        self.log_cycle()
        input_forcings = self.loop_through_forcing_products(future_time)
        self.process_suplemental_precip(input_forcings)
        self.write_output()
        self.update_bmi_output_dict()

        ## Update BMI model time index to next iteration
        self._bmi._job_meta.bmi_time_index += 1

    @time_function
    def set_cycle_timing_attrs(self, future_time: float) -> None:
        """Determine the forecast for the given future time and configuration.

        :warning: Modifies mutable arguments in-place
        """
        # Assign the future time to the configuration
        self._bmi._job_meta.bmi_time = future_time
        self.disaggregate_fun = disaggregateMod.disaggregate_factory(
            self._bmi._job_meta
        )

        # Calculate current time stamp based on operational configuration
        if self._bmi._job_meta.ana_flag:
            # If we're in an AnA configuration, then must offset the BMI future
            # timestamp to account for the "lookback" period being properly iterated
            # over between 3-28 hour look back time period and operation configuration
            # TODO confirm these codes, and should they consider all input_forcings not just [0]?
            if self._bmi._job_meta.input_forcings[0] in [20, 22]:
                # NOTE This appears to be intending to operate on Alaska-only AnA.
                delta = pd.TimedeltaIndex(
                    np.array([future_time - 7200.0], dtype=float), "s"
                )[0]
                self._bmi._job_meta.current_fcst_cycle = (
                    self._bmi._job_meta.b_date_proc + delta
                )
                self._bmi._job_meta.current_time = (
                    self._bmi._job_meta.b_date_proc + delta
                )
                self._bmi._job_meta.future_time = future_time
            else:
                # NOTE below comment was original, but this appears to be operating on all non-Alaska AnA, not just Puerto Rico / Hawaii AnA.
                # Puerto Rico / Hawaii AnA: 1-hour lookback (based on 6-hourly forecast cycles)
                delta = pd.TimedeltaIndex(
                    np.array([future_time - 3600.0], dtype=float), "s"
                )[0]
                self._bmi._job_meta.current_fcst_cycle = (
                    self._bmi._job_meta.b_date_proc + delta
                )
                self._bmi._job_meta.current_time = (
                    self._bmi._job_meta.b_date_proc + delta
                )
        else:
            # Forecast-only mode — use BMI timestamp as-is
            self._bmi._job_meta.current_fcst_cycle = self._bmi._job_meta.b_date_proc
            self._bmi._job_meta.current_time = pd.Timestamp(
                self._bmi._job_meta.b_date_proc
            ) + pd.to_timedelta(future_time, unit="s")

        self.log_debug(
            msg="NextGen Forcings Engine processing meteorological forcings for BMI timestamp"
        )
        self.log_debug(msg=f"Model.py current time: {self._bmi._job_meta.current_time}")
        self.log_debug(
            msg=f"Model.py current fcst cycle: {self._bmi._job_meta.current_fcst_cycle}"
        )

        if self._bmi._job_meta.first_fcst_cycle is None:
            self._bmi._job_meta.first_fcst_cycle = (
                self._bmi._job_meta.current_fcst_cycle
            )

    @time_function
    def set_skip_flags(self) -> None:
        """Adjust precipitation for the given forecast cycle."""
        if not self._bmi._job_meta.precip_only_flag:
            # reset skips if present
            for force_key in self._bmi._job_meta.input_forcings:
                self._bmi._input_forcing_mod[force_key].skip = False

            self.check_program_status()

    @time_function
    def log_cycle(self) -> None:
        """Log information about the current forecast cycle."""
        if self._bmi._mpi_meta.rank == 0:
            self.log_debug(msg="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            self.log_debug(
                msg=f"Processing Forecast Cycle: {self._bmi._job_meta.current_fcst_cycle.strftime('%Y-%m-%d %H:%M')}"
            )
            self.log_debug(
                msg=f"Forecast Cycle Length is: {self._bmi._job_meta.cycle_length_minutes!s} minutes"
            )
        # self._bmi._mpi_meta.comm.barrier()

    @time_function
    def loop_through_forcing_products(
        self, future_time: float
    ) -> forcingInputMod.InputForcingsHydrofabric | None:
        """Loop through each forcing product and process it for the current forecast cycle.

        Loop through each output timestep and perform the following steps:

        1. Calculate all necessary input files per user options.
        2. Read input forcings from GRIB/NetCDF files.
        3. Regrid the forcings and perform temporal interpolation.
        4. Downscale.
        5. Layer and write output as necessary.

        :param float future_time: See description in ``self.run``.
        :returns: Processed input forcings for the current timestep.
        :rtype: forcingInputMod.InputForcings | None
        """
        ana_factor = 1 if self._bmi._job_meta.ana_flag is False else 0
        if not self._bmi._job_meta.precip_only_flag:
            if self._bmi._job_meta.grid_type == "gridded":
                # Reset out final grids to missing values.
                self._bmi._output_obj.output_local[:, :, :] = (
                    self._bmi._job_meta.globalNdv
                )
            elif self._bmi._job_meta.grid_type == "unstructured":
                # Reset out final grids to missing values.
                self._bmi._output_obj.output_local[:, :] = self._bmi._job_meta.globalNdv
                self._bmi._output_obj.output_local_elem[:, :] = (
                    self._bmi._job_meta.globalNdv
                )
            elif self._bmi._job_meta.grid_type == "hydrofabric":
                # Reset out final grids to missing values.
                self._bmi._output_obj.output_local[:, :] = self._bmi._job_meta.globalNdv
            else:
                raise ValueError(
                    f"Unexpected grid_type: {repr(self._bmi._job_meta.grid_type)}"
                )

            # Increment or initialize output step count
            if self._bmi._job_meta.current_output_step is None:
                self._bmi._job_meta.current_output_step = 1
            else:
                self._bmi._job_meta.current_output_step += 1

            # Optional sub-output timestamp
            # if self._bmi._job_meta.sub_output_hour is not None:
            #     raise NotImplementedError(
            #         f"sub_output_hour (config SubOutputHour) is {repr(self._bmi._job_meta.sub_output_hour)} (not None) but is not used."
            #     )
            # # TODO This is not used. The raise not implemented error causes a fail on medium range blen due to the sub_output_hour being
            # specified in the config file. Testing was performed and not specifying sub_output_hour produces the same results for medium range blend
            # as of 7/13/2026. Not sure what this was intended to do but it is not used/effective at this time. Commenting it out to ensure medium range blend completes
            # and it is retained in case the intent is realized and it should be resurrected.

            # subOutDate = self._bmi._job_meta.first_fcst_cycle + datetime.timedelta(
            #     hours=self._bmi._job_meta.sub_output_hour
            # )

            # Compute the output timestamp for this step
            if self._bmi._job_meta.ana_flag:
                self._bmi._output_obj.outDate = (
                    self._bmi._job_meta.current_fcst_cycle
                    + datetime.timedelta(seconds=self._bmi._job_meta.output_freq * 60)
                )
            else:
                self._bmi._output_obj.outDate = (
                    self._bmi._job_meta.current_fcst_cycle
                    + datetime.timedelta(seconds=future_time)
                )

            self._bmi._job_meta.current_output_date = self._bmi._output_obj.outDate

            # Adjust file_date for AnA if needed
            file_date = (
                self._bmi._output_obj.outDate
                - datetime.timedelta(seconds=self._bmi._job_meta.output_freq * 60)
                if self._bmi._job_meta.ana_flag
                else self._bmi._output_obj.outDate
            )

            # Compute previous output date (used for downscaling logic)
            if self._bmi._job_meta.current_output_step == ana_factor:
                self._bmi._job_meta.prev_output_date = (
                    self._bmi._job_meta.current_output_date
                )
            else:
                self._bmi._job_meta.prev_output_date = (
                    self._bmi._job_meta.current_output_date
                    - datetime.timedelta(seconds=future_time)
                )

            # Print message on log file indicating the timestamp
            # we are currently processing for forcings
            if self._bmi._mpi_meta.rank == 0:
                self.log_debug(msg="=========================================")
                self.log_debug(
                    msg=f"Processing for output timestep: {file_date.strftime('%Y-%m-%d %H:%M')}"
                )

            self._bmi._job_meta.currentForceNum = 0
            self._bmi._job_meta.currentCustomForceNum = 0
            self.log_debug(
                msg=f"config_options.input_forcings: {self._bmi._job_meta.input_forcings}"
            )
            # Loop over each of the input forcings specified.
            self.log_debug(
                msg=f"Model.py forcing loop: {len(self._bmi._job_meta.input_forcings)} forcings configured: {self._bmi._job_meta.input_forcings}"
            )

            for force_key in self._bmi._job_meta.input_forcings:
                self.log_debug(msg=f"force_key: {force_key}")
                self.log_debug(msg=f"config_options.aws: {self._bmi._job_meta.aws}")
                # Pass these methods for AORC data is ERA5-Interim blend is requested
                # so we can finish filling in the missing gaps
                if (
                    force_key == 23
                    and 12 in self._bmi._job_meta.input_forcings
                    and 21 in self._bmi._job_meta.input_forcings
                ):
                    input_forcings = self._bmi._input_forcing_mod[force_key]

                    # These are not used
                    # AORC_mask = input_forcings.regridded_mask_AORC
                    # AORC_elem_mask = input_forcings.regridded_mask_elem_AORC
                else:
                    input_forcings = self._bmi._input_forcing_mod[force_key]
                    input_forcings.calc_neighbor_files(
                        self._bmi._job_meta,
                        self._bmi._output_obj.outDate,
                        self._bmi._mpi_meta,
                    )

                # Handle AORC and NWM force keys
                self.__handle_aorc_and_nwm_force_keys(input_forcings, force_key)

                # If skipping this forcing, continue early
                # NOTE this is used by the esmf regrid pytests, to halt the loop before "manually" calling a particular regrid function.
                if input_forcings.skip is True:
                    self.log_debug(msg=f"Breaking loop for force_key {force_key}")
                    break

                # Regrid forcings.
                input_forcings.regrid_inputs(
                    self._bmi._job_meta, self._bmi.geo_meta, self._bmi._mpi_meta
                )
                self.check_program_status()

                # Run check on regridded fields for reasonable values that are not missing values.
                err_handler.check_forcing_bounds(
                    self._bmi._job_meta, input_forcings, self._bmi._mpi_meta
                )
                self.check_program_status()

                # If we are restarting a forecast cycle, re-calculate the neighboring files, and regrid the
                # next set of forcings as the previous step just regridded the previous forcing.
                self.__use_rstFlag(input_forcings)

                # Run temporal interpolation on the grids.
                input_forcings.temporal_interpolate_inputs(
                    self._bmi._job_meta, self._bmi._mpi_meta
                )
                self.check_program_status()

                # Run bias correction.
                bias_correction.run_bias_correction(
                    input_forcings,
                    self._bmi._job_meta,
                    self._bmi.geo_meta,
                    self._bmi._mpi_meta,
                )
                self.check_program_status()

                # Run downscaling on grids for this output timestep.
                downscale.run_downscaling(
                    input_forcings,
                    self._bmi._job_meta,
                    self._bmi.geo_meta,
                    self._bmi._mpi_meta,
                )
                self.check_program_status()

                # Layer in forcings from this product.
                layeringMod.layer_final_forcings(
                    self._bmi._output_obj,
                    input_forcings,
                    self._bmi._job_meta,
                )
                self.check_program_status()

                self._bmi._job_meta.currentForceNum += 1

                # NOTE currentCustomForceNum does not appear to be used.
                if force_key == 10:
                    self._bmi._job_meta.currentCustomForceNum += 1

                self.log_debug(msg=f"End of loop for force_key {force_key}")

            # Process supplemental precipitation if we specified in the configuration file.
            if self._bmi._job_meta.number_supp_pcp > 0:
                for supp_pcp_key in self._bmi._job_meta.supp_precip_forcings:
                    if supp_pcp_key != 13:
                        # Below comment copied from earlier code, the comment had been just above the call to ``disaggregate_fun``.
                        # TODO input_forcings has not yet been initialized, so this is a bug waiting to happen
                        self.__process_supp_precip_key(input_forcings, supp_pcp_key)

            # Call the output routines
            #   adjust date for AnA if necessary
            if self._bmi._job_meta.ana_flag:
                self._bmi._output_obj.outDate = file_date

                ################ Commenting this out to bypass NWM forcing file output functionality #########
                # self._bmi._output_obj.output_final_ldasin(self._bmi._job_meta, self._bmi.geo_meta, self._bmi._mpi_meta)
                # self.check_program_status()
                ##############################################################################################
        else:
            input_forcings = None

        return input_forcings

    def __handle_aorc_and_nwm_force_keys(
        self, input_forcings: forcingInputMod.InputForcings, force_key: int
    ) -> None:
        """During ``loop_through_forcing_products``, handle the case where the force key is AORC or NWM.

        This code block was cut and pasted from the method ``loop_through_forcing_products`` during refactor.

        :param input_forcings forcingInputMod.InputForcings: Input forcings object to be modified.
        :param int force_key: Identifier for the forcing type.

        :warning: Modifies mutable arguments in-place.
        """
        proc_args = (self._bmi._job_meta, self._bmi._mpi_meta, self._bmi.geo_meta)

        if force_key in [12, 21, 27]:
            if self._bmi._job_meta.aws is None:
                # Calculate the previous and next input cycle files from the inputs.
                input_forcings.calc_neighbor_files(
                    self._bmi._job_meta,
                    self._bmi._output_obj.outDate,
                    self._bmi._mpi_meta,
                )
                self.check_program_status()
            else:
                if len(self._bmi._job_meta.input_forcings) != 1:
                    raise ValueError(
                        f"Expected to have 1 forcing key, but have {len(self._bmi._job_meta.input_forcings)}: {list(self._bmi._job_meta.input_forcings)}"
                    )
                if self.source_data_processor is None:
                    # Flag to indicate the AWS .zarr AORC method
                    if force_key == 12:
                        proc_cls = AORCConusProcessor
                    elif force_key == 21:
                        proc_cls = AORCAlaskaProcessor
                    # Flag to indicate the AWS .zarr NWMv3 Forcing file method
                    elif force_key == 27:
                        if self._bmi._job_meta.nwm_domain == "CONUS":
                            proc_cls = NWMV3ConusProcessor
                        elif self._bmi._job_meta.nwm_domain == "Hawaii":
                            proc_cls = NWMV3HawaiiProcessor
                        elif self._bmi._job_meta.nwm_domain == "PR":
                            proc_cls = NWMV3PuertoRicoProcessor
                        elif self._bmi._job_meta.nwm_domain == "Alaska":
                            proc_cls = NWMV3AlaskaProcessor
                        else:
                            raise ValueError(
                                f"Unsupported domain type ({self._bmi._job_meta.nwm_domain} for forcing type: {force_key} )"
                            )
                    else:
                        raise ValueError(f"Unexpected force_key: {force_key}")
                    self.source_data_processor = proc_cls(*proc_args)

                self._bmi._job_meta.aws_obj = (
                    self.source_data_processor.process_historical_data(
                        self._bmi._job_meta.current_time
                    )
                )

    def __process_supp_precip_key(
        self, input_forcings: forcingInputMod.InputForcings, supp_pcp_key: int
    ) -> None:
        """Process supplemental precipitation for a single supplemental precipitation key.

        This code block was cut and pasted from the methods
        ``loop_through_forcing_products`` and ``process_suplemental_precip`` during refactor.

        :param input_forcings forcingInputMod.InputForcings: Input forcings object to be modified.
        :param int supp_pcp_key: Identifier for the supplemental precipitation forcing.

        :warning: Modifies mutable arguments in-place.
        """
        # Like with input forcings, calculate the neighboring files to use.
        self._bmi._supp_pcp_mod[supp_pcp_key].calc_neighbor_files(
            self._bmi._job_meta,
            self._bmi._output_obj.outDate,
            self._bmi._mpi_meta,
        )
        self.check_program_status()

        # Regrid the supplemental precipitation.
        self._bmi._supp_pcp_mod[supp_pcp_key].regrid_inputs(
            self._bmi._job_meta, self._bmi.geo_meta, self._bmi._mpi_meta
        )
        self.check_program_status()

        if (
            self._bmi._supp_pcp_mod[supp_pcp_key].regridded_precip1 is not None
            and self._bmi._supp_pcp_mod[supp_pcp_key].regridded_precip2 is not None
        ):
            # Run check on regridded fields for reasonable values that are not missing values.
            err_handler.check_supp_pcp_bounds(
                self._bmi._job_meta,
                self._bmi._supp_pcp_mod[supp_pcp_key],
                self._bmi._mpi_meta,
                self._bmi.geo_meta,
            )
            self.check_program_status()

            self.disaggregate_fun(
                input_forcings,
                self._bmi._supp_pcp_mod[supp_pcp_key],
                self._bmi._job_meta,
                self._bmi._mpi_meta,
            )
            self.check_program_status()

            # Run temporal interpolation on the grids.
            self._bmi._supp_pcp_mod[supp_pcp_key].temporal_interpolate_inputs(
                self._bmi._job_meta, self._bmi._mpi_meta
            )
            self.check_program_status()

            # Layer in the supplemental precipitation into the current output object.
            layeringMod.layer_supplemental_forcing(
                self._bmi._output_obj,
                self._bmi._supp_pcp_mod[supp_pcp_key],
                self._bmi._job_meta,
            )
            self.check_program_status()

    def __use_rstFlag(self, input_forcings: forcingInputMod.InputForcings) -> None:
        """If restarting a forecast cycle, re-calculate neighboring files and regrid the
        next set of forcings, as the previous step regridded the prior forcing.

        This code block was cut and pasted from the method
        ``loop_through_forcing_products`` during refactor.

        :param input_forcings forcingInputMod.InputForcings: Input forcings object to be modified.

        :warning: Modifies mutable arguments in-place.
        """
        if input_forcings.rstFlag == 1:
            if (
                input_forcings.regridded_forcings1 is not None
                and input_forcings.regridded_forcings2 is not None
            ):
                # Set the forcings back to reflect we just regridded the previous set of inputs, not the next.
                if self._bmi._job_meta.grid_type == "gridded":
                    input_forcings.regridded_forcings1[:, :, :] = (
                        input_forcings.regridded_forcings2[:, :, :]
                    )
                elif self._bmi._job_meta.grid_type == "unstructured":
                    input_forcings.regridded_forcings1[:, :] = (
                        input_forcings.regridded_forcings2[:, :]
                    )
                    input_forcings.regridded_forcings1_elem[:, :] = (
                        input_forcings.regridded_forcings2_elem[:, :]
                    )
                elif self._bmi._job_meta.grid_type == "hydrofabric":
                    input_forcings.regridded_forcings1[:, :] = (
                        input_forcings.regridded_forcings2[:, :]
                    )
                else:
                    raise ValueError(
                        f"Unexpected grid_type: {repr(self._bmi._job_meta.grid_type)}"
                    )
            # Re-calculate the neighbor files.
            input_forcings.calc_neighbor_files(
                self._bmi._job_meta,
                self._bmi._output_obj.outDate,
                self._bmi._mpi_meta,
            )
            self.check_program_status()

            # Regrid the forcings for the end of the window.
            input_forcings.regrid_inputs(
                self._bmi._job_meta, self._bmi.geo_meta, self._bmi._mpi_meta
            )
            self.check_program_status()

            input_forcings.rstFlag = 0

    @time_function
    def process_suplemental_precip(
        self, input_forcings: forcingInputMod.InputForcings
    ) -> None:
        """Process supplemental precipitation for the current forecast cycle.

        :param input_forcings forcingInputMod.InputForcings: Input forcings object to be modified.

        :warning: Modifies mutable arguments in-place.
        """
        if self._bmi._job_meta.customSuppPcpFreq is not None:
            # Process supplemental precipitation if we specified in the configuration file.
            if self._bmi._job_meta.number_supp_pcp > 0:
                for supp_pcp_key in self._bmi._job_meta.supp_precip_forcings:
                    if supp_pcp_key == 14:
                        self.__process_supp_precip_key(input_forcings, supp_pcp_key)

    @time_function
    def write_output(self) -> None:
        """Write the output for the current forecast cycle.

        If user requests output for given domain, then call
        the I/O module to update opened netcdf file with forcing fields.
        """
        if (
            self._bmi._job_meta.forcing_output == 1
            or self._bmi._job_meta.grid_type == "hydrofabric"
        ):
            self._bmi._output_obj.gather_global_outputs(
                self._bmi._job_meta, self._bmi.geo_meta, self._bmi._mpi_meta
            )

    @time_function
    def update_bmi_output_dict(self) -> None:
        """Flatten the Forcings Engine output object and update the BMI dictionary.

        Loop through the Forcings Engine output object, flatten the 2D forcing arrays,
        and append them to the BMI object for advertisement through the BMI interface.

        The flattened variables are ordered as follows:

        0. U-wind (m/s)
        1. V-wind (m/s)
        2. Surface incoming longwave radiation flux (W/m²)
        3. Precipitation rate (mm/s)
        4. 2-meter air temperature (K)
        5. 2-meter specific humidity (kg/kg)
        6. Surface pressure (Pa)
        7. Surface incoming shortwave radiation flux (W/m²)
        8. Liquid precipitation fraction (%), available only in certain operational configurations
        """
        variables = copy.deepcopy(model_consts["update_dict_base_vars"])
        if self._bmi._job_meta.include_lqfrac == 1:
            variables.append(model_consts["update_dict_var_include_lqfraq"])

        if self._bmi._job_meta.grid_type == "gridded":
            for count, variable in enumerate(variables):
                self._bmi._values[f"{variable}_ELEMENT"] = (
                    self._bmi._output_obj.output_local[count, :, :].flatten()
                )
        elif self._bmi._job_meta.grid_type == "unstructured":
            for count, variable in enumerate(variables):
                self._bmi._values[f"{variable}_ELEMENT"] = (
                    self._bmi._output_obj.output_local_elem[count, :].flatten()
                )
                self._bmi._values[f"{variable}_NODE"] = (
                    self._bmi._output_obj.output_local[count, :].flatten()
                )
        elif self._bmi._job_meta.grid_type == "hydrofabric":
            for count, variable in enumerate(variables):
                self._bmi._values[f"{variable}_ELEMENT"] = (
                    self._bmi._output_obj.output_global[count, :].flatten()
                )
            self._bmi._values["CAT-ID"] = self._bmi._cat_ids
        else:
            raise ValueError(
                f"Unexpected grid_type: {repr(self._bmi._job_meta.grid_type)}"
            )
