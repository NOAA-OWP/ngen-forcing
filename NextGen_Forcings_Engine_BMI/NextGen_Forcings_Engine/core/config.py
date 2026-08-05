import configparser
import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone

# Use the Error, Warning, and Trapping System Package for logging
import numpy as np

from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.err_handler import (
    err_out_screen,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.time_handling import (
    calculate_lookback_window,
)

from . import mpi_utils

LOG = logging.getLogger("FORCING")
FORCE_COUNT = 27


class ConfigOptions:
    """Configuration abstract class for configuration options read in from the file specified by the user."""

    def __init__(self, config: dict, b_date=None, geogrid_arg=None):
        """Initialize the configuration class to empty None attributes.

        param config: The user-specified path to the configuration file.
        """
        self.bmi_time = None
        self.current_time = None
        self.bmi_time_index = 0
        self.input_forcings = None
        self.precip_only_flag = False
        self.supp_precip_forcings = None
        self.input_force_dirs = None
        self.input_force_types = None
        self.supp_precip_dirs = None
        self.supp_precip_file_types = None
        self.supp_precip_param_dir = None
        self.input_force_mandatory = None
        self.supp_precip_mandatory = None
        self.supp_pcp_max_hours = None
        self.number_inputs = None
        self.number_supp_pcp = None
        self.number_custom_inputs = 0
        self.output_freq = None
        self.sub_output_hour = None
        self.sub_output_freq = None
        self.scratch_dir = None
        self.useCompression = 0
        self.useFloats = 0
        self.num_output_steps = None
        self.num_supp_output_steps = None
        self.actual_output_steps = None
        self.realtime_flag = None
        self.refcst_flag = None
        self.ana_flag = None
        self.b_date_proc = b_date
        self.e_date_proc = None
        self.first_fcst_cycle = None
        self.current_fcst_cycle = None
        self.current_output_step = None
        self.cycle_length_minutes = None
        self.prev_output_date = None
        self.current_output_date = None
        self.look_back = None
        self.future_time = None
        self.fcst_freq = None
        self.nFcsts = None
        self.fcst_shift = None
        self.fcst_input_horizons = None
        self.fcst_input_offsets = None
        self.process_window = None
        self.spatial_meta = None
        self.grid_type = None
        self.grid_meta = None
        self.ExactExtract = None
        self.lat_var = None
        self.lon_var = None
        self.hgt_var = None
        self.cosalpha_var = None
        self.sinalpha_var = None
        self.slope_var = None
        self.slope_azimuth_var = None
        self.slope_var_elem = None
        self.slope_azimuth_var_elem = None
        self.nodecoords_var = None
        self.elemcoords_var = None
        self.elemconn_var = None
        self.numelemconn_var = None
        self.element_id_var = None
        self.hgt_elem_var = None
        self.ignored_border_widths = None
        self.regrid_opt = None
        self.weightsDir = None
        self.regrid_opt_supp_pcp = None
        self.config_path = config
        self.errMsg = None
        self.statusMsg = None
        self.logFile = None
        self.logHandle = None
        self.dScaleParamDirs = None
        self.paramFlagArray = None
        self.forceTemoralInterp = None
        self.suppTemporalInterp = None
        self.t2dDownscaleOpt = None
        self.swDownscaleOpt = None
        self.psfcDownscaleOpt = None
        self.precipDownscaleOpt = None
        self.q2dDownscaleOpt = None
        self.t2BiasCorrectOpt = None
        self.psfcBiasCorrectOpt = None
        self.q2BiasCorrectOpt = None
        self.windBiasCorrect = None
        self.swBiasCorrectOpt = None
        self.lwBiasCorrectOpt = None
        self.precipBiasCorrectOpt = None
        self.runCfsNldasBiasCorrect = False
        self.cfsv2EnsMember = None
        self.customSuppPcpFreq = None
        self.customFcstFreq = None
        self.rqiMethod = None
        self.rqiThresh = 1.0
        self.globalNdv = -9999.0
        self.d_program_init = datetime.now(timezone.utc)
        self.errFlag = 0
        self.nwmVersion = None
        self.nwmConfig = None
        self.include_lqfrac = False
        self.forcing_output = None
        self.aws = None
        self.aws_obj = None
        self.aws_time = None
        self.aorc_conus_source = "s3://noaa-nws-aorc-v1-1-1km"
        self.aorc_conus_year_url = "{source}/{year}.zarr"
        self.aorc_alaska_source = "s3://ngwpc-data/AORC/Alaska"
        self.aorc_alaska_url = (
            "{source}/{year}/{year}{month:02d}/AK_AORC-OWP_{date}.nc4"
        )
        self.nwm_source = "s3://noaa-nwm-retrospective-3-0-pds"

        self.nwm_geogrid = None
        self.geogrid = geogrid_arg
        self.geopackage = None

        self.uid64 = None
        self.broadcast_new_64bit_uid()

        self._scratch_dir_has_been_uniquefied = False

    def uniquefy_scratch_dir_as_child(self, uid: str) -> None:
        """Modify the existing scratch dir by adding the UID string available to all ranks from the MpiConfig class.
        This may only be called once. Subsequent calls will result in an error.
        This must be called by all ranks, once."""
        LOG.debug(f"Uniquefying scratch dir: adding suffix {uid} to {self.scratch_dir}")
        if not isinstance(uid, str):
            raise TypeError(f"Expected str, got {type(uid)} for type of uid: {uid}")
        if self.scratch_dir is None:
            raise ValueError("This cannot be ran while scratch_dir is None")
        if self._scratch_dir_has_been_uniquefied is True:
            raise ValueError(
                f"scratch_dir path has already been uniquefied: {self.scratch_dir}"
            )
        self.scratch_dir = os.path.join(self.scratch_dir, uid)
        self._scratch_dir_has_been_uniquefied = True
        self.make_scratch_dir()

    def make_scratch_dir(self) -> None:
        """Make the scratch dir and its parents."""
        os.makedirs(self.scratch_dir, exist_ok=True)
        LOG.debug(f"Scratch dir: {self.scratch_dir}")

    def broadcast_new_64bit_uid(self):
        """Broadcast a random uint64 then save the hash of that to self.uid64, which effectively broadcasts the same unique string to all ranks.
        Should be called once to avoid confusion."""
        if self.uid64 is not None:
            raise RuntimeError("self.uid64 has already been initialized.")
        self.uid64 = mpi_utils.get_new_broadcasted_uid()

    def validate_config(self, cfg_bmi: dict) -> None:
        """Validate in options from the configuration file and check that proper options were provided."""
        # Ensure b_date_proc is set; if not, read from the configuration file
        if self.b_date_proc is None:
            try:
                self.b_date_proc = cfg_bmi.get(
                    "RefcstBDateProc", None
                )  # Default to None if not found
                if self.b_date_proc is None:
                    err_out_screen(
                        "Unable to locate RefcstBDateProc under Logistics section in configuration file."
                    )
            except KeyError as e:
                err_out_screen(
                    "Unable to locate RefcstBDateProc under Logistics section in configuration file.",
                    e,
                )

        # Ensure geopackage is set; if not, read from the configuration file
        if self.geopackage is None:
            try:
                self.geopackage = cfg_bmi.get(
                    "Geopackage", None
                )  # Default to None if not found
                if self.geopackage is None:
                    err_out_screen(
                        "Unable to locate Geopackage in the configuration file."
                    )
            except KeyError as e:
                err_out_screen(
                    "Unable to locate Geopackage in the configuration file.", e
                )

        # Ensure geogrid is set; if not, read from the configuration file
        if self.geogrid is None:
            try:
                geogrid_base = cfg_bmi.get(
                    "GeogridIn", None
                )  # Default to None if not found
            except KeyError as e:
                err_out_screen(
                    "Unable to locate GeogridIn in the configuration file.", e
                )
            if geogrid_base is None:
                err_out_screen("Unable to locate GeogridIn in the configuration file.")
                self.geogrid = None
            else:
                geogrid_parent = os.path.dirname(geogrid_base)
                geogrid_filename = os.path.basename(geogrid_base)
                if self.uid64 is None:
                    raise ValueError("self.uid64 cannot be None, please initialize it.")
                self.geogrid = os.path.join(
                    geogrid_parent, f"{self.uid64}_{geogrid_filename}"
                )
            # Create directory for esmf_mesh file
            if not os.path.isdir(geogrid_parent):
                try:
                    os.makedirs(geogrid_parent, exist_ok=True)
                    LOG.debug(f"Created esmf mesh directory: {geogrid_parent}")
                except OSError as e:
                    err_out_screen(
                        f"Unable to create esmf_mesh directory: {geogrid_parent}. Error: {e}"
                    )

        # Read in the base input forcing options as an array of values to map.
        try:
            self.supp_precip_forcings = cfg_bmi["SuppPcp"]
        except KeyError as e:
            err_out_screen(
                "Unable to locate SuppPcp under SuppForcing section in configuration file.",
                e,
            )
        except configparser.NoOptionError as e:
            err_out_screen(
                "Unable to locate SuppPcp under SuppForcing section in configuration file.",
                e,
            )
        except json.decoder.JSONDecodeError as e:
            err_out_screen("Improper SuppPcp option specified in configuration file", e)

        self.number_supp_pcp = len(self.supp_precip_forcings)

        if self.number_supp_pcp == 1:
            if int(self.supp_precip_forcings[0]) == 14:
                self.precip_only_flag = True

        if not self.precip_only_flag:
            # Read in the base input forcing options as an array of values to map.
            try:
                self.input_forcings = cfg_bmi["InputForcings"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate InputForcings under Input section in configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate InputForcings under Input section in configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper InputForcings option specified in configuration file", e
                )
            if len(self.input_forcings) == 0:
                err_out_screen(
                    "Please choose at least one InputForcings dataset to process"
                )
            self.number_inputs = len(self.input_forcings)

            # Check to make sure forcing options make sense
            for force_opt in self.input_forcings:
                if force_opt < 0 or force_opt > FORCE_COUNT:
                    err_out_screen(
                        f"Please specify InputForcings values between 1 and {FORCE_COUNT}."
                    )

                # Keep tabs on how many custom input forcings we have.
                if force_opt == 10:
                    self.number_custom_inputs = self.number_custom_inputs + 1

                # Flag to force mandatory configuration option to specify the NWM geogrid file if user requests
                # NWM forcing files to be regridded to a given domain configuration
                if force_opt == 27:
                    try:
                        self.nwm_geogrid = cfg_bmi["NWM_Geogrid"]
                    except KeyError as e:
                        err_out_screen(
                            "Unable to locate NWM Geogrid file required for the NWM forcings module. Need to specify the pathway to the NWM geo_em_DOMAIN.nc file to the NWM_Geogrid configuration input option within the configuration file.",
                            e,
                        )
                    except configparser.NoOptionError as e:
                        err_out_screen(
                            "Unable to locate NWM Geogrid file required for the NWM forcings module. Need to specify the pathway to the NWM geo_em_DOMAIN.nc file to the NWM_Geogrid configuration input option within the configuration file.",
                            e,
                        )
                    except json.decoder.JSONDecodeError as e:
                        err_out_screen(
                            "Improper NWM Geogrid file option specified in configuration file",
                            e,
                        )

            # Read in the input forcings types (GRIB[1|2], NETCDF)
            try:
                # self.input_force_types = config.get('Input', 'InputForcingTypes').strip("[]").split(',')
                # self.input_force_types = [ftype.strip() for ftype in self.input_force_types]
                self.input_force_types = cfg_bmi["InputForcingTypes"]
                if self.input_force_types == [""]:
                    self.input_force_types = []
            except KeyError as e:
                err_out_screen(
                    "Unable to locate InputForcingTypes in Input section in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate InputForcingTypes in Input section in the configuration file.",
                    e,
                )
            if len(self.input_force_types) != self.number_inputs:
                err_out_screen(
                    "Number of InputForcingTypes must match the number "
                    "of InputForcings in the configuration file."
                )
            for file_type in self.input_force_types:
                if file_type not in [
                    "GRIB1",
                    "GRIB2",
                    "NETCDF",
                    "NETCDF4",
                    "NWM",
                    "ZARR",
                    "GRIB2_CFS",
                ]:
                    err_out_screen(
                        f'Invalid forcing file type "{file_type}" specified. '
                        "Only GRIB1, GRIB2, NETCDF, NWM, ZARR, and GRIB2_CFS are supported"
                    )

            # Read in the input directories for each forcing option.
            try:
                self.input_force_dirs = cfg_bmi["InputForcingDirectories"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate InputForcingDirectories in Input section in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate InputForcingDirectories in Input section in the configuration file.",
                    e,
                )
            if len(self.input_force_dirs) != self.number_inputs:
                err_out_screen(
                    "Number of InputForcingDirectories must match the number "
                    "of InputForcings in the configuration file."
                )
            # Loop through and ensure all input directories exist. Also strip out any whitespace
            # or new line characters.
            for dir_tmp in range(0, len(self.input_force_dirs)):
                self.input_force_dirs[dir_tmp] = self.input_force_dirs[dir_tmp].strip()

                dir_path = self.input_force_dirs[dir_tmp]
                forcing_type = self.input_forcings[dir_tmp]
                is_aws_forcing = forcing_type in [12, 21, 27]

                if not os.path.isdir(dir_path):
                    if is_aws_forcing:
                        self.aws = True
                    else:
                        try:
                            os.makedirs(dir_path, exist_ok=True)
                            LOG.debug(f"Created missing forcing directory: {dir_path}")
                        except OSError as e:
                            err_out_screen(
                                f"Unable to create forcing directory: {dir_path}. Error: {e}"
                            )

            # Read in the mandatory enforcement options for input forcings.
            try:
                self.input_force_mandatory = cfg_bmi["InputMandatory"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate InputMandatory under Input section in configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate InputMandatory under Input section in configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper InputMandatory option specified in configuration file", e
                )

            if len(self.input_force_mandatory) != self.number_inputs:
                err_out_screen(
                    "Please specify InputMandatory values for each corresponding input "
                    "forcings in the configuration file."
                )
            # Check to make sure enforcement options makes sense.
            for enforce_opt in self.input_force_mandatory:
                if enforce_opt < 0 or enforce_opt > 1:
                    err_out_screen(
                        "Invalid InputMandatory chosen in the configuration file. Please choose a value of 0 or 1 for each corresponding input forcing."
                    )

        # Read in the output frequency
        try:
            self.output_freq = cfg_bmi["OutputFrequency"]
        except ValueError as e:
            err_out_screen(
                "Improper OutputFrequency value specified  in the configuration file."
            )
        except KeyError as e:
            err_out_screen(
                "Unable to locate OutputFrequency in the configuration file."
            )
        except configparser.NoOptionError as e:
            err_out_screen(
                "Unable to locate OutputFrequency in the configuration file."
            )
        if self.output_freq <= 0:
            err_out_screen(
                "Please specify an OutputFrequency that is greater than zero minutes."
            )

        if self.precip_only_flag:
            # Read in the custom supp output frequency
            try:
                self.customSuppPcpFreq = int(cfg_bmi["customSuppPcpFreq"])
            except ValueError as e:
                err_out_screen(
                    "Improper customSuppPcpFreq value specified  in the configuration file.",
                    e,
                )
            except KeyError as e:
                err_out_screen(
                    "Unable to locate customSuppPcpFreq in the configuration file.", e
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate customSuppPcpFreq in the configuration file.", e
                )
            if self.output_freq <= 0:
                err_out_screen(
                    "Please specify an customSuppPcpFreq that is greater than zero minutes."
                )

        # Read in the sub output hour
        try:
            self.sub_output_hour = int(cfg_bmi["SubOutputHour"])
        except ValueError as e:
            err_out_screen(
                "Improper SubOutputHour value specified  in the configuration file.", e
            )
        except KeyError as e:
            err_out_screen(
                "Unable to locate SubOutputHour in the configuration file.", e
            )
        except configparser.NoOptionError as e:
            err_out_screen(
                "Unable to locate SubOutputHour in the configuration file.", e
            )
        if self.sub_output_hour < 0:
            err_out_screen(
                "Please specify an SubOutputHour that is greater than zero minutes."
            )
        if self.sub_output_hour == 0:
            self.sub_output_hour = None
        # Read in the output frequency
        try:
            self.sub_output_freq = int(cfg_bmi["SubOutFreq"])
        except ValueError as e:
            err_out_screen(
                "Improper SubOutFreq value specified  in the configuration file.", e
            )
        except KeyError as e:
            err_out_screen("Unable to locate SubOutFreq in the configuration file.", e)
        except configparser.NoOptionError as e:
            err_out_screen("Unable to locate SubOutFreq in the configuration file.", e)
        if self.sub_output_freq < 0:
            err_out_screen(
                "Please specify an SubOutFreq that is greater than zero minutes."
            )
        if self.sub_output_freq == 0:
            self.sub_output_freq = None

        # TODO Can this be a /tmp directory?
        # Read in the scratch temporary directory, which also may contain output forcing file if requested.
        try:
            self.scratch_dir = cfg_bmi["ScratchDir"]
        except ValueError as e:
            err_out_screen(
                "Improper ScratchDir specified in the configuration file.", e
            )
        except KeyError as e:
            err_out_screen("Unable to locate ScratchDir in the configuration file.", e)
        except configparser.NoOptionError as e:
            err_out_screen("Unable to locate ScratchDir in the configuration file.", e)

        self.make_scratch_dir()

        # Read in compression option
        try:
            self.useCompression = cfg_bmi["compressOutput"]
        except KeyError as e:
            err_out_screen("Unable to locate compressOut in the configuration file.", e)
        except configparser.NoOptionError as e:
            err_out_screen("Unable to locate compressOut in the configuration file.", e)
        except ValueError as e:
            err_out_screen("Improper compressOut value.", e)
        if self.useCompression < 0 or self.useCompression > 1:
            err_out_screen("Please choose a compressOut value of 0 or 1.")

        # Read in floating-point option
        try:
            self.useFloats = cfg_bmi["floatOutput"]
        except KeyError as e:
            # err_out_screen('Unable to locate floatOutput in the configuration file.', e)
            self.useFloats = 0
        except configparser.NoOptionError as e:
            # err_out_screen('Unable to locate floatOutput in the configuration file.', e)
            self.useFloats = 0
        except ValueError as e:
            err_out_screen(
                "Improper floatOutput value: {}".format(cfg_bmi["includeLQFraq"])
            )
        if self.useFloats < 0 or self.useFloats > 1:
            err_out_screen("Please choose a floatOutput value of 0 or 1.")

        # Read in lqfrac option
        try:
            self.include_lqfrac = cfg_bmi["includeLQFrac"]
        except KeyError as e:
            # err_out_screen('Unable to locate includeLQFraq in the configuration file.', e)
            self.include_lqfrac = 0
        except configparser.NoOptionError as e:
            # err_out_screen('Unable to locate includeLQFraq in the configuration file.', e)
            self.useFinclude_lqfracloats = 0
        except ValueError as e:
            err_out_screen(
                "Improper includeLQFrac value: {}".format(cfg_bmi["includeLQFraq"]), e
            )
        if self.include_lqfrac < 0 or self.include_lqfrac > 1:
            err_out_screen("Please choose an includeLQFrac value of 0 or 1.")

        # Read in Forcing output option
        try:
            self.forcing_output = cfg_bmi["Output"]
        except KeyError as e:
            self.forcing_output = 0
        except configparser.NoOptionError as e:
            self.forcing_output = 0
        except ValueError as e:
            err_out_screen(
                "Improper Forcing Output value: {}".format(cfg_bmi["Output"]), e
            )
        if self.forcing_output < 0 or self.forcing_output > 1:
            err_out_screen(
                "Please choose a Forcing Output value of 0 (No output) or 1 (output)."
            )

        # Read AnA flag option
        try:
            # check both the Forecast section and if it's not there, the old BiasCorrection location
            self.ana_flag = int(cfg_bmi["AnAFlag"])
        except KeyError as e:
            err_out_screen("Unable to locate AnAFlag in the configuration file.", e)
        except configparser.NoOptionError as e:
            err_out_screen("Unable to locate AnAFlag in the configuration file.", e)
        except ValueError as e:
            err_out_screen("Improper AnAFlag value ", e)
        if self.ana_flag < 0 or self.ana_flag > 1:
            err_out_screen("Please choose a AnAFlag value of 0 or 1.")

        # For the NextGen Forcings Engine BMI, we are assuming a realtime or reforecast simulation.
        try:
            self.look_back = cfg_bmi["LookBack"]
            if self.look_back <= 0 and self.look_back != -9999:
                err_out_screen(
                    "Please specify a positive LookBack or -9999 for realtime."
                )
        except ValueError as e:
            err_out_screen(
                "Improper LookBack value entered into the configuration file. Please check your entry.",
                e,
            )
        except KeyError as e:
            err_out_screen(
                "Unable to locate LookBack in the configuration file. Please verify entries exist.",
                e,
            )
        except configparser.NoOptionError as e:
            err_out_screen(
                "Unable to locate LookBack in the configuration file. Please verify entries exist.",
                e,
            )

        # Process the beginning date of reforecast forcings to process

        if self.b_date_proc:
            beg_date_tmp = self.b_date_proc
            e = ""
        else:
            try:
                beg_date_tmp = cfg_bmi["RefcstBDateProc"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate RefcstBDateProc under Logistics section in configuration file.",
                    e,
                )
                beg_date_tmp = None
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate RefcstBDateProc under Logistics section in configuration file.",
                    e,
                )
                beg_date_tmp = None

        if beg_date_tmp != -9999:
            if isinstance(beg_date_tmp, str) and len(beg_date_tmp) != 12:
                err_out_screen(
                    "Improper RefcstBDateProc length entered into the configuration file. Please check your entry.",
                    e,
                )
            try:
                self.b_date_proc = datetime.strptime(beg_date_tmp, "%Y%m%d%H%M")
            except ValueError as e:
                err_out_screen(
                    "Improper RefcstBDateProc value entered into the configuration file. Please check your entry.",
                    e,
                )
        else:
            self.b_date_proc = -9999

        LOG.info(f"Begin date: {beg_date_tmp}")

        # If the Retro flag is off, and lookback is off, then we assume we are
        # running a reforecast.
        if self.look_back == -9999:
            self.realtime_flag = False
            self.refcst_flag = True
        elif self.b_date_proc == -9999:
            self.realtime_flag = True
            self.refcst_flag = True
        else:
            # The processing window will be calculated based on current time and the
            # lookback option since this is a realtime instance.
            self.realtime_flag = False
            self.refcst_flag = False
            # self.b_date_proc = -9999
            # self.e_date_proc = -9999

        # Calculate the delta time between the beginning and ending time of processing.
        # self.process_window = self.e_date_proc - self.b_date_proc

        # Read in the ForecastFrequency option.
        try:
            self.fcst_freq = cfg_bmi["ForecastFrequency"]
        except ValueError as e:
            err_out_screen(
                "Improper ForecastFrequency value entered into the configuration file. Please check your entry.",
                e,
            )
        except KeyError as e:
            err_out_screen(
                "Unable to locate ForecastFrequency in the configuration file. Please verify entries exist.",
                e,
            )
        except configparser.NoOptionError as e:
            err_out_screen(
                "Unable to locate ForecastFrequency in the configuration file. Please verify entries exist.",
                e,
            )
        if self.fcst_freq <= 0:
            err_out_screen(
                "Please specify a ForecastFrequency in the configuration file greater than zero."
            )
        # Currently, we only support daily or sub-daily forecasts. Any other iterations should
        # be done using custom config files for each forecast cycle.
        if self.fcst_freq > 1440:
            err_out_screen(
                "Only forecast cycles of daily or sub-daily are supported at this time"
            )

        # Read in the ForecastShift option. This is ONLY done for the realtime instance as
        # it's used to calculate the beginning of the processing window.
        if True:  # was: self.realtime_flag:
            try:
                self.fcst_shift = cfg_bmi["ForecastShift"]
            except ValueError as e:
                err_out_screen(
                    "Improper ForecastShift value entered into the configuration file. Please check your entry.",
                    e,
                )
            except KeyError as e:
                err_out_screen(
                    "Unable to locate ForecastShift in the configuration file. Please verify entries exist.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate ForecastShift in the configuration file. Please verify entries exist.",
                    e,
                )
            if self.fcst_shift < 0:
                err_out_screen(
                    "Please specify a ForecastShift in the configuration file greater than or equal to zero."
                )

            # Calculate the beginning/ending processing dates if we are running realtime
            if self.realtime_flag:
                calculate_lookback_window(self)

        # if self.refcst_flag:
        # Calculate the number of forecasts to issue, and verify the user has chosen a
        # correct divider based on the dates
        # dt_tmp = self.e_date_proc - self.b_date_proc
        # if (dt_tmp.days * 1440 + dt_tmp.seconds / 60.0) % self.fcst_freq != 0:
        #    err_out_screen('Please choose an equal divider forecast frequency for your '
        #                               'specified reforecast range.')
        # self.nFcsts = int((dt_tmp.days * 1440 + dt_tmp.seconds / 60.0) / self.fcst_freq)

        # Flag to constrain AORC forcing data cycle output
        # for optTmp in self.input_forcings:
        # if optTmp == 12:
        # self.nFcsts = 1
        self.nFcsts = 1

        if self.look_back != -9999:
            calculate_lookback_window(self)

        if not self.precip_only_flag:
            # Read in the ForecastInputHorizons options.
            try:
                self.fcst_input_horizons = cfg_bmi["ForecastInputHorizons"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate ForecastInputHorizons under Forecast section in configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate ForecastInputHorizons under Forecast section in configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper ForecastInputHorizons option specified in configuration file",
                    e,
                )
            if len(self.fcst_input_horizons) != self.number_inputs:
                err_out_screen(
                    "Please specify ForecastInputHorizon values for each corresponding input forcings for forecasts."
                )

            # Check to make sure the horizons options make sense. There will be additional
            # checking later when input choices are mapped to input products.
            for horizonOpt in self.fcst_input_horizons:
                if horizonOpt <= 0:
                    err_out_screen(
                        "Please specify ForecastInputHorizon values greater than zero."
                    )
        else:
            # Read in the ForecastInputHorizons options.
            try:
                self.fcst_input_horizons = cfg_bmi["ForecastInputHorizons"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate ForecastInputHorizons under Forecast section in configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate ForecastInputHorizons under Forecast section in configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper ForecastInputHorizons option specified in configuration file",
                    e,
                )
                if len(self.fcst_input_horizons) != 1:
                    err_out_screen(
                        "Please specify ForecastInputHorizon values for each corresponding input forcings for forecasts."
                    )

        if not self.precip_only_flag:
            # Read in the ForecastInputOffsets options.
            try:
                self.fcst_input_offsets = cfg_bmi["ForecastInputOffsets"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate ForecastInputOffsets under Forecast section in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate ForecastInputOffsets under Forecast section in the configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper ForecastInputOffsets option specified in the configuration file.",
                    e,
                )
            if len(self.fcst_input_offsets) != self.number_inputs:
                err_out_screen(
                    "Please specify ForecastInputOffset values for each corresponding input forcings for forecasts."
                )
            # Check to make sure the input offset options make sense. There will be additional
            # checking later when input choices are mapped to input products.
            for inputOffset in self.fcst_input_offsets:
                if inputOffset < 0:
                    err_out_screen(
                        "Please specify ForecastInputOffset values greater than or equal to zero."
                    )

        # Calculate the length of the forecast cycle, based on the maximum
        # length of the input forcing length chosen by the user.
        self.cycle_length_minutes = max(self.fcst_input_horizons)

        # Ensure the number maximum cycle length is an equal divider of the output
        # time step specified by the user.
        if self.cycle_length_minutes % self.output_freq != 0:
            err_out_screen(
                "Please specify an output time step that is an equal divider of the maximum of the forecast time horizons specified."
            )

        if self.sub_output_hour is None:
            # Calculate the number of output time steps per forecast cycle.
            self.num_output_steps = int(self.cycle_length_minutes / self.output_freq)
            if self.precip_only_flag:
                self.num_supp_output_steps = (
                    int(self.cycle_length_minutes) / self.customSuppPcpFreq
                )
            if self.ana_flag:
                self.actual_output_steps = np.int32(self.nFcsts)
            else:
                self.actual_output_steps = np.int32(self.num_output_steps)
        else:
            # Calculate the number of output time steps per forecast cycle.
            self.num_output_steps = (
                int(
                    (self.cycle_length_minutes - (self.sub_output_hour * 60))
                    / self.sub_output_freq
                )
                + int((self.sub_output_hour * 60) / self.output_freq)
                - 1
            )
            if self.precip_only_flag:
                self.num_supp_output_steps = (
                    int(self.cycle_length_minutes) / self.customSuppPcpFreq
                )
            if self.ana_flag:
                self.actual_output_steps = np.int32(self.nFcsts)
            else:
                self.actual_output_steps = np.int32(self.num_output_steps)

        # Process the grid type
        try:
            self.grid_type = cfg_bmi["GRID_TYPE"]
        except KeyError as e:
            err_out_screen("Unable to locate GRID_TYPE in the configuration file.", e)
        except configparser.NoOptionError as e:
            err_out_screen("Unable to locate GRID_TYPE in the configuration file.", e)
        if (
            self.grid_type.lower() != "gridded"
            and self.grid_type.lower() != "unstructured"
            and self.grid_type.lower() != "hydrofabric"
        ):
            err_out_screen(
                'GRID_TYPE in the configuration file only accepts "unstructured", "gridded", or "hydrofabric" as options.'
            )

        if self.grid_type.lower() == "gridded":
            # Process the geogrid variable information
            try:
                self.lon_var = cfg_bmi["LONVAR"]
            except KeyError as e:
                err_out_screen("Unable to locate LONVAR in the configuration file.", e)
            except configparser.NoOptionError as e:
                err_out_screen("Unable to locate LONVAR in the configuration file.", e)
            try:
                self.lat_var = cfg_bmi["LATVAR"]
            except KeyError as e:
                err_out_screen("Unable to locate LATVAR in the configuration file.", e)
            except configparser.NoOptionError as e:
                err_out_screen("Unable to locate LATVAR in the configuration file.", e)

        elif self.grid_type.lower() == "unstructured":
            # Process the geogrid variable information
            try:
                self.nodecoords_var = cfg_bmi["NodeCoords"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate NodeCoords for unstructured mesh in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate NodeCoords for unstructured mesh in the configuration file.",
                    e,
                )
            try:
                self.elemcoords_var = cfg_bmi["ElemCoords"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate ElemCoords for unstructured mesh in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate ElemCoords for unstructured mesh in the configuration file.",
                    e,
                )
            try:
                self.elemconn_var = cfg_bmi["ElemConn"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate ElemConn for unstructured mesh in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate ElemConn for unstructured mesh in the configuration file.",
                    e,
                )
            try:
                self.numelemconn_var = cfg_bmi["NumElemConn"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate NumElemConn for unstructured mesh in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate NumElemConn for unstructured mesh in the configuration file.",
                    e,
                )

        elif self.grid_type.lower() == "hydrofabric":
            # Process the geogrid variable information
            try:
                self.nodecoords_var = cfg_bmi["NodeCoords"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate NodeCoords for unstructured mesh in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate NodeCoords for unstructured mesh in the configuration file.",
                    e,
                )
            try:
                self.elemcoords_var = cfg_bmi["ElemCoords"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate ElemCoords for unstructured mesh in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate ElemCoords for unstructured mesh in the configuration file.",
                    e,
                )
            try:
                self.element_id_var = cfg_bmi["ElemID"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate ElemID for unstructured mesh in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate ElemID for unstructured mesh in the configuration file.",
                    e,
                )
            try:
                self.elemconn_var = cfg_bmi["ElemConn"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate ElemConn for unstructured mesh in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate ElemConn for unstructured mesh in the configuration file.",
                    e,
                )
            try:
                self.numelemconn_var = cfg_bmi["NumElemConn"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate NumElemConn for unstructured mesh in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate NumElemConn for unstructured mesh in the configuration file.",
                    e,
                )

        # Process geospatial information

        if self.geogrid:
            LOG.debug(f"Geogrid: {self.geogrid}")
        else:
            try:
                self.geogrid = cfg_bmi["GeogridIn"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate GeogridIn in the configuration file.", e
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate GeogridIn in the configuration file.", e
                )

        # Check for the optional geospatial land metadata file.
        try:
            self.spatial_meta = cfg_bmi["SpatialMetaIn"]
        except KeyError as e:
            err_out_screen(
                "Unable to locate SpatialMetaIn in the configuration file.", e
            )
        if len(self.spatial_meta) == 0:
            # No spatial metadata file found.
            self.spatial_meta = None
        else:
            if not os.path.isfile(self.spatial_meta):
                err_out_screen(
                    "Unable to locate optional spatial metadata file: "
                    + self.spatial_meta
                )

        if not self.precip_only_flag:
            # Check for the IgnoredBorderWidths
            try:
                self.ignored_border_widths = cfg_bmi["IgnoredBorderWidths"]
            except (KeyError, configparser.NoOptionError):
                # if didn't specify, no worries, just set to 0
                self.ignored_border_widths = [0.0] * self.number_inputs
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper IgnoredBorderWidths option specified in the configuration file."
                    "({} was supplied".format(
                        cfg_bmi["Geospatial"]["IgnoredBorderWidths"]
                    ),
                    e,
                )
            if len(self.ignored_border_widths) != self.number_inputs:
                err_out_screen(
                    "Please specify IgnoredBorderWidths values for each "
                    "corresponding input forcings for SuppForcing."
                    "({} was supplied".format(self.ignored_border_widths)
                )
            if any(map(lambda x: x < 0, self.ignored_border_widths)):
                err_out_screen(
                    "Please specify IgnoredBorderWidths values greater than or equal to zero:"
                    "({} was supplied".format(self.ignored_border_widths)
                )

        if not self.precip_only_flag:
            # Process regridding options.
            try:
                self.regrid_opt = cfg_bmi["RegridOpt"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate RegridOpt under the Regridding section in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate RegridOpt under the Regridding section in the configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper RegridOpt options specified in the configuration file.", e
                )
            if len(self.regrid_opt) != self.number_inputs:
                err_out_screen(
                    "Please specify RegridOpt values for each corresponding input forcings in the configuration file.",
                    e,
                )
            # Check to make sure regridding options makes sense.
            for regridOpt in self.regrid_opt:
                if regridOpt < 1 or regridOpt > 3:
                    err_out_screen(
                        "Invalid RegridOpt chosen in the configuration file. Please choose a "
                        "value of 1-2 for each corresponding input forcing."
                    )
            try:
                # Read weight file directory (optional)
                self.weightsDir = cfg_bmi["RegridWeightsDir"]
            except Exception:
                # Set wieghtsDir to None; this will create regrid object in memory
                self.weightsDir = None
            if self.weightsDir:
                # if we do have one specified, make sure it exists
                if not os.path.exists(self.weightsDir):
                    err_out_screen(
                        "ESMF Weights file directory specified ({}) but does not exist"
                    ).format(self.weightsDir)

        # Calculate the beginning/ending processing dates if we are running realtime
        if self.realtime_flag:
            calculate_lookback_window(self)

        # Create temporary array to hold flags if we need input parameter files.
        param_flag = np.empty([len(self.input_forcings)], int)
        param_flag[:] = 0
        if not self.precip_only_flag:
            # Read in temporal interpolation options.
            try:
                self.forceTemoralInterp = cfg_bmi["ForcingTemporalInterpolation"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate ForcingTemporalInterpolation under the Interpolation section in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate ForcingTemporalInterpolation under the Interpolation section in the configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper ForcingTemporalInterpolation options specified in the configuration file.",
                    e,
                )
            if len(self.forceTemoralInterp) != self.number_inputs:
                err_out_screen(
                    "Please specify ForcingTemporalInterpolation values for each corresponding input forcings in the configuration file."
                )
            # Ensure the forcingTemporalInterpolation values make sense.
            for temporalInterpOpt in self.forceTemoralInterp:
                if temporalInterpOpt < 0 or temporalInterpOpt > 2:
                    err_out_screen(
                        "Invalid ForcingTemporalInterpolation chosen in the configuration file. "
                        "Please choose a value of 0-2 for each corresponding input forcing."
                    )

            # Read in the temperature downscaling options.
            try:
                self.t2dDownscaleOpt = cfg_bmi["TemperatureDownscaling"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate TemperatureDownscaling under the Downscaling section of the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate TemperatureDownscaling under the Downscaling section of the configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper TemperatureDownscaling options specified in the configuration file.",
                    e,
                )
            if len(self.t2dDownscaleOpt) != self.number_inputs:
                err_out_screen(
                    "Please specify TemperatureDownscaling values for each corresponding input forcings in the configuration file."
                )
            # Ensure the downscaling options chosen make sense.
            count_tmp = 0
            for optTmp in self.t2dDownscaleOpt:
                if optTmp < 0 or optTmp > 2:
                    err_out_screen(
                        "Invalid TemperatureDownscaling options specified in the configuration file."
                    )
                if optTmp == 2:
                    param_flag[count_tmp] = 1
                count_tmp = count_tmp + 1

            # Read in the pressure downscaling options.
            try:
                self.psfcDownscaleOpt = cfg_bmi["PressureDownscaling"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate PressureDownscaling under the Downscaling section of the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate PressureDownscaling under the Downscaling section of the configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper PressureDownscaling options specified in the configuration file."
                )
            if len(self.psfcDownscaleOpt) != self.number_inputs:
                err_out_screen(
                    "Please specify PressureDownscaling values for each corresponding input forcings in the configuration file."
                )
            # Ensure the downscaling options chosen make sense.
            for optTmp in self.psfcDownscaleOpt:
                if optTmp < 0 or optTmp > 1:
                    err_out_screen(
                        "Invalid PressureDownscaling options specified in the configuration file."
                    )

            # Read in the shortwave downscaling options
            try:
                self.swDownscaleOpt = cfg_bmi["ShortwaveDownscaling"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate ShortwaveDownscaling under the Downscaling section of the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate ShortwaveDownscaling under the Downscaling section of the configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper ShortwaveDownscaling options specified in the configuration file.",
                    e,
                )
            if len(self.swDownscaleOpt) != self.number_inputs:
                err_out_screen(
                    "Please specify ShortwaveDownscaling values for each corresponding input forcings in the configuration file."
                )
            # Ensure the downscaling options chosen make sense.
            for optTmp in self.swDownscaleOpt:
                if optTmp < 0 or optTmp > 1:
                    err_out_screen(
                        "Invalid ShortwaveDownscaling options specified in the configuration file."
                    )

            # Read in humidity downscaling options.
            try:
                self.q2dDownscaleOpt = cfg_bmi["HumidityDownscaling"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate HumidityDownscaling under the Downscaling section of the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate HumidityDownscaling under the Downscaling section of the configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper HumidityDownscaling options specified in the configuration file.",
                    e,
                )
            if len(self.q2dDownscaleOpt) != self.number_inputs:
                err_out_screen(
                    "Please specify HumidityDownscaling values for each corresponding "
                    "input forcings in the configuration file."
                )
            # Ensure the downscaling options chosen make sense.
            for optTmp in self.q2dDownscaleOpt:
                if optTmp < 0 or optTmp > 1:
                    err_out_screen(
                        "Invalid HumidityDownscaling options specified in the configuration file."
                    )

        # Read in the precipitation downscaling options
        try:
            self.precipDownscaleOpt = cfg_bmi["PrecipDownscaling"]
        except KeyError as e:
            err_out_screen(
                "Unable to locate PrecipDownscaling under the Downscaling section of the configuration file.",
                e,
            )
        except configparser.NoOptionError as e:
            err_out_screen(
                "Unable to locate PrecipDownscaling under the Downscaling section of the configuration file.",
                e,
            )
        except json.decoder.JSONDecodeError as e:
            err_out_screen(
                "Improper PrecipDownscaling options specified in the configuration file.",
                e,
            )
        if not self.precip_only_flag:
            if len(self.precipDownscaleOpt) != self.number_inputs:
                err_out_screen(
                    "Please specify PrecipDownscaling values for each corresponding "
                    "input forcings in the configuration file."
                )
        # Ensure the downscaling options chosen make sense.
        count_tmp = 0
        for optTmp in self.precipDownscaleOpt:
            if optTmp < 0 or optTmp > 1:
                err_out_screen(
                    "Invalid PrecipDownscaling options specified in the configuration file."
                )
            if optTmp == 1:
                param_flag[count_tmp] = 1
            count_tmp = count_tmp + 1

        # Read in the downscaling parameter directory.
        try:
            self.dScaleParamDirs = cfg_bmi["DownscalingParamDirs"]
        except KeyError as e:
            err_out_screen(
                "Unable to locate DownscalingParamDirs in the configuration file.", e
            )
        except configparser.NoOptionError as e:
            err_out_screen(
                "Unable to locate DownscalingParamDirs in the configuration file.", e
            )
        if len(self.dScaleParamDirs) != len(self.input_forcings):
            err_out_screen(
                "Please specify a downscaling parameter directory for each "
                "corresponding downscaling option that requires one."
            )
        # Loop through each downscaling parameter directory and make sure they exist.
        for dirTmp in range(0, len(self.dScaleParamDirs)):
            if not os.path.isdir(self.dScaleParamDirs[dirTmp]):
                err_out_screen(
                    "Unable to locate parameter directory: "
                    + os.path.abspath(self.dScaleParamDirs[dirTmp])
                )

        if (
            [1] in self.q2dDownscaleOpt
            or [1] in self.swDownscaleOpt
            or [1] in self.psfcDownscaleOpt
            or [1, 2] in self.t2dDownscaleOpt
        ):
            # Process the geogrid information for downscaling
            try:
                self.sinalpha_var = cfg_bmi["SINALPHA"]
            except Exception:
                self.sinalpha_var = None
            try:
                self.cosalpha_var = cfg_bmi["COSALPHA"]
            except Exception:
                self.cosalpha_var = None
            if self.grid_type.lower() == "hydrofabric":
                try:
                    self.slope_var = cfg_bmi["SLOPE"]
                except KeyError as e:
                    err_out_screen(
                        "Unable to locate SLOPE variable in the hydrofabric configuration file. Required variable since user turned on a downscaling option.",
                        e,
                    )
                except configparser.NoOptionError as e:
                    err_out_screen(
                        "Unable to locate SLOPE variable in the hydrofabric configuration file. Required variable since user turned on a downscaling option.",
                        e,
                    )
                try:
                    self.slope_azimuth_var = cfg_bmi["SLOPE_AZIMUTH"]
                except KeyError as e:
                    err_out_screen(
                        "Unable to locate SLOPE_AZIMUTH variable in the hydrofabric configuration file. Required variable since user turned on a downscaling option.",
                        e,
                    )
                except configparser.NoOptionError as e:
                    err_out_screen(
                        "Unable to locate SLOPE_AZIMUTH variable in the hydrofabric configuration file. Required variable since user turned on a downscaling option.",
                        e,
                    )
            else:
                try:
                    self.slope_var = cfg_bmi["SLOPE"]
                except Exception:
                    self.slope_var = None
                try:
                    self.slope_azimuth_var = cfg_bmi["SLOPE_AZIMUTH"]
                except Exception:
                    self.slope_azimuth_var = None
                if self.grid_type.lower() == "unstructured":
                    try:
                        self.slope_var_elem = cfg_bmi["SLOPE_ELEM"]
                    except Exception:
                        self.slope_var_elem = None
                    try:
                        self.slope_azimuth_var_elem = cfg_bmi["SLOPE_AZIMUTH_ELEM"]
                    except Exception:
                        self.slope_azimuth_var_elem = None

            if self.grid_type.lower() == "unstructured":
                try:
                    self.hgt_elem_var = cfg_bmi["HGTVAR_ELEM"]
                except KeyError as e:
                    err_out_screen(
                        "Unable to locate HGTVAR_ELEM in the configuration file. Required variable since user turned on a downscaling option.",
                        e,
                    )
                except configparser.NoOptionError as e:
                    err_out_screen(
                        "Unable to locate HGTVAR_ELEM in the configuration file. Required variable since user turned on a downscaling option.",
                        e,
                    )

            try:
                self.hgt_var = cfg_bmi["HGTVAR"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate HGTVAR in the configuration file. Required variable since user turned on a downscaling option.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate HGTVAR in the configuration file. Required variable since user turned on a downscaling option.",
                    e,
                )

        #   * Bias Correction Options *
        if not self.precip_only_flag:
            # Read in temperature bias correction options
            try:
                self.t2BiasCorrectOpt = cfg_bmi["TemperatureBiasCorrection"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate TemperatureBiasCorrection under the BiasCorrection section of the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate TemperatureBiasCorrection under the BiasCorrection section of the configuration file.",
                    e,
                )
            except json.JSONDecodeError as e:
                err_out_screen(
                    "Improper TemperatureBiasCorrection options specified in the configuration file.",
                    e,
                )
            if len(self.t2BiasCorrectOpt) != self.number_inputs:
                err_out_screen(
                    "Please specify TemperatureBiasCorrection values for each corresponding input forcings in the configuration file."
                )
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.t2BiasCorrectOpt:
                if optTmp < 0 or optTmp > 4:
                    err_out_screen(
                        "Invalid TemperatureBiasCorrection options specified in the configuration file."
                    )

            # Read in surface pressure bias correction options.
            try:
                self.psfcBiasCorrectOpt = cfg_bmi["PressureBiasCorrection"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate PressureBiasCorrection under the BiasCorrection section of the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate PressureBiasCorrection under the BiasCorrection section of the configuration file.",
                    e,
                )
            except json.JSONDecodeError as e:
                err_out_screen(
                    "Improper PressureBiasCorrection options specified in the configuration file.",
                    e,
                )
            if len(self.psfcDownscaleOpt) != self.number_inputs:
                err_out_screen(
                    "Please specify PressureBiasCorrection values for each corresponding input forcings in the configuration file."
                )
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.psfcBiasCorrectOpt:
                if optTmp < 0 or optTmp > 1:
                    err_out_screen(
                        "Invalid PressureBiasCorrection options specified in the configuration file."
                    )
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Read in humidity bias correction options.
            try:
                self.q2BiasCorrectOpt = cfg_bmi["HumidityBiasCorrection"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate HumidityBiasCorrection under the  BiasCorrection section of the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate HumidityBiasCorrection under the BiasCorrection section of the configuration file.",
                    e,
                )
            except json.JSONDecodeError as e:
                err_out_screen(
                    "Improper HumdityBiasCorrection options specified in the configuration file.",
                    e,
                )
            if len(self.q2BiasCorrectOpt) != self.number_inputs:
                err_out_screen(
                    "Please specify HumidityBiasCorrection values for each corresponding input forcings in the configuration file."
                )
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.q2BiasCorrectOpt:
                if optTmp < 0 or optTmp > 2:
                    err_out_screen(
                        "Invalid HumidityBiasCorrection options specified in the configuration file."
                    )
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Read in wind bias correction options.
            try:
                self.windBiasCorrect = cfg_bmi["WindBiasCorrection"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate WindBiasCorrection under the BiasCorrection section of the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate WindBiasCorrection under the BiasCorrection section of the configuration file.",
                    e,
                )
            except json.JSONDecodeError as e:
                err_out_screen(
                    "Improper WindBiasCorrection options specified in the configuration file.",
                    e,
                )
            if len(self.windBiasCorrect) != self.number_inputs:
                err_out_screen(
                    "Please specify WindBiasCorrection values for each corresponding input forcings in the configuration file."
                )
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.windBiasCorrect:
                if optTmp < 0 or optTmp > 4:
                    err_out_screen(
                        "Invalid WindBiasCorrection options specified in the configuration file."
                    )
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Read in shortwave radiation bias correction options.
            try:
                self.swBiasCorrectOpt = cfg_bmi["SwBiasCorrection"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate SwBiasCorrection under the BiasCorrection section of the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate SwBiasCorrection under the BiasCorrection section of the configuration file.",
                    e,
                )
            except json.JSONDecodeError as e:
                err_out_screen(
                    "Improper SwBiasCorrection options specified in the configuration file.",
                    e,
                )
            if len(self.swBiasCorrectOpt) != self.number_inputs:
                err_out_screen(
                    "Please specify SwBiasCorrection values for each corresponding input forcings in the configuration file."
                )
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.swBiasCorrectOpt:
                if optTmp < 0 or optTmp > 2:
                    err_out_screen(
                        "Invalid SwBiasCorrection options specified in the configuration file."
                    )
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Read in longwave radiation bias correction options.
            try:
                self.lwBiasCorrectOpt = cfg_bmi["LwBiasCorrection"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate LwBiasCorrection under the BiasCorrection section of the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate LwBiasCorrection under the BiasCorrection section of the configuration file.",
                    e,
                )
            except json.JSONDecodeError as e:
                err_out_screen(
                    "Improper LwBiasCorrection options specified in the configuration file.",
                    e,
                )
            if len(self.lwBiasCorrectOpt) != self.number_inputs:
                err_out_screen(
                    "Please specify LwBiasCorrection values for each corresponding input forcings in the configuration file."
                )
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.lwBiasCorrectOpt:
                if optTmp < 0 or optTmp > 4:
                    err_out_screen(
                        "Invalid LwBiasCorrection options specified in the configuration file."
                    )
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Read in precipitation bias correction options.
            try:
                self.precipBiasCorrectOpt = cfg_bmi["PrecipBiasCorrection"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate PrecipBiasCorrection under the BiasCorrection section of the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate PrecipBiasCorrection under the BiasCorrection section of the configuration file.",
                    e,
                )
            except json.JSONDecodeError as e:
                err_out_screen(
                    "Improper PrecipBiasCorrection options specified in the configuration file.",
                    e,
                )
            if not self.precip_only_flag:
                if len(self.precipBiasCorrectOpt) != self.number_inputs:
                    err_out_screen(
                        "Please specify PrecipBiasCorrection values for each corresponding input forcings in the configuration file."
                    )
            # Ensure the bias correction options chosen make sense.
            for optTmp in self.precipBiasCorrectOpt:
                if optTmp < 0 or optTmp > 1:
                    err_out_screen(
                        "Invalid PrecipBiasCorrection options specified in the configuration file."
                    )
                if optTmp == 1:
                    # We are running NWM-Specific bias-correction of CFSv2 that needs to take place prior to regridding.
                    self.runCfsNldasBiasCorrect = True

            # Putting a constraint here that CFSv2-NLDAS bias correction (NWM only) is chosen, it must be turned on
            # for ALL variables.
            if self.runCfsNldasBiasCorrect:
                if (
                    min(self.precipBiasCorrectOpt) != 1
                    and max(self.precipBiasCorrectOpt) != 1
                ):
                    err_out_screen(
                        "CFSv2-NLDAS NWM bias correction must be activated for Precipitation under this configuration."
                    )
                if min(self.lwBiasCorrectOpt) != 1 and max(self.lwBiasCorrectOpt) != 1:
                    err_out_screen(
                        "CFSv2-NLDAS NWM bias correction must be activated for long-wave radiation under this configuration."
                    )
                if min(self.swBiasCorrectOpt) != 1 and max(self.swBiasCorrectOpt) != 1:
                    err_out_screen(
                        "CFSv2-NLDAS NWM bias correction must be activated for short-wave radiation under this configuration."
                    )
                if min(self.t2BiasCorrectOpt) != 1 and max(self.t2BiasCorrectOpt) != 1:
                    err_out_screen(
                        "CFSv2-NLDAS NWM bias correction must be activated for surface temperature under this configuration."
                    )
                if min(self.windBiasCorrect) != 1 and max(self.windBiasCorrect) != 1:
                    err_out_screen(
                        "CFSv2-NLDAS NWM bias correction must be activated for wind forcings under this configuration."
                    )
                if min(self.q2BiasCorrectOpt) != 1 and max(self.q2BiasCorrectOpt) != 1:
                    err_out_screen(
                        "CFSv2-NLDAS NWM bias correction must be activated for specific humidity under this configuration."
                    )
                if (
                    min(self.psfcBiasCorrectOpt) != 1
                    and max(self.psfcBiasCorrectOpt) != 1
                ):
                    err_out_screen(
                        "CFSv2-NLDAS NWM bias correction must be activated for surface pressure under this configuration."
                    )
                # Make sure we don't have any other forcings activated. This can only be ran for CFSv2.
                for opt_tmp in self.input_forcings:
                    if opt_tmp != 7:
                        err_out_screen(
                            "CFSv2-NLDAS NWM bias correction can only be used in CFSv2-only configurations"
                        )

        # Read in supplemental precipitation options as an array of values to map.
        try:
            self.supp_precip_forcings = cfg_bmi["SuppPcp"]
        except KeyError as e:
            err_out_screen(
                "Unable to locate SuppPcp under SuppForcing section in configuration file.",
                e,
            )
        except configparser.NoOptionError as e:
            err_out_screen(
                "Unable to locate SuppPcp under SuppForcing section in configuration file.",
                e,
            )
        except json.decoder.JSONDecodeError as e:
            err_out_screen("Improper SuppPcp option specified in configuration file", e)
        self.number_supp_pcp = len(self.supp_precip_forcings)

        # Read in the supp pcp types (GRIB[1|2], NETCDF)
        try:
            self.supp_precip_file_types = cfg_bmi["SuppPcpForcingTypes"]
            self.supp_precip_file_types = [
                stype.strip() for stype in self.supp_precip_file_types
            ]
            if self.supp_precip_file_types == [""]:
                self.supp_precip_file_types = []
        except KeyError as e:
            err_out_screen(
                "Unable to locate SuppPcpForcingTypes in SuppForcing section in the configuration file.",
                e,
            )
        except configparser.NoOptionError as e:
            err_out_screen(
                "Unable to locate SuppPcpForcingTypes in SuppForcing section in the configuration file.",
                e,
            )
        if len(self.supp_precip_file_types) != self.number_supp_pcp:
            err_out_screen(
                "Number of SuppPcpForcingTypes ({}) must match the number "
                "of SuppPcp inputs ({}) in the configuration file.".format(
                    len(self.supp_precip_file_types), self.number_supp_pcp
                )
            )
        for file_type in self.supp_precip_file_types:
            if file_type not in ["GRIB1", "GRIB2", "NETCDF"]:
                err_out_screen(
                    'Invalid SuppForcing file type "{}" specified. '
                    "Only GRIB1, GRIB2, and NETCDF are supported".format(file_type)
                )

        if self.number_supp_pcp > 0:
            # Check to make sure supplemental precip options make sense. Also read in the RQI threshold
            # if any radar products where chosen.
            for suppOpt in self.supp_precip_forcings:
                if suppOpt < 0 or suppOpt > 16:
                    err_out_screen(
                        "Please specify SuppForcing values between 1 and 16."
                    )
                # Read in RQI threshold to apply to radar products.
                if suppOpt in (1, 2, 7, 10, 11, 12):
                    try:
                        self.rqiMethod = cfg_bmi["RqiMethod"]
                    except KeyError as e:
                        err_out_screen(
                            "Unable to locate RqiMethod under SuppForcing section in the configuration file.",
                            e,
                        )
                    except configparser.NoOptionError as e:
                        err_out_screen(
                            "Unable to locate RqiMethod under SuppForcing section in the configuration file.",
                            e,
                        )
                    except json.decoder.JSONDecodeError as e:
                        err_out_screen(
                            "Improper RqiMethod option in the configuration file.", e
                        )

                    # Check that if we have more than one RqiMethod, it's the correct number
                    if type(self.rqiMethod) is list:
                        if len(self.rqiMethod) != self.number_supp_pcp:
                            err_out_screen(
                                "Number of RqiMethods ({}) must match the number "
                                "of SuppPcp inputs ({}) in the configuration file, or "
                                "supply a single method for all inputs".format(
                                    len(self.rqiMethod), self.number_supp_pcp
                                )
                            )
                    elif type(self.rqiMethod) is int:
                        # Support 'classic' mode of single method
                        self.rqiMethod = [self.rqiMethod] * self.number_supp_pcp

                    # Make sure the RqiMethod(s) makes sense.
                    for method in self.rqiMethod:
                        if method < 0 or method > 2:
                            err_out_screen(
                                "Please specify RqiMethods of either 0, 1, or 2."
                            )

                    try:
                        self.rqiThresh = cfg_bmi["RqiThreshold"]
                    except KeyError as e:
                        err_out_screen(
                            "Unable to locate RqiThreshold under SuppForcing section in the configuration file.",
                            e,
                        )
                    except configparser.NoOptionError as e:
                        err_out_screen(
                            "Unable to locate RqiThreshold under SuppForcing section in the configuration file.",
                            e,
                        )
                    except json.decoder.JSONDecodeError as e:
                        err_out_screen(
                            "Improper RqiThreshold option in the configuration file.", e
                        )

                    # Check that if we have more than one RqiThreshold, it's the correct number
                    if type(self.rqiThresh) is list:
                        if len(self.rqiThresh) != self.number_supp_pcp:
                            err_out_screen(
                                "Number of RqiThresholds ({}) must match the number "
                                "of SuppPcp inputs ({}) in the configuration file, or "
                                "supply a single threshold for all inputs".format(
                                    len(self.rqiThresh), self.number_supp_pcp
                                )
                            )
                    elif type(self.rqiThresh) is float:
                        # Support 'classic' mode of single threshold
                        self.rqiThresh = [self.rqiThresh] * self.number_supp_pcp

                    # Make sure the RQI threshold makes sense.
                    for threshold in self.rqiThresh:
                        if threshold < 0.0 or threshold > 1.0:
                            err_out_screen(
                                "Please specify RqiThresholds between 0.0 and 1.0."
                            )

            # Read in the input directories for each supplemental precipitation product.
            try:
                self.supp_precip_dirs = cfg_bmi["SuppPcpDirectories"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate SuppPcpDirectories in SuppForcing section in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate SuppPcpDirectories in SuppForcing section in the configuration file.",
                    e,
                )

            # Loop through and ensure all supp pcp directories exist. Also strip out any whitespace
            # or new line characters.
            for dirTmp in range(0, len(self.supp_precip_dirs)):
                self.supp_precip_dirs[dirTmp] = self.supp_precip_dirs[dirTmp].strip()
                if not os.path.isdir(self.supp_precip_dirs[dirTmp]):
                    try:
                        os.makedirs(self.supp_precip_dirs[dirTmp], exist_ok=True)
                        LOG.debug(
                            f"Created supp pcp directory: {self.supp_precip_dirs[dirTmp]}"
                        )
                    except OSError as e:
                        err_out_screen(
                            f"Unable to create supp pcp directory: {self.supp_precip_dirs[dirTmp]}. Error: {e}"
                        )

            # Special case for ExtAnA where we treat comma separated stage IV, MRMS data as one SuppPcp input
            if 11 in self.supp_precip_forcings or 12 in self.supp_precip_forcings:
                if len(self.supp_precip_forcings) != 1:
                    err_out_screen(
                        "CONUS or Alaska Stage IV/MRMS SuppPcp option is only supported as a standalone option"
                    )
                self.supp_precip_dirs = [",".join(self.supp_precip_dirs)]

            if len(self.supp_precip_dirs) != self.number_supp_pcp:
                err_out_screen(
                    "Number of SuppPcpDirectories must match the number of SuppForcing in the configuration file."
                )

            # Process supplemental precipitation enforcement options
            try:
                self.supp_precip_mandatory = cfg_bmi["SuppPcpMandatory"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate SuppPcpMandatory under the SuppForcing section in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate SuppPcpMandatory under the SuppForcing section in the configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper SuppPcpMandatory options specified in the configuration file.",
                    e,
                )
            if len(self.supp_precip_mandatory) != self.number_supp_pcp:
                err_out_screen(
                    "Please specify SuppPcpMandatory values for each corresponding "
                    "supplemental precipitation options in the configuration file."
                )
            # Check to make sure enforcement options makes sense.
            for enforceOpt in self.supp_precip_mandatory:
                if enforceOpt < 0 or enforceOpt > 1:
                    err_out_screen(
                        "Invalid SuppPcpMandatory chosen in the configuration file. "
                        "Please choose a value of 0 or 1 for each corresponding "
                        "supplemental precipitation product."
                    )

            # Read in the regridding options.
            try:
                self.regrid_opt_supp_pcp = cfg_bmi["RegridOptSuppPcp"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate RegridOptSuppPcp under the SuppForcing section in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate RegridOptSuppPcp under the SuppForcing section in the configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper RegridOptSuppPcp options specified in the configuration file.",
                    e,
                )
            if len(self.regrid_opt_supp_pcp) != self.number_supp_pcp:
                err_out_screen(
                    "Please specify RegridOptSuppPcp values for each corresponding supplemental "
                    "precipitation product in the configuration file."
                )
            # Check to make sure regridding options makes sense.
            for regridOpt in self.regrid_opt_supp_pcp:
                if regridOpt < 1 or regridOpt > 3:
                    err_out_screen(
                        "Invalid RegridOptSuppPcp chosen in the configuration file. "
                        "Please choose a value of 1-3 for each corresponding "
                        "supplemental precipitation product."
                    )

            # Read in temporal interpolation options.
            try:
                self.suppTemporalInterp = cfg_bmi["SuppPcpTemporalInterpolation"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate SuppPcpTemporalInterpolation under the SuppForcing section in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate SuppPcpTemporalInterpolation under the SuppForcing section in the configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper SuppPcpTemporalInterpolation options specified in the configuration file.",
                    e,
                )
            if len(self.suppTemporalInterp) != self.number_supp_pcp:
                err_out_screen(
                    "Please specify SuppPcpTemporalInterpolation values for each "
                    "corresponding supplemental precip products in the configuration file."
                )
            # Ensure the SuppPcpTemporalInterpolation values make sense.
            for temporalInterpOpt in self.suppTemporalInterp:
                if temporalInterpOpt < 0 or temporalInterpOpt > 2:
                    err_out_screen(
                        "Invalid SuppPcpTemporalInterpolation chosen in the configuration file. "
                        "Please choose a value of 0-2 for each corresponding input forcing"
                    )

            # Read in max time option
            try:
                self.supp_pcp_max_hours = cfg_bmi["SuppPcpMaxHours"]
            except (KeyError, configparser.NoOptionError):
                self.supp_pcp_max_hours = (
                    None  # if missing, don't care, just assume all time
                )

            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper SuppPcpMaxHours options specified in the configuration file.",
                    e,
                )

            if type(self.supp_pcp_max_hours) is list:
                if len(self.supp_pcp_max_hours) != self.number_supp_pcp:
                    err_out_screen(
                        "Number of SuppPcpMaxHours ({}) must match the number "
                        "of SuppPcp inputs ({}) in the configuration file, or "
                        "supply a single threshold for all inputs".format(
                            len(self.supp_pcp_max_hours), self.number_supp_pcp
                        )
                    )
            elif type(self.supp_pcp_max_hours) is float:
                # Support 'classic' mode of single threshold
                self.supp_pcp_max_hours = [
                    self.supp_pcp_max_hours
                ] * self.number_supp_pcp

            # Read in the SuppPcpInputOffsets options.
            try:
                self.supp_input_offsets = cfg_bmi["SuppPcpInputOffsets"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate SuppPcpInputOffsets under SuppForcing section in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate SuppPcpInputOffsets under SuppForcing section in the configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as e:
                err_out_screen(
                    "Improper SuppPcpInputOffsets option specified in the configuration file.",
                    e,
                )
            if len(self.supp_input_offsets) != self.number_supp_pcp:
                err_out_screen(
                    "Please specify SuppPcpInputOffsets values for each "
                    "corresponding input forcings for SuppForcing."
                )
            # Check to make sure the input offset options make sense. There will be additional
            # checking later when input choices are mapped to input products.
            for inputOffset in self.supp_input_offsets:
                if inputOffset < 0:
                    err_out_screen(
                        "Please specify SuppPcpInputOffsets values greater than or equal to zero."
                    )

            # Read in the optional parameter directory for supplemental precipitation.
            try:
                self.supp_precip_param_dir = cfg_bmi["SuppPcpParamDir"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate SuppPcpParamDir under the SuppForcing section  in the configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate SuppPcpParamDir under the SuppForcing section in the configuration file.",
                    e,
                )
            except ValueError as e:
                err_out_screen(
                    "Improper SuppPcpParamDir option specified in the configuration file.",
                    e,
                )
            if not os.path.isdir(self.supp_precip_param_dir):
                try:
                    os.makedirs(self.supp_precip_param_dir, exist_ok=True)
                    LOG.debug(
                        f"Created missing SuppPcpParamDir: {self.supp_precip_param_dir}"
                    )
                except OSError as e:
                    err_out_screen(
                        f"Unable to locate SuppPcpParamDir: {self.supp_precip_param_dir}. Error: {e}"
                    )

        if not self.precip_only_flag:
            # Read in Ensemble information
            # Read in CFS ensemble member information IF we have chosen CFSv2 as an input
            # forcing.
            for opt_tmp in self.input_forcings:
                if opt_tmp == 7:
                    try:
                        self.cfsv2EnsMember = cfg_bmi["cfsEnsNumber"]
                        LOG.debug(f"ens mem: {self.cfsv2EnsMember}")
                        LOG.debug(f"cfg ens mem: {cfg_bmi['cfsEnsNumber']}")
                    except KeyError as e:
                        err_out_screen(
                            "Unable to locate cfsEnsNumber under the Ensembles section of the configuration file",
                            e,
                        )
                    except configparser.NoOptionError as e:
                        err_out_screen(
                            "Unable to locate cfsEnsNumber under the Ensembles section of the configuration file",
                            e,
                        )
                    except json.JSONDecodeError as e:
                        err_out_screen(
                            "Improper cfsEnsNumber options specified in the configuration file",
                            e,
                        )
                    if int(self.cfsv2EnsMember) < 1 or int(self.cfsv2EnsMember) > 4:
                        err_out_screen(
                            "Please chose an cfsEnsNumber value of 1,2,3 or 4."
                        )

            # Read in information for the custom input NetCDF files that are to be processed.
            # Read in the ForecastInputHorizons options.
            try:
                self.customFcstFreq = cfg_bmi["custom_input_fcst_freq"]
            except KeyError as e:
                err_out_screen(
                    "Unable to locate custom_input_fcst_freq under Custom section in configuration file.",
                    e,
                )
            except configparser.NoOptionError as e:
                err_out_screen(
                    "Unable to locate custom_input_fcst_freq under Custom section in configuration file.",
                    e,
                )
            except json.decoder.JSONDecodeError as je:
                err_out_screen(
                    "Improper custom_input_fcst_freq  option specified in configuration file: "
                    + str(je)
                )
            if len(self.customFcstFreq) != self.number_custom_inputs:
                err_out_screen(
                    f"Improper custom_input fcst_freq specified. "
                    f"This number ({len(self.customFcstFreq)}) must "
                    f"match the frequency of custom input forcings selected "
                    f"({self.number_custom_inputs})."
                )

    @property
    def nwm_domain(self) -> str:
        """Extract NWM domain from the geogrid filename, using regex pattern."""
        if self.nwm_geogrid is None:
            return None
        pattern = r"geo_em_([a-zA-Z-_]+)\.nc$"  # E.g. extract "Puerto_Rico" from /foo/bar/esmf_mesh/NWM/domain/geo_em_Puerto_Rico.nc
        groups = re.findall(pattern, self.nwm_geogrid)
        if len(groups) != 1:
            raise ValueError(
                f"Could not determine NWM domain. {len(groups)} groups found (expected 1) in {self.nwm_geogrid} using regex pattern {pattern}"
            )
        domain = groups[0]
        if domain in ["PuertoRico", "Puerto_Rico", "PR"]:
            return "PR"
        else:
            return domain

    @property
    def nwm_url(self):
        """Construct NWM Zarr URL based on domain."""
        if self.nwm_domain is None:
            return None
        elif self.nwm_domain == "CONUS":
            return "{source}/{domain}/zarr/forcing/{var}.zarr"
        elif self.nwm_domain in ["Hawaii", "PR", "Alaska"]:
            return "{source}/{domain}/zarr/forcing.zarr"
        else:
            raise ValueError(
                f"Unknown domain. Expected 'CONUS', 'Hawaii', 'PR', or 'Alaska'; received: '{self.nwm_domain}'"
            )

    @property
    def use_data_at_current_time(self):
        """Determine if supplemental precipitation data can be used at the current output time."""
        if self.supp_pcp_max_hours:
            hrs_since_start = self.current_output_date - self.current_fcst_cycle
            return hrs_since_start <= timedelta(hours=self.supp_pcp_max_hours)
        else:
            return True
