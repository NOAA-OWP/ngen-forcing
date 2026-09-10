from __future__ import annotations

import configparser
import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

# Use the Error, Warning, and Trapping System Package for logging
import numpy as np

from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core import mpi_utils
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.consts import (
    CONFIGOPTIONS,
    FORCINGINPUTMOD,
    SUPPPRECIPMOD,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.err_handler import (
    err_out_screen,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.time_handling import (
    calculate_lookback_window,
)

LOG = logging.getLogger("FORCING")


class ConfigOptions:
    """Configuration abstract class for configuration options read in from the file specified by the user."""

    def __init__(self, cfg_bmi: dict, b_date: str = None, geogrid: str = None) -> None:
        """Initialize the configuration class to empty None attributes.

        The attributes of this class are populated by the validate_config function, which reads in the configuration file and checks that all necessary options are provided and properly formatted. The attributes of this class are used to control the flow of the program and the processing of input forcings.

        Args:
            cfg_bmi (dict): The configuration dictionary read in from the configuration file specified by the user. This should be read in using the config_utils.read_config function, which also handles any necessary preprocessing of the configuration file.
            b_date (str, optional): The beginning date of processing in the format YYYYMMDDHHMM. This is used to calculate the processing window for realtime simulations. If not provided, it will be read from the configuration file.
            geogrid (str, optional): The filepath to the geogrid file to be used for processing. This is used to specify the grid information for regridding input forcings. If not provided, it will be read from the configuration file.

        """
        if geogrid is not None:
            self.user_provided_geogrid_flag = True
        else:
            self.user_provided_geogrid_flag = False

        # If b_date not provided, try to read from config file
        if b_date is None:
            b_date = cfg_bmi.get("RefcstBDateProc", None)
        if b_date is None:
            err_out_screen(
                "RefcstBDateProc is either missing or None in configuration file."
            )

        self.b_date_proc = b_date
        self.cfg_bmi = cfg_bmi
        self.geogrid = geogrid

        self.bmi_time_index = 0
        self.globalNdv = -9999.0
        self.d_program_init = datetime.now(timezone.utc)
        self.errFlag = 0
        self.aorc_conus_source = "s3://noaa-nws-aorc-v1-1-1km"
        self.aorc_conus_year_url = "{source}/{year}.zarr"
        self.aorc_alaska_source = "s3://ngwpc-data/AORC/Alaska"
        self.aorc_alaska_url = (
            "{source}/{year}/{year}{month:02d}/AK_AORC-OWP_{date}.nc4"
        )
        self.nwm_source = "s3://noaa-nwm-retrospective-3-0-pds"

        self._scratch_dir_has_been_uniquefied = False

        # These must exist (as None) before the properties are accessed
        self._supp_precip_forcings = None
        self._b_date_proc = None
        self._input_forcings = None
        self._nwm_geogrid = None
        self._output_freq = None
        self._sub_output_hour = None
        self._sub_output_freq = None
        self._scratch_dir = None
        self._useCompression = None
        self._ana_flag = None
        self._look_back = None
        self._fcst_freq = None
        self._fcst_input_horizons = None
        self._spatial_meta = None
        self._geopackage = None
        self._geogrid = None
        self._grid_type = None
        # Backing vars for setters that guard on precip_only_flag and do not unconditionally assign
        self._fcst_input_offsets = None
        self._ignored_border_widths = None
        self._weightsDir = None
        self._forceTemoralInterp = None
        self._t2dDownscaleOpt = None
        self._psfcDownscaleOpt = None
        self._swDownscaleOpt = None
        self._q2dDownscaleOpt = None
        self._precipDownscaleOpt = None
        self._t2BiasCorrectOpt = None
        self._psfcBiasCorrectOpt = None
        self._q2BiasCorrectOpt = None
        self._windBiasCorrect = None
        self._swBiasCorrectOpt = None
        self._lwBiasCorrectOpt = None
        self._precipBiasCorrectOpt = None
        self._input_force_types = None
        self._dScaleParamDirs = None

        # set list of attributes from consts.py to None early on in the init process.
        # These are indexed from the consts dictionary.
        # This must happen before accessing properties like precip_only_flag
        for attr in CONFIGOPTIONS[__class__.__name__]:
            setattr(self, attr, None)
        self.broadcast_new_64bit_uid()

        # Extract optional forcing inputs (SuppPcp, InputForcings) from config; default to empty lists if not provided since code assumes these are always iterable
        supp_pcp = self.extract_input_variable("SuppPcp")
        self.supp_precip_forcings = supp_pcp if supp_pcp is not None else []
        if not self.precip_only_flag:
            input_forc = self.extract_input_variable("InputForcings")
            self.input_forcings = input_forc if input_forc is not None else []
            # Create temporary array to hold flags if we need input parameter files.
            self.param_flag = np.zeros([len(self.input_forcings)], int)
        else:
            self.input_forcings = []
            self.param_flag = np.array([], int)

        for (
            cfg_bmi_attr,
            config_options_attr,
        ) in self.try_config_get_except_attr_map.items():
            setattr(self, config_options_attr, self.try_config_get(cfg_bmi_attr))

        self.set_attrs(CONFIGOPTIONS["extract_input_variable_attrs_map"])

        if self.precip_only_flag:
            self.set_attrs(
                CONFIGOPTIONS["extract_input_variable_attrs_map_precip_only"]
            )
            self.set_attrs(
                CONFIGOPTIONS["extract_input_variable_attrs_map_not_precip_only"],
                set_none=True,
            )
        else:
            self.set_attrs(
                CONFIGOPTIONS["extract_input_variable_attrs_map_not_precip_only"]
            )
            self.set_attrs(
                CONFIGOPTIONS["extract_input_variable_attrs_map_precip_only"],
                set_none=True,
            )
            if 27 in self.input_forcings:
                self.nwm_geogrid = self.extract_input_variable("NWM_Geogrid")

        if self.perform_downscaling:
            self.set_attrs(CONFIGOPTIONS["downscaling_attrs_map"])
            if self.grid_type == "unstructured":
                self.set_attrs(CONFIGOPTIONS["downscaling_unstructred_attrs_map"])
        else:
            # Initialize downscaling attributes to None even if downscaling is not performed
            self.set_attrs(CONFIGOPTIONS["downscaling_attrs_map"], set_none=True)
            if self.grid_type == "unstructured":
                self.set_attrs(
                    CONFIGOPTIONS["downscaling_unstructred_attrs_map"], set_none=True
                )

        for cfg_bmi_attr, config_options_attr in CONFIGOPTIONS[
            "extract_input_variable_set_default_attrs_map"
        ].items():
            if config_options_attr in ["supp_pcp_max_hours", "weightsDir"]:
                default = None
            else:
                default = 0
            setattr(
                self,
                config_options_attr,
                self.extract_input_variable_set_default(cfg_bmi_attr, default),
            )

        # Call post_init to perform any calculations that depend on multiple attributes
        self.post_init()

    def post_init(self) -> None:
        """Attr setting/calculating operations that depend on others already being set.

        This method is called at the end of __init__ to perform any side-effect calculations
        that require multiple attributes to be initialized. This ensures independence of initialization order
        and matches the original pre-refactored code behavior, where calculations
        were performed after all attrs were read.
        """
        # Calculate the beginning/ending processing dates if we are running realtime or if look_back != -9999
        if self.look_back is not None and self.look_back != -9999:
            calculate_lookback_window(self)
        elif self.realtime_flag:
            calculate_lookback_window(self)

    @property
    def try_config_get_except_attr_map(self) -> dict:
        """Get the mapping of configuration variable names to class attribute names
        for variables that are extracted directly from the configuration file
        without any additional processing. This is used to control how variables are
        extracted from the configuration file and assigned to class attributes in a
        consistent way based on the mapping specified in the consts.py file.

        Don't mutate the module-level CONFIGOPTIONS object.
        Operate on a copy instead (and return that modified copy).
        """

        dict_map = CONFIGOPTIONS["try_config_get_except_attr_map"].copy()
        if self._b_date_proc is not None and "RefcstBDateProc" in dict_map:
            dict_map.pop("RefcstBDateProc")
        if self.geogrid is not None and "GeogridIn" in dict_map:
            dict_map.pop("GeogridIn")
        return dict_map

    @property
    def cfg_bmi(self) -> dict:
        """Return the configuration dictionary read in from the configuration file specified by the user."""
        return self._cfg_bmi

    @cfg_bmi.setter
    def cfg_bmi(self, value: dict) -> None:
        """Set the configuration dictionary read in from the configuration file specified by the user."""
        if not isinstance(value, dict):
            raise TypeError(
                f"Expected dict, got {type(value)} for type of cfg_bmi: {value}"
            )
        self._cfg_bmi = value

    @property
    def force_count(self) -> int:
        """Calculate the number of total possible input forcing options based on the
        length of the InputForcings list in consts.py. This is used for error checking
        to ensure users specify valid input forcing options in the configuration file.
        """
        return len(FORCINGINPUTMOD["PRODUCT_NAME"])

    @property
    def supp_precip_count(self) -> int:
        """Calculate the number of total possible supplemental precip forcing options
        based on the length of the Supplemental Precip PRODUCT_NAMES dict in consts.py.
        This is used for error checking to ensure users specify valid supplemental
        precip forcing options in the configuration file.
        """
        return len(SUPPPRECIPMOD["PRODUCT_NAMES"])

    @property
    def precip_only_flag(self) -> bool:
        """Flag to indicate whether the user has chosen to run the supplemental precip
        forcings module only, which will trigger some different processing pathways and
        error checking for certain configuration options.
        """
        precip_only = False
        if self.supp_precip_forcings is not None and len(self.supp_precip_forcings) > 0:
            if int(self.supp_precip_forcings[0]) == 14:
                precip_only = True
        return precip_only

    def set_attrs(self, attrs_dict: dict, set_none: bool = False):
        """Set the attributes of the class based on the configuration file. This is
        used to populate the attributes of the class after they have been read in and
        validated from the configuration file.
        """
        for cfg_bmi_attr, config_options_attr in attrs_dict.items():
            if set_none:
                attr = None
            else:
                attr = self.extract_input_variable(cfg_bmi_attr)
            setattr(self, config_options_attr, attr)

    def set_attrs_use_default(self, attrs_dict: dict):
        """Set the attributes of the class based on the configuration file. Set default
        value to default if not found in config file.
        """
        for cfg_bmi_attr, config_options_attr in attrs_dict.items():
            setattr(
                self,
                config_options_attr,
                self.extract_input_variable_set_default(cfg_bmi_attr),
            )

    def extract_input_variable(self, variable_name: str) -> str:
        """Extract the variable name from the configuration file for a given variable."""
        try:
            return self.cfg_bmi[variable_name]
        except ValueError as e:
            err_out_screen(
                f"Improper {variable_name} value specified  in the configuration file. Error: {e}"
            )
        except (KeyError, configparser.NoOptionError) as e:
            err_out_screen(
                f"Unable to locate {variable_name} in the configuration file. Error: {e}"
            )
        except json.decoder.JSONDecodeError as e:
            err_out_screen(
                f"Improper {variable_name} file option specified in configuration file. Error: {e}",
                e,
            )

    def extract_input_variable_set_default(self, variable_name: str, default=0) -> str:
        """Extract the variable name from the configuration file for a given variable,
        and set it to a default value if it is not found.
        """
        try:
            variable = self.cfg_bmi[variable_name]
        except (KeyError, configparser.NoOptionError) as e:
            variable = default
        except ValueError as e:
            err_out_screen(
                f"Improper {variable_name} value: {self.cfg_bmi[variable_name]}", e
            )
        if default == 0:
            if variable not in [0, 1]:
                err_out_screen(f"Please choose a {variable_name} value of 0  or 1.")
        return variable

    def try_config_get(self, variable_name: str) -> str:
        """Try to get a variable from the configuration file, and return a default value if it is not found."""
        try:
            var = self.cfg_bmi.get(variable_name)
            if var is None:
                err_out_screen(
                    f"Unable to locate {variable_name} in the configuration file."
                )
            return var
        except (KeyError, configparser.NoOptionError) as e:
            err_out_screen(
                f"Unable to locate {variable_name} in the configuration file.", e
            )

    def check_number_of_inputs(
        self, value: list, variable_name: str, input_type: str, number_inputs: int
    ) -> None:
        """Check that the number of inputs specified by the user in the configuration
        file matches the expected number of inputs for a given variable.
        """
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"Expected list or tuple for `value`, got type {type(value)}"
            )
        if len(value) != number_inputs:
            err_out_screen(
                f"Number of {variable_name} values must match the number of {input_type} in the configuration file."
            )

    def check_number_of_inputs_forcings(self, value: list, variable_name: str) -> None:
        """Check that the number of inputs specified by the user in the configuration
        file matches the expected number of inputs for a given variable, specifically
        for input forcings variables which should match the number of input forcing
        options specified by the user in the configuration file.
        """
        return self.check_number_of_inputs(
            value, variable_name, " InputForcings", self.number_inputs
        )

    def check_number_of_inputs_supp_pcp(self, value: list, variable_name: str) -> None:
        """Check that the number of inputs specified by the user in the configuration
        file matches the expected number of inputs for a given variable, specifically
        for supplemental precip forcing variables which should match the number of
        supplemental precip forcing options specified by the user in the configuration file.
        """
        return self.check_number_of_inputs(
            value, variable_name, " SupplementalPrecipForcings", self.number_supp_pcp
        )

    def check_input_values_in_range(
        self, value: list, variable_name: str, valid_input_options: list
    ) -> None:
        """Check that the input values specified by the user in the configuration file
        are within a valid range for a given variable.
        """
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"Expected list or tuple for `value`, got type {type(value)}"
            )
        for val in value:
            if val not in valid_input_options:
                err_out_screen(
                    f"Invalid {variable_name} value '{val}' specified in configuration file. Please specify valid values: {valid_input_options}."
                )

    def check_input_values_non_negative(self, value: list, variable_name: str) -> None:
        """Check that the input values specified by the user in the configuration file
        are positive for a given variable.
        """
        for val in value:
            if float(val) < 0:
                err_out_screen(
                    f"Invalid {variable_name} value '{val}' specified in configuration file. Please specify values greater than or equal to zero."
                )

    def check_input_values_positive(self, value: list, variable_name: str) -> None:
        """Check that the input values specified by the user in the configuration file
        are positive for a given variable.
        """
        for val in value:
            if val <= 0:
                err_out_screen(
                    f"Invalid {variable_name} value '{val}' specified in configuration file. Please specify values greater than zero."
                )

    def uniquefy_scratch_dir_as_child(self, uid: str) -> None:
        """Modify the existing scratch dir by adding the UID string available to all ranks from the MpiConfig class.

        This may only be called once. Subsequent calls will result in an error.
        This must be called by all ranks, once.
        """
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

    def make_scratch_dir(self, scratch_dir: str) -> None:
        """Make the scratch dir and its parents."""
        os.makedirs(scratch_dir, exist_ok=True)
        LOG.debug(f"Scratch dir: {scratch_dir}")

    def broadcast_new_64bit_uid(self) -> None:
        """Broadcast a random uint64 then save the hash of that to self.uid64, which effectively broadcasts the same unique string to all ranks.

        Should be called once to avoid confusion.
        """
        if self.uid64 is not None:
            raise RuntimeError("self.uid64 has already been initialized.")
        self.uid64 = mpi_utils.get_new_broadcasted_uid()

    @property
    def supp_precip_forcings(self):
        """Choose a set of supplemental precipitation file(s) to layer into the final
        LDASIN forcing files processed from the options above. The following is a mapping
        of numeric values to external input native forcing files.

        1. MRMS GRIB2 hourly radar-only QPE
        2. MRMS GRIB2 hourly gage-corrected radar QPE
        3. WRF-ARW 2.5 km 48-hr Hawaii nest precipitation.
        4. WRF-ARW 2.5 km 48-hr Puerto Rico nest precipitation.
        5. CONUS MRMS GRIB2 hourly MultiSensor QPE (Pass 2 or Pass 1)
        6. Hawaii MRMS GRIB2 hourly MultiSensor QPE (Pass 2 or Pass 1)
        7. MRMS SBCv2 Liquid Water Fraction (netCDF only)
        8. NBM Conus MR
        9. NBM Alaska MR
        10. Alaska MRMS (no liquid water fraction)
        11. Alaska Stage IV NWS Precip
        12. CONUS Stage IV NWS Precip
        13. MRMS PrecipFlag precipitation classification file
        14. Custom Frequency Supplementary Precipitation product (sub-hourly precip)
        15. NBM Puerto Rico
        16. NBM Hawaii
        - Example- SuppPcp: [1, 5, 13]
        """
        return self._supp_precip_forcings

    @supp_precip_forcings.setter
    def supp_precip_forcings(self, value: list) -> None:
        """Set the list of supplemental precip forcing options specified by the user in
        the configuration file. This is used to control which supplemental precip
        forcings are processed and how they are processed based on the other
        configuration options specified for each supplemental precip forcing.
        """
        if value is not None and len(value) > 0:
            self.check_input_values_in_range(
                [int(i) for i in value],
                "SuppPcp",
                list(range(1, self.supp_precip_count + 1)),
            )
        self._supp_precip_forcings = value

    @property
    def output_freq(self) -> int:
        """Get the output frequency in minutes specified by the user in the
        configuration file. This is used to control the output frequency of the
        processed forcings, and is necessary for both realtime and reforecast simulations.
        """
        return self._output_freq

    @output_freq.setter
    def output_freq(self, value: int) -> None:
        """Specify the output frequency in minutes. Note that any frequencies at higher
        intervals than what if provided as input will entail input forcing data being
        temporally interpolated.

        Example- OutputFrequency: 60
        """
        self.check_input_values_positive([value], "OutputFrequency")
        self._output_freq = value

    @property
    def sub_output_hour(self) -> int:
        """Get the sub-daily output hour specified by the user in the configuration file.
        This is used to control the output frequency of the processed forcings for
        sub-daily output frequencies, and is only necessary if the user has chosen a
        sub-daily output frequency in the configuration file.
        """
        return self._sub_output_hour

    @sub_output_hour.setter
    def sub_output_hour(self, value: int) -> None:
        """Sub output hour.

        New variable currently for NWMv3.1 operations to properly ingest GFS 13km
        forecast data that outputs various frequencies throughout the forecast cycle
        lifetime. This variable will properly account for reading time slices of the
        forecast cycle. Currently only needed for GFS 13km operational configuration.
        Otherwise, set this value to 0.

        Example- SubOutputHour: 0
        """
        self.check_input_values_non_negative([value], "SubOutputHour")
        if value == 0:
            value = None
        self._sub_output_hour = value

    @property
    def sub_output_freq(self) -> int:
        """Calculate the sub-daily output frequency in minutes based on the output
        frequency and sub-daily output hour specified by the user in the configuration
        file. This is used to control the output frequency of the processed forcings for
        sub-daily output frequencies, and is only necessary if the user has chosen a
        sub-daily output frequency in the configuration file.
        """
        return self._sub_output_freq

    @sub_output_freq.setter
    def sub_output_freq(self, value: int) -> None:
        """Sub output frequency.

        New variable currently for NWMv3.1 operations to properly ingest GFS 13km
        forecast data that outputs various frequencies throughout the forecast cycle
        lifetime. This variable will properly account for reading time slices of the
        forecast cycle. Currently only needed for GFS 13km operational configuration.
        Otherwise, set this value to 0.

        Example- SubOutputFreq: 0
        """
        if value < 0:
            err_out_screen(
                "Please specify an SubOutFreq that is greater than zero minutes."
            )
        if value == 0:
            value = None
        self._sub_output_freq = value

    @property
    def scratch_dir(self) -> str:
        """Specify a scratch directory that will be used for storage of temporary files.
        These files will be removed automatically by the program. at the end of the BMI
        instance. However, this directory will also store the output forcing file if
        requested by the user as well (will not be deleted in this instance).

        Example- ScratchDir: "./ScratchDir
        """
        return self._scratch_dir

    @scratch_dir.setter
    def scratch_dir(self, value: str) -> None:
        """Set the pathway to the scratch directory specified by the user in the
        configuration file. This is used to control where intermediate files are written
        during processing, and is necessary for both realtime and reforecast simulations.
        """
        self.make_scratch_dir(value)
        self._scratch_dir = value

    @property
    def useCompression(self) -> int:
        """Flag to activate scale_factor / add_offset byte packing in the output files.
            0 - Deactivate compression
            1 - Activate compression
        Only applicable in this instance when you request a netcdf output forcing file
        (Output: 1). Otherwise, just set to 0.

        Example- compressOutput: 0
        """
        return self._useCompression

    @useCompression.setter
    def useCompression(self, value: int) -> None:
        """Set the flag for whether to use compression when writing output files
        specified by the user in the configuration file. This is used to control whether
        output files are compressed, which can save disk space but may increase
        processing time.
        """
        if value is None:
            value = 0
        self.check_input_values_in_range([value], "compressOutput", [0, 1])
        self._useCompression = value

    @property
    def ana_flag(self) -> int:
        """If this is AnA run, set AnAFlag to 1, otherwise 0. Setting this flag will
        change the behavior of some Bias Correction routines as the ForecastInputOffsets
        options.

        Example- AnAFlag: 1
        """
        return self._ana_flag

    @ana_flag.setter
    def ana_flag(self, value: int) -> None:
        """Set the flag for whether to include the analysis time step in the output
        files specified by the user in the configuration file. This is used to control
        whether the analysis time step is included in the output files, which can be
        useful for certain applications but may not be necessary for all users.
        """
        value = int(value)
        self.check_input_values_in_range([value], "AnAFlag", [0, 1])
        self._ana_flag = value

    @property
    def look_back(self) -> int:
        """Specify a lookback period in minutes to process data. This is required if
        you are only processing an AnA operational configuration. This value should
        specify how far back you need to look in time from your "RefcstBDateProc" start
        date that you specified. In this instance, that start date will be your actual
        end date. If no LookBack specified, please specify -9999.

        Example- LookBack: 180
        """
        return self._look_back

    @look_back.setter
    def look_back(self, value: int) -> None:
        """Set the look back window in hours specified by the user in the configuration
        file. This is used to calculate the processing window for reforecast simulations,
        and is only necessary if the user is running a reforecast simulation with a
        specified processing window rather than a realtime simulation.
        """
        if value <= 0 and value != -9999:
            err_out_screen("Please specify a positive LookBack or -9999 for realtime.")
        # NOTE: Side effect (calculate_lookback_window) removed - now called in post_init()
        self._look_back = value

    @property
    def fcst_freq(self) -> int:
        """Specify a forecast frequency in minutes. This value specifies how often to
        generate a set of forecast forcings. If generating hourly retrospective forcings,
        specify this value to be 60.

        Example- ForecastFrequency: 60
        """
        return self._fcst_freq

    @fcst_freq.setter
    def fcst_freq(self, value: int) -> None:
        """Set the forecast frequency in hours specified by the user in the configuration file.

        This is used to calculate the processing window for reforecast simulations, and
        is only necessary if the user is running a reforecast simulation with a
        specified processing window rather than a realtime simulation.

        NOTE: this property is hardened such that it allows being set one time, but may
        not be mutated after that initial set.
        For rationale, see: https://github.com/NGWPC/ngen-forcing/pull/107
        """
        if self._fcst_freq is not None and self._fcst_freq != value:
            raise ValueError(
                f"fcst_freq is immutable after initialization. Current: {self._fcst_freq}, Attempted: {value}"
            )
        self.check_input_values_positive([value], "ForecastFrequency")
        if value > 1440:
            err_out_screen(
                "Only forecast cycles of daily or sub-daily are supported at this time"
            )
        self._fcst_freq = value

    @property
    def spatial_meta(self):
        """Specify the optional land spatial metadata file. If found, coordinate
        projection information and coordinate will be translated from to the final
        output file. This variable is only a special case if the user is specifying the
        original WRF-Hydro domain from earlier NWM versions. Otherwise, just leave the
        one blank ('').

        Example- SpatialMetaIn: ./GEOGRID_LDASOUT_Spatial_Metadata_CONUS.nc
        """
        return self._spatial_meta

    @spatial_meta.setter
    def spatial_meta(self, value: str) -> None:
        """Set the spatial metadata options specified by the user in the configuration
        file. This is used to control how spatial metadata is handled during processing,
        and is necessary for both realtime and reforecast simulations.
        """
        if value is None:
            raise TypeError(
                "spatial_meta setter received None; pass '' to indicate no spatial metadata file."
            )
        if len(value) == 0:
            # No spatial metadata file found.
            value = None
        else:
            if not os.path.isfile(value):
                err_out_screen(
                    f"Unable to locate optional spatial metadata file: {value}."
                )
        self._spatial_meta = value

    @property
    def b_date_proc(self) -> str:
        """If running an operational configuration in realtime or just using a
        retrospective dataset (NWM, AORC, ERA5), this will be the defined start date for
        the NextGen Forcing Engine BMI which is assumed to be the beginning of the
        forecast cycle (i.e. hour 0) or just the start date of the retrospective dataset.
        From there the first time step will be hour 1 from the start date specified here.
        If you're running an AnA configuration however, this variable becomes the end
        date of the simulation and the "LookBack" value specified above will be how far
        back you look in time for the AnA operational configuration.

        Example- RefcstBDateProc: 202210071400
        """
        return self._b_date_proc

    @b_date_proc.setter
    def b_date_proc(self, value: str | datetime) -> None:
        """Set the beginning date of processing for reforecast simulations. This is used
        to calculate the processing window for reforecast simulations.
        """
        if value is None:
            self._b_date_proc = None
            return
        if isinstance(value, datetime):
            self._b_date_proc = value
            return
        if value != -9999:
            if isinstance(value, str) and len(value) != 12:
                err_out_screen(
                    "Improper RefcstBDateProc length entered into the configuration file. Please check your entry."
                )
            try:
                self._b_date_proc = datetime.strptime(value, "%Y%m%d%H%M")
            except ValueError as e:
                err_out_screen(
                    "Improper RefcstBDateProc value entered into the configuration file. Please check your entry.",
                    e,
                )
        else:
            self._b_date_proc = -9999
        LOG.info(f"Begin date: {value}")

    @property
    def realtime_flag(self) -> bool:
        """Flag to indicate whether the user has chosen to run a realtime simulation,
        which will trigger some different processing pathways and error checking for
        certain configuration options, and will also control how the processing window
        is calculated.
        """
        if self.look_back == -9999:
            value = False
        elif self.b_date_proc == -9999:
            value = True
        else:
            value = False
        # NOTE: Side effect (calculate_lookback_window) removed from property getter - now called in post_init()
        return value

    @property
    def refcst_flag(self) -> bool:
        """Flag to indicate whether the user has chosen to run a reforecast simulation,
        which will trigger some different processing pathways and error checking for
        certain configuration options, and will also control how the processing window
        is calculated.
        """
        if self.look_back == -9999:
            return True
        elif self.b_date_proc == -9999:
            return True
        else:
            return False

    @property
    def geopackage(self) -> str:
        """Get the pathway to the geopackage file to be used for processing. This is
        used to specify the grid information for regridding input forcings, and is only
        necessary if the user is running a simulation that requires regridding of input
        forcings.
        """
        return self._geopackage

    @geopackage.setter
    def geopackage(self, value: str) -> None:
        """Set the pathway to the geopackage file to be used for processing. This is
        used to specify the grid information for regridding input forcings, and is only
        necessary if the user is running a simulation that requires regridding of input
        forcings.
        """
        self._geopackage = value

    @property
    def geogrid(self) -> str:
        """Specify a geogrid file (e.g. latitude, longitude, mesh connectivity,
        elevation, slope) that defines domain to which the forcings are being processed to.

        Example- GeogridIn: ./geo_em_CONUS.nc
        """
        return self._geogrid

    @geogrid.setter
    def geogrid(self, value: str) -> None:
        """Set the pathway to the geogrid file to be used for processing. This is used
        to specify the grid information for regridding input forcings, and is only
        necessary if the user is running a simulation that requires regridding of input
        forcings.
        """
        # If user provided geogrid, use it as-is
        if self.user_provided_geogrid_flag:
            self._geogrid = value
        # If value is None, just set to None
        elif value is None:
            self._geogrid = value
        # Otherwise, process the path value from config with uid prefix
        else:
            geogrid_parent = os.path.dirname(value)
            geogrid_filename = os.path.basename(value)
            if self.uid64 is None:
                raise ValueError("self.uid64 cannot be None, please initialize it.")
            self._geogrid = os.path.join(
                geogrid_parent, f"{self.uid64}_{geogrid_filename}"
            )
            self.try_make_dir(geogrid_parent, " esmf_mesh")

    def try_make_dir(self, directory: str, optional_str: str = "") -> None:
        """Try to make a directory, and catch any errors."""
        if not os.path.isdir(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                LOG.debug(f"Created{optional_str} directory: {directory}")
            except OSError as e:
                err_out_screen(
                    f"Unable to create{optional_str} directory: {directory}. Error: {e}"
                )

    @property
    def input_forcings(self) -> list:
        """Get the list of input forcing options specified by the user in the
        configuration file. This is used to control which input forcings are processed
        and how they are processed based on the other configuration options specified
        for each input forcing.
        """
        return self._input_forcings

    @input_forcings.setter
    def input_forcings(self, value: list) -> None:
        """Set the list of input forcing options specified by the user in the
        configuration file. This is used to control which input forcings are processed
        and how they are processed based on the other configuration options specified
        for each input forcing.
        """
        if value is not None and not self.precip_only_flag:
            self.check_input_values_in_range(
                value, "InputForcings", list(range(1, self.force_count + 1))
            )
        self._input_forcings = value

    @property
    def number_inputs(self) -> int:
        """Calculate the number of input forcing options specified by the user in the
        configuration file. This is used for error checking to ensure users specify
        valid input forcing options in the configuration file, and to control the flow
        of the program based on how many input forcings are being processed.
        """
        if self.input_forcings is None:
            return 0
        if not self.precip_only_flag:
            if len(self.input_forcings) == 0:
                err_out_screen(
                    "Please choose at least one InputForcings dataset to process"
                )
            return len(self.input_forcings)
        return 0

    @property
    def number_custom_inputs(self) -> int:
        """Calculate the number of custom input forcing options specified by the user
        in the configuration file. This is used to control the flow of the program based
        on how many custom input forcings are being processed, since custom input
        forcings require some different processing pathways.
        """
        if not self.precip_only_flag:
            count = 0
            for force_opt in self.input_forcings:
                if force_opt == 10:
                    count += 1
            return count
        else:
            return 0

    @number_custom_inputs.setter
    def number_custom_inputs(self, value: int) -> None:
        """This is a read-only computed property based on input_forcings."""
        raise AttributeError(
            f"number_custom_inputs is read-only (tried to set to: {value})"
        )

    @property
    def nwm_geogrid(self) -> str:
        """Only for the NWM v3 retorspective forcing module option (27) that requires
        the geo_em_NWM_DOMAIN.nc file as input for the NextGen Forcings Engine to
        properly setup up the ESMF grid object for the NWM forcing files since that
        information is not readily available in the NWM v3 retrospective forcing files.
        """
        return self._nwm_geogrid

    @nwm_geogrid.setter
    def nwm_geogrid(self, value: str) -> None:
        """Set the pathway to the NWM geogrid file specified by the user in the
        configuration file. This is used to specify the grid information for regridding
        NWM input forcings, and is only necessary if the user has chosen to regrid NWM
        input forcings in the configuration file.
        """
        if (
            not self.precip_only_flag
            and self.input_forcings is not None
            and 27 in self.input_forcings
        ):
            self._nwm_geogrid = value
        else:
            self._nwm_geogrid = None

    @property
    def input_force_types(self) -> list:
        """Get the list of input forcing file types specified by the user in the
        configuration file. This is used to control how input forcings are read in and
        processed based on the file type specified for each input forcing in the
        configuration file.
        """
        return self._input_force_types

    @input_force_types.setter
    def input_force_types(self, value: list) -> None:
        """Specify the file type for each forcing (comma separated).
        Valid types are GRIB1, GRIB2, NETCDF, and NETCDF4.
        
        Example- InputForcingTypes: [GRIB2,GRIB2]\
        """
        if not self.precip_only_flag:
            if value == [""]:
                value = []
            self.check_number_of_inputs_forcings(value, "InputForcingTypes")
            self.check_input_values_in_range(
                value, "InputForcingTypes", self.file_types
            )
        self._input_force_types = value

    @property
    def file_types(self):
        """Get the list of input forcing file types specified by the user in the
        configuration file. This is used to control how input forcings are read in and
        processed based on the file type specified for each input forcing in the
        configuration file.
        """
        return CONFIGOPTIONS["file_types"]

    @property
    def input_force_dirs(self) -> list:
        """Get the list of input forcing directories specified by the user in the
        configuration file. This is used to control where input forcings are read in
        from for each input forcing specified by the user in the configuration file.
        """
        if self._input_force_dirs:
            return self._input_force_dirs
        return None

    @input_force_dirs.setter
    def input_force_dirs(self, value: list) -> None:
        """Specify the input directories for each forcing product. If a user has the
        ability to connect to the AWS servers and they specify configuration #12
        (CONUS AORC data) or configuration #27 (NWM retrospective forcing data) then
        this specific configuration input can be left as a blank string ("").

        Example- InputForcingDirectories: [./GFS,./NDFD]
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "InputForcingDirectories")
            # Loop through and ensure all input directories exist. Also strip out any whitespace
            # or new line characters.
            for dir_tmp in range(0, len(value)):
                value[dir_tmp] = value[dir_tmp].strip()
                dir_path = value[dir_tmp]
                forcing_type = self.input_forcings[dir_tmp]
                is_aws_forcing = forcing_type in [12, 21, 27]

                if not os.path.isdir(dir_path):
                    if is_aws_forcing:
                        self.aws = True
                    else:
                        self.try_make_dir(dir_path, " forcing")
        self._input_force_dirs = value

    @property
    def input_force_mandatory(self) -> list:
        """Get the list of input forcing mandatory flags specified by the user in the
        configuration file. This is used to control whether the program should raise an
        error if input forcings for a given forecast cycle are not found for each input
        forcing specified by the user in the configuration file.
        """
        return self._input_force_mandatory

    @input_force_mandatory.setter
    def input_force_mandatory(self, value: list) -> None:
        """Specify whether the input forcings listed above are mandatory, or optional.
        This is important for layering contingencies if a product is missing, but forcing
        files are still desired. 0 - Not mandatory, 1 - Mandatory.
        NOTE!!! If no files are found for any products, code will error out indicating
        the final field is all missing values.

        Example- InputMandatory: [1,1]
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "InputMandatory")
            self.check_input_values_in_range(value, "InputMandatory", [0, 1])
        self._input_force_mandatory = value

    @property
    def customSuppPcpFreq(self) -> int:
        """Get the custom supplemental precip output frequency specified by the user in
        the configuration file. This is used to control the output frequency of
        supplemental precip forcings if the user has chosen to run the supplemental
        precip forcings module only.
        """
        return self._customSuppPcpFreq

    @customSuppPcpFreq.setter
    def customSuppPcpFreq(self, value: int) -> None:
        """Set the custom supplemental precip output frequency specified by the user in
        the configuration file. This is used to control the output frequency of
        supplemental precip forcings if the user has chosen to run the supplemental
        precip forcings module only.
        """
        if self.precip_only_flag:
            self.check_input_values_non_negative([value], "customSuppPcpFreq")
            self._customSuppPcpFreq = value
        else:
            self._customSuppPcpFreq = None

    @property
    def fcst_shift(self) -> int:
        """Forecast cycles are determined by splitting up a day by equal ForecastFrequency
        interval. If there is a desire to shift the cycles to a different time step,
        ForecastShift will shift forecast cycles ahead by a determined set of minutes.
        For example, ForecastFrequency of 6 hours will produce forecasts cycles at 00, 06, 12, and 18 UTC.
        However, a ForecastShift of 1 hour will produce forecast cycles at 01, 07, 13, and 18 UTC.
        NOTE - This is only used by the realtime instance to calculate forecast cycles accordingly.
        Re-forecasts will use the beginning and ending dates specified in conjunction
        with the forecast frequency to determine forecast cycle dates.

        Example- ForecastShift: 0
        """
        return self._fcst_shift

    @fcst_shift.setter
    def fcst_shift(self, value: int) -> None:
        if True:  # was: self.realtime_flag:
            self.check_input_values_non_negative([value], "ForecastShift")
            # NOTE: Side effect (calculate_lookback_window) removed - now called in post_init()
            self._fcst_shift = value

            # NOTE this commented out code copied from pre-refactored code on 5/6/2026
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

    @property
    def nFcsts(self):
        """Get the number of forecasts to issue for a reforecast simulation based on the
        forecast shift and the processing window specified by the user in the
        configuration file. This is used to control how many forecast time steps are
        output for a reforecast simulation, and is only necessary if the user is running
        a reforecast simulation with a specified processing window rather than a
        realtime simulation.
        """
        return self._nFcsts

    @nFcsts.setter
    def nFcsts(self, value: int) -> None:
        """Set the number of forecasts to issue for a reforecast simulation based on the
        forecast shift and the processing window specified by the user in the
        configuration file. This is used to control how many forecast time steps are
        output for a reforecast simulation, and is only necessary if the user is running
        a reforecast simulation with a specified processing window rather than a realtime
        simulation.
        """
        if value is None:
            value = 1
        self._nFcsts = value

    @property
    def fcst_input_horizons(self) -> list:
        """Specify how much (in minutes) of each input forcing is desires for each
        forecast cycle. See documentation for examples. The length of this array must
        match the input forcing choices.

        - Example- ForecastInputHorizons: [60, 60]
        """
        return self._fcst_input_horizons

    @fcst_input_horizons.setter
    def fcst_input_horizons(self, value: list) -> None:
        """Setter for ``fcst_input_horizons``.

        NOTE: this property is hardened such that it allows being set one time, but may
        not be mutated after that initial set.
        For rationale, see: https://github.com/NGWPC/ngen-forcing/pull/107
        """
        if self._fcst_input_horizons is not None and self._fcst_input_horizons != value:
            raise ValueError(
                f"fcst_input_horizons is immutable after initialization. Current: {self._fcst_input_horizons}, Attempted: {value}"
            )
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "ForecastInputHorizons")
            self.check_input_values_non_negative(value, "ForecastInputHorizons")
        else:
            if value is not None and len(value) != 1:
                err_out_screen(
                    "Please specify ForecastInputHorizon values for each corresponding input forcings for forecasts."
                )
        self._fcst_input_horizons = value

    @property
    def fcst_input_offsets(self):
        """Option for applying an offset to input forcings to use a different forecasted
        interval. For example, a user may wish to use 4-5 hour forecasted fields from an
        NWP grid from one of their input forcings. In that instance the offset would be
        4 hours, but 0 for other remaining forcings.

        Example- ForecastInputOffsets: [0, 0]
        """
        return self._fcst_input_offsets

    @fcst_input_offsets.setter
    def fcst_input_offsets(self, value: list) -> None:
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "ForecastInputOffsets")
            self.check_input_values_non_negative(value, "ForecastInputOffsets")
            self._fcst_input_offsets = value

    @property
    def cycle_length_minutes(self) -> int:
        """Get the forecast cycle length in minutes, which is calculated based on the
        maximum of the forecast input horizons specified by the user in the configuration
        file.

        Ensure the number maximum cycle length is an equal divider of the output time step specified by the user.
        """
        cycle_len = max(self.fcst_input_horizons)
        if cycle_len % self.output_freq != 0:
            err_out_screen(
                "Please specify an output time step that is an equal divider of the maximum of the forecast time horizons specified."
            )
        return cycle_len

    @property
    def num_output_steps(self) -> int:
        """Calculate the number of output time steps per forecast cycle based on the
        forecast cycle length and the output frequency specified by the user in the
        configuration file.
        """
        if self.sub_output_hour is None:
            num_steps = int(self.cycle_length_minutes / self.output_freq)
        else:
            num_steps = (
                int(
                    (self.cycle_length_minutes - (self.sub_output_hour * 60))
                    / self.sub_output_freq
                )
                + int((self.sub_output_hour * 60) / self.output_freq)
                - 1
            )
        return num_steps

    @property
    def num_supp_output_steps(self) -> int:
        """Calculate the number of supplemental precip output time steps per forecast
        cycle based on the forecast cycle length and the custom supplemental precip
        output frequency specified by the user in the configuration file.
        """
        if self.precip_only_flag:
            return int(self.cycle_length_minutes / self.customSuppPcpFreq)

    @property
    def actual_output_steps(self) -> int:
        """Calculate the actual number of output time steps per forecast cycle based on
        whether the user has chosen to run a reforecast simulation with a specified
        processing window, which will only output time steps for which input forcings
        are available based on the processing window and forecast time horizons
        specified by the user in the configuration file.
        """
        if self.ana_flag:
            return np.int32(self.nFcsts)
        else:
            return np.int32(self.num_output_steps)

    @property
    def grid_type(self) -> str:
        """Tells the NextGen Forcings Engine BMI which grid type the engine is
        initalizing as a BMI instance. This is a required field and the proper string
        values should be "gridded", "hydrofabric", or "unstructured".

        Example- GRID_TYPE: "gridded"
        """
        return self._grid_type

    @grid_type.setter
    def grid_type(self, value: str) -> None:
        """Set the grid type specified by the user in the configuration file. This is
        used to control how the program reads in and processes the geogrid information
        for regridding input forcings based on the grid type specified by the user in
        the configuration file.
        """
        self.check_input_values_in_range(
            [value.lower()], "GRID_TYPE", ["gridded", "unstructured", "hydrofabric"]
        )
        self._grid_type = value.lower()

    @property
    def lon_var(self) -> str:
        """Naming convention of the longitude variable within the "GeogridIn" file the
        user has specified. Variable naming convention ONLY for gridded domain
        configurations. This is required so the NextGen Forcings Engine BMI can
        dyanmically initialize the domain geogrid as an ESMF regridding object. In the
        case for "gridded" domain configuration options and a user specifying downscaling
        options while only specifying a height variable feature on the grid, this netcdf
        variable (LONVAR) is then EXPECTED to contain a netcdf metadata attribute called
        "dx" that specifies the grid spacing in the longtiudinal direction. Otherwise,
        it will throw an error and not be able to calculate the slope and tilt of each
        grid cell.

        Example- LONVAR: "XLONG_M"
        """
        if self.grid_type == "gridded":
            return self.extract_input_variable("LONVAR")

    @property
    def lat_var(self) -> str:
        """Naming convention of the latitude variable within the "GeogridIn" file the
        user has specified. Variable naming convention ONLY for gridded domain
        configurations. This is required so the NextGen Forcings Engine BMI can
        dyanmically initialize the domain geogrid as an ESMF regridding object. In the
        case for "gridded" domain configuration options and a user specifying
        downscaling options while only specifying a height variable feature on the grid,
        this netcdf variable (LATVAR) is then EXPECTED to contain a netcdf metadata
        attribute called "dy" that specifies the grid spacing in the latitudinal
        direction. Otherwise, it will throw an error and not be able to calculate the
        slope and tilt of each grid cell.

        Example- LATVAR: "XLAT_M"
        """
        if self.grid_type == "gridded":
            return self.extract_input_variable("LATVAR")

    @property
    def nodecoords_var(self) -> str:
        """Naming convention of the node coordinates variable within the "GeogridIn"
        file the user has specified for ONLY an unstructured mesh or the NextGen hydrofabric.
        This is a 2-D array stating the latitude and longitude coordinates for all the nodes in the mesh.
        This is required so the NextGen Forcings Engine BMI can dyanmically initialize the domain geogrid as an ESMF regridding object.

        Example- NodeCoods: "nodecoords"
        """
        if self.grid_type in ["unstructured", "hydrofabric"]:
            return self.extract_input_variable("NodeCoords")

    @property
    def elemcoords_var(self) -> str:
        """Naming convention of the element coordinates variable within the "GeogridIn"
        file the user has specified for ONLY an unstructured mesh or the NextGen hydrofabric.
        This is a 2-D array stating the latitude and longitude coordinates for all the elements in the mesh.
        This is required so the NextGen Forcings Engine BMI can dyanmically initialize the domain geogrid as an ESMF regridding object.

        Example- ElemCoods: "elemcoords"
        """
        if self.grid_type in ["unstructured", "hydrofabric"]:
            return self.extract_input_variable("ElemCoords")

    @property
    def elemconn_var(self) -> str:
        """Naming convention of the element connectivity variable within the "GeogridIn"
        file the user has specified for ONLY an unstructured mesh or the NextGen hydrofabric.
        This is a 2-D array stating the node ids for each element connecting the entire mesh structure.
        This is required so the NextGen Forcings Engine BMI can dyanmically initialize the domain geogrid as an ESMF regridding object.

        Example- ElemConn: "elemconn"
        """
        if self.grid_type in ["unstructured", "hydrofabric"]:
            return self.extract_input_variable("ElemConn")

    @property
    def numelemconn_var(self) -> str:
        """Naming convention of the number of nodes per element variable within the "GeogridIn"
        file the user has specified for ONLY an unstructured mesh or the NextGen hydrofabric.
        This is a 1-D array stating the how many nodes are connecting each element within the unstructured mesh.
        This is required so the NextGen Forcings Engine BMI can dyanmically initialize the domain geogrid as an ESMF regridding object.

        Example- NumElemConn: "numelemconn"
        """
        if self.grid_type in ["unstructured", "hydrofabric"]:
            return self.extract_input_variable("NumElemConn")

    @property
    def element_id_var(self) -> str:
        """Naming convention of the element id variable within the "GeogridIn"
        file the user has specified for ONLY the NextGen hydrofabric.
        This is a 1-D array stating the catchment id numeric naming convention within the "divides" geopackage layer of a given NextGen hydrofabric file.
        This variable is required in order for the NextGen Forcings Engine to properly advertise the element ids of the unstructured mesh linked to the NextGen hydrofabric catchment ids.

        Example- ElemID: "element_ids"
        """
        if self.grid_type == "hydrofabric":
            return self.extract_input_variable("ElemID")

    @property
    def ignored_border_widths(self) -> list:
        """Border width (in grid cells) to ignore for each input dataset.
        NOTE: generally, the first input forcing should always be zero or there will be missing data in the final output.

        Example- IgnoredBorderWidths: [0,10]
        """
        return self._ignored_border_widths

    @ignored_border_widths.setter
    def ignored_border_widths(self, value: list) -> None:
        """Set the list of ignored border widths specified by the user in the configuration file.
        This is used to control how the program processes input forcings based on the
        ignored border widths specified for each input forcing in the configuration file.
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "IgnoredBorderWidths")
            self.check_input_values_non_negative(value, "IgnoredBorderWidths")
            self._ignored_border_widths = value

    @property
    def regrid_opt(self):
        """Choose regridding options for each input forcing files being used.
        Options available are:
        1 - ESMF Bilinear,
        2 - ESMF Nearest Neighbor,
        3 - ESMF Conservative Bilinear.

        Example- RegridOpt: [1,1]
        """
        return self._regrid_opt

    @regrid_opt.setter
    def regrid_opt(self, value: list) -> None:
        """Set the list of regridding options specified by the user in the configuration file.
        This is used to control how input forcings are regridded based on the regridding
        option specified for each input forcing in the configuration file.
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "RegridOpt")
            self.check_input_values_in_range(value, "RegridOpt", [1, 2, 3])
            self._regrid_opt = value
        else:
            self._regrid_opt = None

    @property
    def weightsDir(self) -> str:
        """Get the pathway to the ESMF weights directory specified by the user in the configuration file.
        This is used to control where the program looks for ESMF weights files if the
        user has chosen to use pre-generated ESMF weights files for regridding input
        forcings in the configuration file.
        """
        return self._weightsDir

    @weightsDir.setter
    def weightsDir(self, value: str) -> None:
        """Set the pathway to the ESMF weights directory specified by the user in the configuration file.
        This is used to control where the program looks for ESMF weights files if the
        user has chosen to use pre-generated ESMF weights files for regridding input
        forcings in the configuration file.
        """
        if not self.precip_only_flag:
            if value is not None and not os.path.exists(value):
                err_out_screen(
                    f"ESMF Weights file directory specified ({value}) but does not exist"
                )
            self._weightsDir = value

    @property
    def forceTemoralInterp(self) -> list:
        """Get the list of forcing temporal interpolation options specified by the user in the configuration file.
        This is used to control how input forcings are temporally interpolated based on
        the temporal interpolation option specified for each input forcing in the
        configuration file.
        """
        return self._forceTemoralInterp

    @forceTemoralInterp.setter
    def forceTemoralInterp(self, value: list) -> None:
        """Specify an temporal interpolation for the forcing variables. Interpolation
        will be done between the two neighboring input forcing states that exist.
        If only one nearest state exist (I.E. only a state forward in time, or behind),
        then that state will be used as a "nearest neighbor".
        NOTE - All input options here must be of the same length of the input forcing number.
        Also note all temporal interpolation occurs BEFORE downscaling and bias correction.
            0 - No temporal interpolation.
            1 - Nearest Neighbor,
            2 - Linear weighted average.

        Example- ForcingTemporalInterpolation: [0,0]
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "ForcingTemporalInterpolation")
            self.check_input_values_in_range(
                value, "ForcingTemporalInterpolation", [0, 1, 2]
            )
            self._forceTemoralInterp = value

    @property
    def t2dDownscaleOpt(self) -> list:
        """Specify a temperature downscaling method:
            0 - No downscaling,
            1 - Use a simple lapse rate of 6.75 degrees Celsius to get from the model elevation to the WRF-Hydro elevation,
            2 - Use a pre-calculated lapse rate regridded to the WRF-Hydro domain (only NWM),
            3 - Use a dynamic lapse rate calculated at each timstep.

        Example- TemperatureDownscaling: [3, 3]
        """
        return self._t2dDownscaleOpt

    @t2dDownscaleOpt.setter
    def t2dDownscaleOpt(self, value: list) -> None:
        """Set the list of temperature downscaling options specified by the user in the configuration file.
        This is used to control how temperature input forcings are downscaled based on
        the temperature downscaling option specified for each input forcing in the
        configuration file.
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "TemperatureDownscaling")
            self.check_input_values_in_range(value, "TemperatureDownscaling", [0, 1, 2])
            count = 0
            for opt in value:
                if opt == 2:
                    self.param_flag[count] = 1
                count += 1
            self._t2dDownscaleOpt = value

    @property
    def psfcDownscaleOpt(self) -> list:
        """Specify a surface pressure downscaling method:
            0 - No downscaling,
            1 - Use input elevation and WRF-Hydro elevation to downscale surface pressure.

        Example- PressureDownscaling: [1, 1]
        """
        return self._psfcDownscaleOpt

    @psfcDownscaleOpt.setter
    def psfcDownscaleOpt(self, value: list) -> None:
        """Set the list of pressure downscaling options specified by the user in the configuration file.
        This is used to control how pressure input forcings are downscaled based on the
        pressure downscaling option specified for each input forcing in the configuration file.
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "PressureDownscaling")
            self.check_input_values_in_range(value, "PressureDownscaling", [0, 1])
            self._psfcDownscaleOpt = value

    @property
    def swDownscaleOpt(self) -> list:
        """Specify a shortwave radiation downscaling routine.
            0 - No downscaling,
            1 - Run a topographic adjustment using the WRF-Hydro elevation.

        Example- ShortwaveDownscaling: [1, 1]
        """
        return self._swDownscaleOpt

    @swDownscaleOpt.setter
    def swDownscaleOpt(self, value: list) -> None:
        """Set the list of shortwave downscaling options specified by the user in the configuration file. This is used to control how shortwave radiation input forcings are downscaled based on the shortwave downscaling option specified for each input forcing in the configuration file."""
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "ShortwaveDownscaling")
            self.check_input_values_in_range(value, "ShortwaveDownscaling", [0, 1])
            self._swDownscaleOpt = value

    @property
    def q2dDownscaleOpt(self) -> list:
        """Specify a specific humidity downscaling routine.
            0 - No downscaling,
            1 - Use regridded humidity, along with downscaled temperature/pressure
                to extrapolate a downscaled surface specific humidty.

        Example- HumidityDownscaling: [1, 1]
        """
        return self._q2dDownscaleOpt

    @q2dDownscaleOpt.setter
    def q2dDownscaleOpt(self, value: list) -> None:
        """Set the list of humidity downscaling options specified by the user in the configuration file.
        This is used to control how humidity input forcings are downscaled based on the
        humidity downscaling option specified for each input forcing in the configuration file.
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "HumidityDownscaling")
            self.check_input_values_in_range(value, "HumidityDownscaling", [0, 1])
            self._q2dDownscaleOpt = value

    @property
    def precipDownscaleOpt(self) -> list:
        """Specify a precipitation downscaling routine.
            0 - No downscaling,
            1 - NWM mountain mapper downscaling using monthly PRISM climo.

        Example- PrecipDownscaling: [0, 0]
        """
        return self._precipDownscaleOpt

    @precipDownscaleOpt.setter
    def precipDownscaleOpt(self, value: list) -> None:
        """Set the list of precipitation downscaling options specified by the user in the configuration file.
        This is used to control how precipitation input forcings are downscaled based on
        the precipitation downscaling option specified for each input forcing in the configuration file.
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "PrecipDownscaling")
            self.check_input_values_in_range(value, "PrecipDownscaling", [0, 1])
            count = 0
            for opt in value:
                if opt == 1:
                    self.param_flag[count] = 1
                count += 1
            self._precipDownscaleOpt = value
        else:
            self._precipDownscaleOpt = None

    @property
    def dScaleParamDirs(self) -> list:
        """Specify the input parameter directory containing necessary downscaling grids.
        This is ONLY needed for the original NWM WRF-Hydro domain.
        Otherwise, just point it to a random directory and it will be ignored.

        Example- DownscalingParamDirs: ["./forcingParam/AnA", "./forcingParam/AnA"]
        """
        return self._dScaleParamDirs

    @dScaleParamDirs.setter
    def dScaleParamDirs(self, value: list) -> None:
        """Set the list of downscaling parameter directories specified by the user in the configuration file.
        This is used to control where the program looks for downscaling parameter
        files for each input forcing based on the downscaling parameter directory
        specified for each input forcing in the configuration file.

        NOTE: The guard on ``precip_only_flag`` is because DownscalingParamDirs is omitted
        from precip-only configurations.
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "DownscalingParamDirs")
            for dirTmp in range(0, len(value)):
                dir_path = value[dirTmp]
                if not os.path.isdir(dir_path):
                    err_out_screen(
                        f"Unable to locate parameter directory: {os.path.abspath(dir_path)}"
                    )
            self._dScaleParamDirs = value
        else:
            self._dScaleParamDirs = None

    @property
    def perform_downscaling(self) -> bool:
        """Determine whether downscaling of input forcings is necessary based on the
        downscaling options specified by the user for each input forcing in the configuration file.
        """
        if self.precip_only_flag:
            return False
        if (
            1 in self.q2dDownscaleOpt
            or 1 in self.swDownscaleOpt
            or 1 in self.psfcDownscaleOpt
            or 1 in self.t2dDownscaleOpt
            or 2 in self.t2dDownscaleOpt
        ):
            return True
        else:
            return False

    @property
    def t2BiasCorrectOpt(self) -> list:
        """Specify a temperature bias correction method.
            0 - No bias correction,
            1 - CFSv2 - NLDAS2 Parametric Distribution - NWM ONLY,
            2 - Custom NCAR bias-correction based on HRRRv3 analysis - based on hour of day (USE WITH CAUTION),
            3 - NCAR parametric GFS bias correction,
            4 - NCAR parametric HRRR bias correction.

        Example- TemperatureBiasCorrection: [0, 4]
        """
        return self._t2BiasCorrectOpt

    @t2BiasCorrectOpt.setter
    def t2BiasCorrectOpt(self, value: list) -> None:
        """Set the list of temperature bias correction options specified by the user in the configuration file.
        This is used to control how temperature input forcings are bias corrected based
        on the temperature bias correction option specified for each input forcing in
        the configuration file.
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "TemperatureBiasCorrection")
            self.check_input_values_in_range(
                value, "TemperatureBiasCorrection", [0, 1, 2, 3, 4]
            )
            self._t2BiasCorrectOpt = value

    @property
    def psfcBiasCorrectOpt(self) -> list:
        """Specify a surface pressure bias correction method.
            0 - No bias correction,
            1 - CFSv2 - NLDAS2 Parametric Distribution - NWM ONLY.

        Example- PressureBiasCorrection: [0,0]
        """
        return self._psfcBiasCorrectOpt

    @psfcBiasCorrectOpt.setter
    def psfcBiasCorrectOpt(self, value: list) -> None:
        """Set the list of pressure bias correction options specified by the user in the configuration file.
        This is used to control how pressure input forcings are bias corrected based on
        the pressure bias correction option specified for each input forcing in the
        configuration file.
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "PressureBiasCorrection")
            self.check_input_values_in_range(value, "PressureBiasCorrection", [0, 1])
            self._psfcBiasCorrectOpt = value

    @property
    def q2BiasCorrectOpt(self):
        """Specify a specific humidity bias correction method.
            0 - No bias correction,
            1 - CFSv2 - NLDAS2 Parametric Distribution - NWM ONLY,
            2 - Custom NCAR bias-correction based on HRRRv3 analysis - based on hour of day (USE WITH CAUTION).

        Example- HumidityBiasCorrection: [0,0]
        """
        return self._q2BiasCorrectOpt

    @q2BiasCorrectOpt.setter
    def q2BiasCorrectOpt(self, value):
        """Set the list of humidity bias correction options specified by the user in the configuration file.
        This is used to control how humidity input forcings are bias corrected based on
        the humidity bias correction option specified for each input forcing in the
        configuration file.
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "HumidityBiasCorrection")
            self.check_input_values_in_range(value, "HumidityBiasCorrection", [0, 1, 2])
            self._q2BiasCorrectOpt = value

    @property
    def windBiasCorrect(self):
        """Specify a wind bias correction.
            0 - No bias correction,
            1 - CFSv2 - NLDAS2 Parametric Distribution - NWM ONLY,
            2 - Custom NCAR bias-correction based on HRRRv3 analysis - based on hour of day (USE WITH CAUTION),
            3 - NCAR parametric GFS bias correction,
            4 - NCAR parametric HRRR bias correction.

        Example- WindBiasCorrection: [0, 4]
        """
        return self._windBiasCorrect

    @windBiasCorrect.setter
    def windBiasCorrect(self, value):
        """Set the list of wind bias correction options specified by the user in the configuration file.
        This is used to control how wind input forcings are bias corrected based on the
        wind bias correction option specified for each input forcing in the configuration file.
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "WindBiasCorrection")
            self.check_input_values_in_range(
                value, "WindBiasCorrection", [0, 1, 2, 3, 4]
            )
            self._windBiasCorrect = value

    @property
    def swBiasCorrectOpt(self) -> list:
        """Specify a bias correction for incoming short wave radiation flux.
            0 - No bias correction,
            1 - CFSv2 - NLDAS2 Parametric Distribution - NWM ONLY,
            2 - Custom NCAR bias-correction based on HRRRv3 analysis (USE WITH CAUTION).

        Example- SwBiasCorrection: [0, 2]
        """
        return self._swBiasCorrectOpt

    @swBiasCorrectOpt.setter
    def swBiasCorrectOpt(self, value: list) -> None:
        """Set the list of shortwave radiation bias correction options specified by the user in the configuration file. This is used to control how shortwave radiation input forcings are bias corrected based on the shortwave radiation bias correction option specified for each input forcing in the configuration file."""
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "SwBiasCorrection")
            self.check_input_values_in_range(value, "SwBiasCorrection", [0, 1, 2])
            self._swBiasCorrectOpt = value

    @property
    def lwBiasCorrectOpt(self) -> list:
        """Specify a bias correction for incoming long wave radiation flux.
            0 - No bias correction,
            1 - CFSv2 - NLDAS2 Parametric Distribution - NWM ONLY,
            2 - Custom NCAR bias-correction based on HRRRv3 analysis, blanket adjustment (USE WITH CAUTION),
            3 - NCAR parametric GFS bias correction.

        Example- LwBiasCorrection: [0, 2]
        """
        return self._lwBiasCorrectOpt

    @lwBiasCorrectOpt.setter
    def lwBiasCorrectOpt(self, value: list) -> None:
        """Set the list of longwave radiation bias correction options specified by the user in the configuration file.
        This is used to control how longwave radiation input forcings are bias corrected
        based on the longwave radiation bias correction option specified for each input
        forcing in the configuration file.
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "LwBiasCorrection")
            self.check_input_values_in_range(value, "LwBiasCorrection", [0, 1, 2, 3, 4])
            self._lwBiasCorrectOpt = value

    @property
    def precipBiasCorrectOpt(self):
        """Specify a bias correction for precipitation.
            0 - No bias correction,
            1 - CFSv2 - NLDAS2 Parametric Distribution - NWM ONLY.

        Example- PrecipBiasCorrection: [0, 0]
        """
        return self._precipBiasCorrectOpt

    @precipBiasCorrectOpt.setter
    def precipBiasCorrectOpt(self, value):
        """Set the list of precipitation bias correction options specified by the user in the configuration file.
        This is used to control how precipitation input forcings are bias corrected
        based on the precipitation bias correction option specified for each input
        forcing in the configuration file.
        """
        if not self.precip_only_flag:
            self.check_number_of_inputs_forcings(value, "PrecipBiasCorrection")
            self.check_input_values_in_range(value, "PrecipBiasCorrection", [0, 1])
            self._precipBiasCorrectOpt = value

    @property
    def bias_correction_properties(self) -> dict:
        """Get the dictionary of bias correction properties specified by the user in the configuration file.
        This is used to control how input forcings are bias corrected based on the bias
        correction options specified for each input forcing in the configuration file.

        # TODO "surface temperature" was excluded from this consideration in the orignal code (5/7/2026 pre-refactor). Should it actually be included?
        """
        return {
            # "surface temperature": self.t2BiasCorrectOpt,
            "surface pressure": self.psfcBiasCorrectOpt,
            "specific humidity": self.q2BiasCorrectOpt,
            "wind forcings": self.windBiasCorrect,
            "short-wave radiation": self.swBiasCorrectOpt,
            "long-wave radiation": self.lwBiasCorrectOpt,
            "Precipitation": self.precipBiasCorrectOpt,
        }

    @property
    def runCfsNldasBiasCorrect(self) -> bool:
        """Get the flag for whether to run the NWM-specific bias correction of CFSv2
        input forcings specified by the user in the configuration file. This is used to
        control whether the NWM-specific bias correction of CFSv2 input forcings is run
        based on whether the user has chosen to run this bias correction in the
        configuration file.

        NOTE: Returns False immediately when precip_only_flag is truthy because
        bias correction is never configured for precip-only runs. In that case the
        correction backing vars are None and iterating them would raise TypeError.
        """
        run_cfs_nldas_bias_correct = False
        if self.precip_only_flag:
            return run_cfs_nldas_bias_correct
        for bias_option in self.bias_correction_properties.values():
            for opt in bias_option:
                if opt == 1:
                    run_cfs_nldas_bias_correct = True
                    break
        if run_cfs_nldas_bias_correct:
            for (
                bias_correct_name,
                bias_correct,
            ) in self.bias_correction_properties.items():
                if min(bias_correct) != 1 and max(bias_correct) != 1:
                    err_out_screen(
                        f"CFSv2-NLDAS NWM bias correction must be activated for {bias_correct_name} under this configuration."
                    )
                # Make sure we don't have any other forcings activated. This can only be ran for CFSv2.
                for opt_tmp in self.input_forcings:
                    if opt_tmp != 7:
                        err_out_screen(
                            "CFSv2-NLDAS NWM bias correction can only be used in CFSv2-only configurations"
                        )
        return run_cfs_nldas_bias_correct

    @property
    def number_supp_pcp(self) -> int:
        """Get the number of supplemental precipitation input forcings specified by the
        user in the configuration file. This is used to control how many supplemental
        precipitation input forcings are processed based on the number of supplemental
        precipitation input forcings specified in the configuration file.
        """
        if self.supp_precip_forcings is None:
            return 0
        return len(self.supp_precip_forcings)

    @property
    def supp_precip_file_types(self) -> list:
        """Get the list of supplemental precipitation input forcing file types
        specified by the user in the configuration file. This is used to control how
        supplemental precipitation input forcing files are read in and processed based
        on the file types specified for each supplemental precipitation input forcing
        in the configuration file.
        """
        return self._supp_precip_file_types

    @supp_precip_file_types.setter
    def supp_precip_file_types(self, value: list) -> None:
        """Set the list of supplemental precipitation input forcing file types
        specified by the user in the configuration file. This is used to control how
        supplemental precipitation input forcing files are read in and processed based
        on the file types specified for each supplemental precipitation input forcing
        in the configuration file.
        """
        if value is not None:
            value = [stype.strip() for stype in value]
        if value == [""]:
            value = []
        self.check_number_of_inputs_supp_pcp(value, "SuppPcpForcingTypes")
        self.check_input_values_in_range(
            value,
            "SuppPcpForcingTypes",
            self.supplemental_precip_file_type_options,
        )
        self._supp_precip_file_types = value

    @property
    def supplemental_precip_file_type_options(self) -> list:
        """Get the list of valid supplemental precipitation input forcing file types
        that can be specified by the user in the configuration file. This is used to
        control how supplemental precipitation input forcing files are read in and
        processed based on the file types specified for each supplemental precipitation
        input forcing in the configuration file.
        """
        return ["GRIB1", "GRIB2", "NETCDF"]

    @property
    def rqiMethod(self) -> int | list[int] | None:
        """Optional RQI method for radar-based data. 0 - Do not use any RQI filtering.
        Use all radar-based estimates.
            1 - Use hourly MRMS Radar Quality Index grids,
            2 - Use NWM monthly climatology grids (NWM only!!!!).

        Example- RqiMethod: 2
        """
        value = None
        if self.number_supp_pcp > 0:
            for suppOpt in self.supp_precip_forcings:
                # Read in RQI threshold to apply to radar products.
                if suppOpt in (1, 2, 7, 10, 11, 12):
                    # Returns None if key missing (not configured for this product)
                    value = self.cfg_bmi.get("RqiMethod")
                    if value is not None:
                        # Validate
                        if type(value) is list:
                            self.check_number_of_inputs_supp_pcp(value, "RqiMethod")
                        elif type(value) in (int, type(None)):
                            # Support configuration file representing this with a single scalar value, apply to all
                            value = [value] * self.number_supp_pcp

                        self.check_input_values_in_range(value, "RqiMethod", [0, 1, 2])
        return value

    @property
    def rqiThresh(self) -> float | list[float] | None:
        """Optional RQI threshold to be used to mask out. Currently used for MRMS products.
        Please choose a value from 0.0-1.0. Associated radar quality index files will
        be expected from MRMS data.

        Example- RqiThreshold: 0.9
        """
        value = None
        if self.number_supp_pcp > 0:
            for supp_opt in self.supp_precip_forcings:
                # Read in RQI threshold to apply to radar products.
                if supp_opt in (1, 2, 7, 10, 11, 12):
                    # Returns None if key missing (not configured for this product)
                    value = self.cfg_bmi.get("RqiThreshold")
                    if value is not None:
                        # Validate
                        if type(value) is list:
                            self.check_number_of_inputs_supp_pcp(value, "RqiThreshold")
                        elif type(value) in (int, float, type(None)):
                            # Support configuration file representing this with a single scalar value, apply to all
                            value = [value] * self.number_supp_pcp

                        for threshold in value:
                            if threshold < 0.0 or threshold > 1.0:
                                err_out_screen(
                                    "Please specify RqiThresholds between 0.0 and 1.0."
                                )
        return value

    @property
    def supp_precip_mandatory(self):
        """Specify whether the Supplemental Precips listed above are mandatory, or optional.
        This is important for layering contingencies if a product is missing, but
        forcing files are still desired.
            0 - Not mandatory,
            1 - Mandatory.

        Example- SuppPcpMandatory: [0, 0, 0]
        """
        return self._supp_precip_mandatory

    @supp_precip_mandatory.setter
    def supp_precip_mandatory(self, value):
        """Set the list of flags for whether each supplemental precipitation input
        forcing specified by the user in the configuration file is mandatory or optional.
        This is used to control whether an error is raised if supplemental precipitation
        input forcing files are not found for each supplemental precipitation input
        forcing based on whether the user has specified each supplemental precipitation
        input forcing as mandatory or optional in the configuration file.
        """
        if self.number_supp_pcp > 0:
            self.check_number_of_inputs_supp_pcp(value, "SuppPcpMandatory")
            self.check_input_values_in_range(value, "SuppPcpMandatory", [0, 1])
            self._supp_precip_mandatory = value
        else:
            self._supp_precip_mandatory = None

    @property
    def regrid_opt_supp_pcp(self):
        """Specify regridding options for the supplemental precipitation products.
        Options available are:
            1 - ESMF Bilinear,
            2 - ESMF Nearest Neighbor,
            3 - ESMF Conservative Bilinear.

        Example- RegridOptSuppPcp: [1, 1, 1]
        """
        return self._regrid_opt_supp_pcp

    @regrid_opt_supp_pcp.setter
    def regrid_opt_supp_pcp(self, value):
        """Set the list of regridding options for supplemental precipitation input
        forcings specified by the user in the configuration file. This is used to
        control how supplemental precipitation input forcings are regridded based on
        the regridding option specified for each supplemental precipitation input
        forcing in the configuration file.
        """
        if self.number_supp_pcp > 0:
            self.check_number_of_inputs_supp_pcp(value, "RegridOptSuppPcp")
            self.check_input_values_in_range(value, "RegridOptSuppPcp", [1, 2, 3])
            self._regrid_opt_supp_pcp = value
        else:
            self._regrid_opt_supp_pcp = None

    @property
    def suppTemporalInterp(self):
        """Specify the time interpretation methods for the supplemental precipitation products.

        Example- SuppPcpTemporalInterpolation: [0, 0, 0]
        """
        return self._suppTemporalInterp

    @suppTemporalInterp.setter
    def suppTemporalInterp(self, value):
        """Set the list of flags for whether temporal interpolation of supplemental
        precipitation input forcings specified by the user in the configuration file is
        performed or not. This is used to control whether temporal interpolation of
        supplemental precipitation input forcings is performed based on whether the
        user has chosen to perform temporal interpolation for each supplemental
        precipitation input forcing in the configuration file.
        """
        if self.number_supp_pcp > 0:
            self.check_number_of_inputs_supp_pcp(value, "SuppPcpTemporalInterpolation")
            self.check_input_values_in_range(
                value, "SuppPcpTemporalInterpolation", [0, 1, 2]
            )
            self._suppTemporalInterp = value
        else:
            self._suppTemporalInterp = None

    @property
    def supp_pcp_max_hours(self):
        """Get the list of maximum forecast hours for supplemental precipitation input
        forcings specified by the user in the configuration file. This is used to
        control how supplemental precipitation input forcings are processed based on the
        maximum forecast hour specified for each supplemental precipitation input
        forcing in the configuration file.
        """
        return self._supp_pcp_max_hours

    @supp_pcp_max_hours.setter
    def supp_pcp_max_hours(self, value):
        """Set the list of maximum forecast hours for supplemental precipitation input
        forcings specified by the user in the configuration file. This is used to control
        how supplemental precipitation input forcings are processed based on the maximum
        forecast hour specified for each supplemental precipitation input forcing in the
        configuration file.
        """
        if self.number_supp_pcp > 0:
            if isinstance(value, list):
                self.check_number_of_inputs_supp_pcp(value, "SuppPcpMaxHours")
            elif isinstance(value, float) or isinstance(value, int):
                value = [value] * self.number_supp_pcp
            self._supp_pcp_max_hours = value
        else:
            self._supp_pcp_max_hours = None

    @property
    def supp_input_offsets(self):
        """In AnA runs, this value is the offset from the available forecast and 00z.
        For example, if forecast are available at 06z and 18z, set this value to 6.

        Example- SuppPcpInputOffsets = [0, 0, 0]
        """
        return self._supp_input_offsets

    @supp_input_offsets.setter
    def supp_input_offsets(self, value):
        """Set the list of time offsets to apply to supplemental precipitation input
        forcing files specified by the user in the configuration file. This is used to
        control how supplemental precipitation input forcing files are processed based
        on the time offset specified for each supplemental precipitation input forcing
        in the configuration file.
        """
        if self.number_supp_pcp > 0:
            self.check_number_of_inputs_supp_pcp(value, "SuppPcpInputOffsets")
            self.check_input_values_non_negative(value, "SuppPcpInputOffsets")
            self._supp_input_offsets = value
        else:
            self._supp_input_offsets = None

    @property
    def supp_precip_dirs(self):
        """Specify the correponding supplemental precipitation directories that will be searched for input files.

        Example- SuppPcpDirectories: ['./MRMS_CONUS_GAUGE', './MRMS_CONUS_MULTISENSOR', './MRMS_CLASSIFICATION']
        """
        return self._supp_precip_dirs

    @supp_precip_dirs.setter
    def supp_precip_dirs(self, value):
        """Set the list of pathways to the supplemental precipitation input forcing
        directories specified by the user in the configuration file. This is used to
        control where the program looks for supplemental precipitation input forcing
        files for each supplemental precipitation input forcing based on the directory
        specified for each supplemental precipitation input forcing in the configuration
        file.
        """
        if self.number_supp_pcp > 0:
            # Loop through and ensure all supp pcp directories exist. Also strip out any whitespace
            # or new line characters.
            for dirTmp in range(0, len(value)):
                value[dirTmp] = value[dirTmp].strip()
                self.try_make_dir(value[dirTmp], " supp pcp")

            # Special case for ExtAnA where we treat comma separated stage IV, MRMS data as one SuppPcp input
            # NOTE: the length check must happen after this join, since ExtAnA may supply 2 dirs for 1 SuppPcp product.
            if 11 in self.supp_precip_forcings or 12 in self.supp_precip_forcings:
                if len(self.supp_precip_forcings) != 1:
                    err_out_screen(
                        "CONUS or Alaska Stage IV/MRMS SuppPcp option is only supported as a standalone option"
                    )
                value = [",".join(value)]
            self.check_number_of_inputs_supp_pcp(value, "SuppPcpDirectories")
            self._supp_precip_dirs = value
        else:
            self._supp_precip_dirs = None

    @property
    def supp_precip_param_dir(self):
        """Specify an optional directory that contains supplemental precipitation
        parameter fields, I.E monthly RQI climatology.
        This is ONLY needed for the original NWM WRF-Hydro domain.
        Otherwise, just point it to a random directory and it will be ignored.

        Example- SuppPcpParamDir: ['./forcingParam/AnA','./forcingParam/AnA','./forcingParam/AnA']
        """
        return self._supp_precip_param_dir

    @supp_precip_param_dir.setter
    def supp_precip_param_dir(self, value):
        """Set the directory where downscaling parameters for supplemental precipitation
        input forcings are stored specified by the user in the configuration file.
        This is used to control where the program looks for downscaling parameter files
        for supplemental precipitation input forcings based on the directory specified
        for supplemental precipitation input forcings in the configuration file.
        """
        if self.number_supp_pcp > 0:
            self.try_make_dir(value, " SuppPcpParamDir")
            self._supp_precip_param_dir = value
        else:
            self._supp_precip_param_dir = None

    @property
    def cfsv2EnsMember(self):
        """Set the CFSv2 ensemble member to process specified by the user in the
        configuration file. This is used to control which CFSv2 ensemble member is
        processed for CFSv2 input forcings based on the ensemble member specified in the
        configuration file.
        """
        value = None
        if not self.precip_only_flag:
            # Read in Ensemble information
            # Read in CFS ensemble member information IF we have chosen CFSv2 as an input
            # forcing.
            for opt_tmp in self.input_forcings:
                if opt_tmp == 7:
                    value = self.extract_input_variable("cfsEnsNumber")
                    self.check_input_values_in_range(
                        [value], "cfsEnsNumber", [1, 2, 3, 4]
                    )
            return value

    @property
    def customFcstFreq(self):
        """Get the custom forecast frequency in minutes specified by the user in the
        configuration file. This is used to control how often forecasts are issued
        based on the custom forecast frequency specified in the configuration file.
        """
        return self._customFcstFreq

    @customFcstFreq.setter
    def customFcstFreq(self, value):
        """Options for specifying custom input NetCDF forcing files (in minutes). Choose
        the input frequency of files that are being processed. I.E., are the input files
        every 15 minutes, 60 minutes, 3-hours, etc. Please specify the length of custom
        input frequencies to match the number of custom NetCDF inputs selected above in
        the Logistics section.

        Example-  custom_input_fcst_freq: []
        """
        if not self.precip_only_flag:
            if len(value) != self.number_custom_inputs:
                err_out_screen(
                    f"Improper custom_input fcst_freq specified. This number ({len(value)}) must match the frequency of custom input forcings selected ({self.number_custom_inputs})."
                )
            self._customFcstFreq = value
        else:
            self._customFcstFreq = None

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
        """Determine if supplemental precipitation data can be used at the current output time.

        TODO: This property has no product context; it asserts that all supp pcp products share
        the same value for ``SuppPcpMaxHours`` in the config file . Needs revisiting if
        support is needed for differing values per-product.
        """
        if self.supp_pcp_max_hours:
            hrs_since_start = self.current_output_date - self.current_fcst_cycle
            if isinstance(self.supp_pcp_max_hours, list):
                if len(set(self.supp_pcp_max_hours)) != 1:
                    raise ValueError(
                        f"use_data_at_current_time does not support differing supp_pcp_max_hours per product: {self.supp_pcp_max_hours}"
                    )
                hours = self.supp_pcp_max_hours[0]
            elif isinstance(self.supp_pcp_max_hours, (int, float)):
                hours = self.supp_pcp_max_hours
            else:
                raise TypeError(
                    f"Unexpected type for supp_pcp_max_hours: {type(self.supp_pcp_max_hours)}"
                )
            return hrs_since_start <= timedelta(hours=hours)
        else:
            return True
