# Need these for BMI
# This is needed for get_var_bytes
import gc
import hashlib
import os

# time debugging
import time
from collections import defaultdict
from pathlib import Path

import ewts
import netCDF4 as nc

# import data_tools
# Basic utilities
import numpy as np
import pandas as pd

# Configuration file functionality
import yaml
from bmipy import Bmi

# Import MPI Python module
from mpi4py import MPI

from NextGen_Forcings_Engine_BMI import esmf_creation, forcing_extraction
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.bmi_grid import Grid, GridType
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.config import (
    ConfigOptions,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.consts import BMI_MODEL
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.geoMod import (
    GriddedGeoMeta,
    HydrofabricGeoMeta,
    UnstructuredGeoMeta,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.parallel import MpiConfig

from .core import (
    err_handler,
    forcingInputMod,
    ioMod,
    suppPrecipMod,
)
from .model import NWMv3ForcingEngineModel

# Import BMI grid functions to advertise grid features
# Here is the model we want to run

###### NWMv3.0 Forcings Engine modules ######
# For ESMF + shapely 2.x, shapely must be imported first, to avoid segfault "address not mapped to object" stemming from calls such as:
# /usr/local/esmf/lib/libO/Linux.gfortran.64.openmpi.default/libesmf_fullylinked.so(get_geom+0x36)
try:
    import esmpy as ESMF
except ImportError:
    import ESMF


from typing import Any

# Use the Error, Warning, and Trapping System Package for logging
import ewts
from numpy.typing import NDArray

LOG = ewts.get_logger(ewts.FORCING_ID)

# If less than 0, then ESMF.__version__ is greater than 8.7.0
if ESMF.version_compare("8.7.0", ESMF.__version__) < 0:
    manager = ESMF.api.esmpymanager.Manager(endFlag=ESMF.constants.EndAction.KEEP_MPI)


class UnknownBMIVariable(RuntimeError):
    """Custom exception raised when an unknown BMI variable is encountered."""

    pass


class NWMv3_Forcing_Engine_BMI_model_Base(Bmi):
    """Defines the BMI (Basic Model Interface) for the NWMv3.0 Forcings Engine model.

    It includes methods for initializing the model, updating it, accessing model variables,
    and managing model configuration. This class is responsible for interacting with
    geospatial data and forcing inputs for the model simulation.

    Attributes
    ----------
    _values : dict
        Dictionary storing model values.
    _start_time : float
        The start time for the simulation.
    _end_time : float
        The end time for the simulation.
    _model : object
        The model object.
    _comm : object
        The MPI communicator.
    var_array_lengths : int
        Length of the variable arrays.

    """

    def __init__(self):
        """Create a model that is ready for initialization.

        Initializes the model with default values for time, variables, and grid types.
        """
        super(NWMv3_Forcing_Engine_BMI_model_Base, self).__init__()
        self._values = {}
        self._start_time = 0.0
        self._end_time = np.finfo(float).max
        self._model = None
        self._comm = None
        self.var_array_lengths = 1

        # Track output configuration status
        self._output_configured = False

        # Initialize attributes in __init__ to avoid PyCharm errors
        self.cfg_bmi = None
        self._job_meta = None
        self._mpi_meta = None
        self.geo_meta = None
        self._grid_type = None
        self._grids = None
        self._grid_map = None
        self._output_var_names = None
        self._var_name_units_map = None
        self._var_name_map_long_first = None
        self._var_name_map_short_first = None
        self._var_units_map = None
        self._input_forcing_mod = None
        self._supp_pcp_mod = None
        self._model_parameters_list = []

        # Diagnostic timing setup

        self._call_counts = defaultdict(int)
        self._call_times = defaultdict(float)
        self._total_start = None

    # ----------------------------------------------
    # Required, static attributes of the model
    # ----------------------------------------------
    _att_map = BMI_MODEL["att_map"]

    # ---------------------------------------------
    # Input variable names (CSDMS standard names)
    # ---------------------------------------------
    # Forcings engine requires no inputs currently
    # and only provides model output
    _input_var_names = []

    _input_var_types = {}

    # ------------------------------------------------------
    # A list of static attributes/parameters.
    # ------------------------------------------------------
    _model_parameters_list = []

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # BMI: Model Control Functions
    # ------------------------------------------------------------
    # ------------------------------------------------------------

    # -------------------------------------------------------------------
    def initialize(self, config_file: str, output_path: str | None = None) -> None:
        """Initialize the model using a configuration file.

        This function is part of the BMI (Basic Model Interface) specification and is automatically
        invoked by the BMI system. When running standalone, call `initialize_with_params()` instead,
        which sets additional parameters such as `b_date`, `geogrid`, and `output_path`.

        This function is responsible for:
        - Setting up core model attributes, grids, and MPI communication.
        - Reading the BMI configuration file and initializing basic model components.

        :param config_file: The path to the configuration file for model initialization.
        :raises RuntimeError: If the configuration file is invalid or missing.
        """
        # This is required prior to the first log message.
        LOG.bind()

        LOG.info("---------------------------")
        LOG.info(f"BMI Forcing Engine initialized with {config_file}")

        # -------------- Read in the BMI configuration -------------------------#
        if not isinstance(config_file, str) or len(config_file) == 0:
            LOG.critical("No BMI initialize configuration provided, nothing to do...")
            raise RuntimeError(
                "No BMI initialize configuration provided, nothing to do..."
            )

        bmi_cfg_file = Path(config_file).resolve()
        if not bmi_cfg_file.is_file():
            LOG.critical(f"Config file {bmi_cfg_file} not found, nothing to do...")
            raise RuntimeError(
                f"Config file {bmi_cfg_file} not found, nothing to do..."
            )

        LOG.info(f"Reading config file: {bmi_cfg_file}")
        with bmi_cfg_file.open("r") as fp:
            cfg = yaml.safe_load(fp)

        self.cfg_bmi = parse_config(cfg)

        # If _job_meta was not set by initialize_with_params(), create a default one
        if self._job_meta is None:
            self._job_meta = ConfigOptions(self.cfg_bmi)

        # Parse the configuration options
        try:
            self._job_meta.validate_config(self.cfg_bmi)
        except KeyboardInterrupt as e:
            err_handler.err_out_screen("User keyboard interrupt", e)
        except ImportError as e:
            err_handler.err_out_screen("Missing Python packages", e)
        except InterruptedError as e:
            err_handler.err_out_screen("External kill signal detected", e)
        except Exception as e:
            err_handler.err_out_screen("Unhandled exception", e)

        # Set NWM version and config, if provided in the config
        if self.cfg_bmi.get("NWM_VERSION") is not None:
            self._job_meta.nwmVersion = self.cfg_bmi["NWM_VERSION"]

        # Place NWM configuration (if provided by the user). This will be placed into the final
        # output files as a global attribute.
        if self.cfg_bmi.get("NWM_CONFIG") is not None:
            self._job_meta.nwmConfig = self.cfg_bmi["NWM_CONFIG"]

        # Initialize MPI communication
        self._mpi_meta = MpiConfig(self._job_meta)

        self.geo_meta = HydrofabricGeoMeta(self._job_meta, self._mpi_meta)

        try:
            comm = MPI.Comm.f2py(self._comm) if self._comm is not None else None
            self._mpi_meta.initialize_comm(comm=comm)
        except Exception as e:
            err_handler.err_out_screen(self._job_meta.errMsg, e)

        ### Reassign the scratch dir to a new child dir of the current scratch dir,
        ### applying uniqueness to the final path. This must be called by all ranks, once.
        self._job_meta.uniquefy_scratch_dir_as_child(self._mpi_meta.uid64)

        # LOG.debug(f"self._job_meta type: {type(self._job_meta)}")
        # Call ESMF mesh creation process
        if self._mpi_meta.rank == 0:
            esmf_creation.create_mesh(self._job_meta)
        self._mpi_meta.comm.Barrier()

        # Call forcing_extraction process
        if self._job_meta.nwmConfig not in ["AORC", "NWM"]:
            if self._mpi_meta.rank == 0:
                err_handler.log_msg(
                    self._job_meta,
                    self._mpi_meta,
                    False,
                    "About to fetch raw forcing data",
                )
                forcing_extraction.retrieve_forcing(self._job_meta)
                err_handler.log_msg(
                    self._job_meta,
                    self._mpi_meta,
                    False,
                    "Finished fetching raw forcing data",
                )
        self._mpi_meta.comm.Barrier()

        # Assign grid type to BMI class for grid information
        self._grid_type = self._job_meta.grid_type.lower()
        self.set_var_names()

        # ----- Create some lookup tabels from the long variable names --------#
        self._var_name_map_long_first = {
            long_name: self._var_name_units_map[long_name][0]
            for long_name in self._var_name_units_map.keys()
        }
        self._var_name_map_short_first = {
            self._var_name_units_map[long_name][0]: long_name
            for long_name in self._var_name_units_map.keys()
        }
        self._var_units_map = {
            long_name: self._var_name_units_map[long_name][1]
            for long_name in self._var_name_units_map.keys()
        }

        # Check to make sure we have enough dimensionality to run regridding. We assume that hydrofabric discretizations are large
        # enough that 1x1 (single catchment) will provide enough points. For gridded and unstructured domains, we need to make sure
        # that the local grid size for each processor is at least 2x2 to run the regridding process.
        # forcing_input dimensionality is checked in regrid.py.

        dimensionality = 1 if self._grid_type == "hydrofabric" else 2

        if (
            self.geo_meta.nx_local < dimensionality
            or self.geo_meta.ny_local < dimensionality
        ):
            self._job_meta.errMsg = (
                f"You have specified too many cores for your WRF-Hydro grid. "
                f"Local grid Must have x/y dimension size of {dimensionality}."
            )
            err_handler.err_out_screen_para(self._job_meta.errMsg, self._mpi_meta)
        err_handler.check_program_status(self._job_meta, self._mpi_meta)

        # Initialize our output object, which includes local slabs from the output grid.
        try:
            self._output_obj = ioMod.OutputObj(self._job_meta, self.geo_meta)
        except Exception as e:
            err_handler.err_out_screen_para(self._job_meta, self._mpi_meta)
        err_handler.check_program_status(self._job_meta, self._mpi_meta)

        # Next, initialize our input forcing classes. These objects will contain
        # information about our source products (I.E. data type, grid sizes, etc).
        # Information will be mapped via the options specified by the user.
        # In addition, input ESMF grid objects will be created to hold data for
        # downscaling and regridding purposes.
        try:
            self._input_forcing_mod = forcingInputMod.init_dict(
                self._job_meta, self.geo_meta, self._mpi_meta
            )
        except Exception as e:
            err_handler.err_out_screen_para(self._job_meta, self._mpi_meta)
        err_handler.check_program_status(self._job_meta, self._mpi_meta)

        # If we have specified supplemental precipitation products, initialize
        # the supp class.
        if self._job_meta.number_supp_pcp > 0:
            self._supp_pcp_mod = suppPrecipMod.initDict(self._job_meta, self.geo_meta)
        else:
            self._supp_pcp_mod = None
        err_handler.check_program_status(self._job_meta, self._mpi_meta)

        # ------------- Initialize the parameters, inputs and outputs ----------#
        for parm in self._model_parameters_list:
            self._values[self._var_name_map_short_first[parm]] = self.cfg_bmi[parm]

        self.get_size_of_arrays()

        # for model_input in self.get_input_var_names():
        #    self._values[model_input] = np.zeros(self._varsize, dtype=float)

        # Set initial time and step
        self._values["current_model_time"] = self.cfg_bmi["initial_time"]
        self._values["time_step_size"] = self.cfg_bmi["time_step_seconds"]

        # Initialize the Forcings Engine model
        self._model = NWMv3ForcingEngineModel()

        # Set catchment ids if using hydrofabric
        if self._grid_type == "hydrofabric":
            self._values["CAT-ID"] = self.geo_meta.element_ids_global

        self._configure_output_path(output_path)

    def initialize_with_params(
        self,
        config_file: str,
        b_date: str = None,
        geogrid: str = None,
        output_path: str = None,
    ) -> None:
        """Initialize the NWMv3 Forcings Engine model with additional job metadata parameters.

        This function **must be called by the user** to fully initialize the NWMv3 Forcings Engine model,
        including both core model setup and additional job metadata configuration (such as b_date, geogrid, and output path).

        It performs the following:
        - Sets up job metadata (b_date, geogrid) by calling `config_options`.
        - Calls the `initialize()` function to handle core model setup (reading the config file,
          initializing basic model attributes like MPI, grids, etc.).
        - Handles additional configuration options, such as determining the output path
          for model results.

        **DO NOT call `initialize()` directly**. Always use this function, which ensures proper
        initialization of all necessary parameters and job metadata.

        :param config_file: The configuration file path for the model initialization.
        :param b_date: The start date for the simulation. Typically the forecast cycle start time.
        :param geogrid: The path to the geospatial grid data, such as a geospatial file for the grid.
        :param output_path: The output path for model results. If omitted, a default path will be generated.
        :raises ValueError: If an invalid grid type is specified, an exception is raised.
        """
        # Set the job metadata parameters (b_date, geogrid) using config_options
        self._job_meta = ConfigOptions(self.cfg_bmi, b_date=b_date, geogrid_arg=geogrid)

        # Now that _job_meta is set, call initialize() to set up the core model
        self.initialize(config_file, output_path=output_path)

    def _configure_output_path(self, output_path: str | None = None) -> None:
        """Set the output path and initializes the output NetCDF file if forcing output is enabled.

        This is safe to call once after model initialization.

        :param output_path: Optional override path.
        """
        gpkg_key = self._job_meta.geopackage
        time_key = str(time.time()).replace(".", "")
        gpkg_hash = hashlib.md5(gpkg_key.encode()).hexdigest()[:8]
        time_hash = hashlib.md5(time_key.encode()).hexdigest()[:8]

        if self._output_configured or self._output_obj is None:
            return  # Already configured or no output object to configure

        if self._job_meta.forcing_output == 1:
            ext = BMI_MODEL["extension_map"].get(self._job_meta.grid_type)

            if ext is None:
                raise ValueError(f"Invalid grid_type: {self._job_meta.grid_type}")

            if output_path:
                self._output_obj.outPath = output_path
            else:
                filename = (
                    f"NextGen_Forcings_Engine_{ext}_{gpkg_hash}_{time_hash}_output_"
                    + pd.Timestamp(self._job_meta.b_date_proc).strftime("%Y%m%d%H%M")
                    + ".nc"
                )
                self._output_obj.outPath = os.path.join(
                    self._job_meta.scratch_dir, filename
                )

            self._output_obj.init_forcing_file(
                self._job_meta, self.geo_meta, self._mpi_meta
            )
            self._output_configured = True

    # ------------------------------------------------------------
    def update(self):
        """Update the model by advancing one time step.

        This method increments the current model time by the time step size
        and then updates the model state by calling the `update_until` method
        with the new time.

        :return: None
        """
        # Run the model to the next timestep
        self.update_until(
            self._values["current_model_time"] + self._values["time_step_size"]
        )

    # ------------------------------------------------------------
    def update_until(self, future_time: float):
        """Update the model to a specified future time.

        This method updates the model by running time steps until the
        `future_time` is reached. If the `future_time` is different from the
        current model time, the model is updated iteratively. If the `future_time`
        matches the current time, a single step is performed.

        :param future_time: The target time to update the model to.
        :return: None

        """
        # Method for running the model on the initial time if the model has not been run,
        # and the future time is the same as the initial time.

        if (
            self._values["current_model_time"]
            == future_time
            == self.cfg_bmi["initial_time"]
        ):
            self._model.run(
                self._values,
                future_time,
                self._job_meta,
                self.geo_meta,
                self._input_forcing_mod,
                self._supp_pcp_mod,
                self._mpi_meta,
                self._output_obj,
            )
        else:
            # Start a while loop to iterate the model time step by step until the
            # current model time reaches or exceeds the future_time.
            while self._values["current_model_time"] < future_time:
                # Advance the model time by the defined time step size.
                self._values["current_model_time"] += self._values["time_step_size"]
                # Run the model for the new current time and update the state.
                self._model.run(
                    self._values,
                    self._values["current_model_time"],
                    self._job_meta,
                    self.geo_meta,
                    self._input_forcing_mod,
                    self._supp_pcp_mod,
                    self._mpi_meta,
                    self._output_obj,
                )

    # ------------------------------------------------------------
    def finalize(self):
        """Finalize the model, performing necessary cleanup tasks.

        This method cleans up any temporary files created during the model run,
        including files in the scratch directory. It also forces the destruction
        of certain objects related to the model. This method is typically called
        after the model has finished running.

        :return: None

        """
        err_handler.log_msg(
            self._job_meta, self._mpi_meta, True, "Starting BMI finalize()"
        )

        # Force destruction of ESMF objects
        self.geo_meta = None
        self._input_forcing_mod = None
        self._supp_pcp_mod = None
        self._model = None

        # Try moving this after all of the ESMF and model bits have
        # been disposed of - maybe they were keeping something open.
        #
        # Potential workaround if that's not enough: uncomment the
        # return before the file cleanup block, leak the files during
        # the job, and let the workflow clean them up after the
        # process exits
        gc.collect()  # make sure objects are deleted from memory

    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # BMI: Model Information Functions
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------

    def get_attribute(self, att_name):
        """Retrieve an attribute from the model's attribute map.

        This method searches the `_att_map` dictionary for the specified attribute name
        and returns its value. If the attribute is not found, an error message is printed.

        :param att_name: The name of the attribute to retrieve.
        :return: The value of the attribute if found.
        """
        try:
            return self._att_map[att_name.lower()]
        except Exception as e:
            LOG.error(f"Could not find attribute: {att_name} - {e}")

    # --------------------------------------------------------
    # Note: These are currently variables needed from other
    #       components vs. those read from files or GUI.
    # --------------------------------------------------------
    def get_input_var_names(self):
        """Get the list of input variable names.

        This method returns the list of input variable names defined in the model.

        :return: List of input variable names.
        """
        return self._input_var_names

    def get_output_var_names(self):
        """Get the list of output variable names.

        This method returns the list of output variable names defined in the model.

        :return: List of output variable names.
        """
        return self._output_var_names

    # ------------------------------------------------------------
    def get_component_name(self):
        """Get the name of the component.

        This method retrieves the model name using the `get_attribute` method.

        :return: The name of the model component.
        """
        return self.get_attribute("model_name")

    # ------------------------------------------------------------
    def get_input_item_count(self):
        """Get the count of input variables.

        This method returns the total number of input variables defined in the model.

        :return: The number of input variables.
        """
        return len(self._input_var_names)

    # ------------------------------------------------------------
    def get_output_item_count(self):
        """Get the count of output variables.

        This method returns the total number of output variables defined in the model.

        :return: The number of output variables.
        """
        return len(self._output_var_names)

    # ------------------------------------------------------------
    def get_value(self, var_name: str, dest: NDArray[Any]) -> NDArray[Any]:
        """Copy the values of a variable into the provided destination array.

        This method copies the values of a specified variable (by its CSDMS Standard Name)
        into the provided destination array (`dest`).

        :param var_name: The name of the variable whose values are to be retrieved.
        :param dest: The numpy array to store the values of the variable.
        :return: The destination array containing the variable values.

        """
        # LOG.debug(f"[BMI get_value] Called with var_name: '{var_name}'")
        # LOG.debug(f"[BMI get_value] Destination array shape: {dest.shape}, dtype: {dest.dtype}")

        if var_name == "grid:count":
            LOG.debug(
                f"[BMI get_value] Special case: 'grid:count', grid_type: {self._job_meta.grid_type}"
            )
            if self._job_meta.grid_type != "unstructured":
                dest[...] = 1
            else:
                dest[...] = 2
        elif var_name == "grid:ids":
            LOG.debug(
                f"[BMI get_value] Special case: 'grid:ids', grid_type: {self._job_meta.grid_type}"
            )
            dest[:] = self.grid_ids(self)

        elif var_name == "grid:ranks":
            LOG.debug(
                f"[BMI get_value] Special case: 'grid:ranks', grid_type: {self._job_meta.grid_type}"
            )
            dest[:] = self.grid_ranks(self)
        else:
            src = self.get_value_ptr(var_name)
            LOG.debug(
                f"[BMI get_value] Source array shape: {src.shape}, dtype: {src.dtype}"
            )
            if dest.shape != src.shape:
                LOG.warning(
                    f"BMI Shape mismatch! dest.shape = {dest.shape}, src.shape = {src.shape}"
                )
            if dest.dtype != src.dtype:
                LOG.warning(
                    f"BMI Dtype mismatch! dest.dtype = {dest.dtype}, src.dtype = {src.dtype}"
                )
            dest[:] = src

        LOG.debug(f"[BMI get_value] Completed assignment for var_name: '{var_name}'")

        return dest

    # -------------------------------------------------------------------
    def get_value_ptr(self, var_name: str) -> NDArray[Any]:
        """Get a reference to the values of a variable.

        This method returns a reference to the values of the specified variable,
        allowing direct access to the array without copying.

        :param var_name: The name of the variable whose values are to be retrieved.
        :return: A flattened array containing the values of the variable.

        """
        # Make sure to return a flattened array
        if (
            var_name == "grid_1_shape"
        ):  # FIXME cannot expose shape as ptr, because it has to side affect variable construction...
            return self.grid_1.shape
        if var_name == "grid_1_spacing":
            return self.grid_1.spacing
        if var_name == "grid_1_origin":
            return self.grid_1.origin
        if var_name == "grid_1_units":
            return self.grid_1.units
        if var_name == "grid_2_shape":
            return self.grid_2.shape
        if var_name == "grid_2_spacing":
            return self.grid_2.spacing
        if var_name == "grid_2_origin":
            return self.grid_2.origin
        if var_name == "grid_2_units":
            return self.grid_2.units
        if var_name == "grid_3_shape":
            return self.grid_3.shape
        if var_name == "grid_3_spacing":
            return self.grid_3.spacing
        if var_name == "grid_3_origin":
            return self.grid_3.origin
        if var_name == "grid_3_units":
            return self.grid_3.units
        if var_name == "grid_4_shape":
            return self.grid_4.shape
        if var_name == "grid_4_spacing":
            return self.grid_4.spacing
        if var_name == "grid_4_origin":
            return self.grid_4.origin
        if var_name == "grid_4_units":
            return self.grid_4.units

        # if var_name not in self._values.keys():
        #     raise (UnknownBMIVariable(f"No known variable in BMI model: {var_name}"))
        if var_name not in self._values:
            LOG.error(f"No known variable in BMI model: '{var_name}'")
            LOG.error("Available variables:")
            for key in self._values:
                LOG.error(f" - {key}")
            LOG.error("Output variable names:")
            for var in self._output_var_names:
                LOG.error(f" - {var}")
            LOG.error("Grid type: {self._grid_type}")
            raise UnknownBMIVariable(f"No known variable in BMI model: '{var_name}'")

        arr = self._values[var_name]
        # LOG.debug(f"[BMI get_value_ptr] Found variable '{var_name}' with shape {arr.shape} and dtype {arr.dtype}")

        # Ensure array is C-contiguous
        if not arr.flags["C_CONTIGUOUS"]:
            LOG.warning(
                f"[BMI] Array for '{var_name}' is not C-contiguous; making a copy."
            )
            arr = np.ascontiguousarray(arr)

        # Ensure dtype is float64 (C double), except for CAT-ID
        if var_name == "CAT-ID":
            if arr.dtype != np.int32:
                msg = f"[BMI] Array for '{var_name}' has dtype {arr.dtype}, expected int32"
                LOG.critical(msg)
                raise RuntimeError(msg)
        elif arr.dtype != np.float64:
            LOG.warning(
                f"[BMI] Array for '{var_name}' has dtype {arr.dtype}, expected float64; converting."
            )
            arr = arr.astype(np.float64)

        # Confirm raveling is safe
        shape = arr.shape
        try:
            # See if raveling is possible without a copy
            arr.shape = (-1,)
            # reset original shape
            arr.shape = shape
        except ValueError as e:
            LOG.critical(
                "Cannot flatten array without copying -- " + str(e).split(": ")[-1]
            )
            raise RuntimeError(
                "Cannot flatten array without copying -- " + str(e).split(": ")[-1]
            )

        # LOG.debug(f"[BMI get_value_ptr] Returning ravelled array for variable '{var_name}'")
        return arr.ravel()

    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # BMI: Variable Information Functions
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    def get_var_name(self, long_var_name):
        """Get the short name of the variable corresponding to the long variable name.

        :param long_var_name: The long variable name as defined in the model.
        :return: The corresponding short name of the variable.
        """
        return self._var_name_map_long_first[long_var_name]

    # -------------------------------------------------------------------
    def get_var_units(self, long_var_name):
        """Get the units of the variable corresponding to the long variable name.

        :param long_var_name: The long variable name as defined in the model.
        :return: The units of the variable.
        """
        return self._var_units_map[long_var_name]

    # -------------------------------------------------------------------
    def get_var_type(self, var_name: str) -> str:
        """Get the data type of a variable.

        :param var_name: The name of the variable as defined in the model.

        :return: The data type of the variable.
        """
        return str(self.get_value_ptr(var_name).dtype)

    # ------------------------------------------------------------
    def get_var_grid(self, name):
        """Get the grid associated with a variable.

        :param name: The name of the variable.

        :return: The grid ID associated with the variable.
        """
        # all vars have grid 0 but check if its in names list first
        if name in self._output_var_names:
            if "ELEMENT" in name and self._job_meta.grid_type == "gridded":
                return 1
            elif "ELEMENT" in name and self._job_meta.grid_type == "unstructured":
                return 2
            elif "NODE" in name and self._job_meta.grid_type == "unstructured":
                return 3
            elif "ELEMENT" in name and self._job_meta.grid_type == "hydrofabric":
                return 4
            else:
                return self._var_grid_id
        raise (UnknownBMIVariable(f"No known variable in BMI model: {name}"))

    # ------------------------------------------------------------
    def get_var_itemsize(self, name):
        """Get the item size (in bytes) of a variable.

        This function retrieves the memory size (in bytes) for each element of the variable
        specified by the `name` parameter.

        :param name: The name of the variable.
        :return: The item size of the variable in bytes.
        """
        return self.get_value_ptr(name).itemsize

    # ------------------------------------------------------------
    def get_var_location(self, name):
        """Get the location of a variable in the grid.

        This function determines the location of a variable (whether it's at a "face"
        or "node" in the grid) based on its name. It assumes that variables with
        "ELEMENT" in the name are at the "face" location, and variables with
        "NODE" are at the "node" location.

        :param name: The name of the variable.
        :return: The location of the variable ("face" or "node").
        :raises ValueError: If the location of the variable cannot be determined.
        """
        if "ELEMENT" in name:
            return "face"
        elif "NODE" in name:
            return "node"
        else:
            raise ValueError(f"get_var_location: grid_id {self._var_grid_id} unknown")

    # -------------------------------------------------------------------
    def get_var_rank(self, long_var_name):
        """Get the rank of a variable.

        This function retrieves the rank (number of dimensions) of a variable
        specified by its long name. Currently, it returns a constant value of
        0 for all variables.

        :param long_var_name: The long name of the variable.
        :return: The rank (number of dimensions) of the variable.
        """
        return np.int16(0)

    # -------------------------------------------------------------------
    def get_start_time(self) -> float:
        """Get the model's start time.

        This function returns the start time of the model, which is used for
        time stepping in the simulation.

        :return: The start time of the model.
        """
        return self._start_time

        # -------------------------------------------------------------------

    def get_end_time(self) -> float:
        """Get the model's end time.

        This function returns the end time of the model, which is used to
        determine the duration of the simulation.

        :return: The end time of the model.

        """
        return self._end_time

        # -------------------------------------------------------------------

    def get_current_time(self) -> float:
        """Get the current time of the model.

        This function returns the current model time, which is updated after
        each time step in the simulation.

        :return: The current time of the model.
        """
        return self._values["current_model_time"]

    # -------------------------------------------------------------------
    def get_time_step(self) -> float:
        """Get the model's time step size.

        This function returns the time step size used for advancing the model
        from one time to the next.

        :return: The time step size of the model.
        """
        return self._values["time_step_size"]

    # -------------------------------------------------------------------
    def get_time_units(self) -> str:
        """Get the units of time for the model.

        This function retrieves the units of time used in the model, typically
        provided during model initialization.

        :return: The units of time for the model (e.g., "seconds").
        """
        return self.get_attribute("time_units")

        # -------------------------------------------------------------------

    def set_value(self, var_name: str, values: NDArray[Any]):
        """Set model values for the provided BMI variable.

        This function sets the values for a model variable. If the variable
        is special (e.g., 'bmi_mpi_comm'), it handles those cases specifically.
        Otherwise, it assigns the values to the given variable.

        :param var_name: Name of the variable for which to set values.
        :param values: The new values to assign to the variable.
        """
        if var_name == "bmi_mpi_comm":
            self._comm = values[0]
        else:
            self._values[var_name][:] = values

    # ------------------------------------------------------------
    def set_value_at_indices(
        self, var_name: str, indices: NDArray[np.int_], src: NDArray[Any]
    ):
        """Set model values for the provided BMI variable at particular indices.

        This function allows setting the values of a variable at specific indices
        rather than for the entire variable.

        :param var_name: The name of the variable for which to set values.
        :param indices: The indices at which the values should be set.
        :param src: The array of new values to set at the specified indices.
        """
        # This is not particularly efficient, but it is functionally correct.
        for i in range(indices.shape[0]):
            bmi_var_value_index = indices[i]
            self.get_value_ptr(var_name)[bmi_var_value_index] = src[i]

    # ------------------------------------------------------------
    def get_var_nbytes(self, var_name) -> int:
        """Get the number of bytes required for a variable.

        This function retrieves the number of bytes used by a variable in memory.
        It is useful for understanding the memory requirements of the model.

        :param var_name: Name of the variable.
        :return: The size of the variable's data array in bytes.
        """
        return self.get_value_ptr(var_name).nbytes

    # ------------------------------------------------------------
    def get_value_at_indices(
        self, var_name: str, dest: NDArray[Any], indices: NDArray[np.int_]
    ) -> NDArray[Any]:
        """Get values at particular indices.

        This function retrieves the values of a variable at specific indices and
        stores them in the provided destination array.

        :param var_name: The name of the variable as a CSDMS Standard Name.
        :param dest: A numpy array into which to place the values.
        :param indices: An array of indices specifying the locations to retrieve the values from.
        :return: The destination array containing the values at the specified indices.
        """
        original: NDArray[Any] = self.get_value_ptr(var_name)
        for i in range(indices.shape[0]):
            value_index = indices[i]
            dest[i] = original[value_index]
        return dest

    # JG Note: remaining grid funcs do not apply for type 'scalar'
    #   Yet all functions in the BMI must be implemented
    #   See https://bmi.readthedocs.io/en/latest/bmi.best_practices.html
    # ------------------------------------------------------------
    def get_grid_edge_count(self, grid_id: int) -> int:
        """Retrieve the number of edges for the specified grid.

        This function accesses the grid and counts the number of unique edges
        based on the element connection data. It is not implemented for grid_id = 1.

        :param grid_id: The ID of the grid to retrieve edge count for.
        :return: The number of edges in the grid.
        :raises NotImplementedError: If grid_id is 1, the function raises an error.
        :raises ValueError: If an unexpected error occurs.

        """
        for _ in self._grids:
            if grid_id != 1:
                mesh = nc.Dataset(self._job_meta.geogrid)
                elem_conn = mesh.variables[self._job_meta.elemconn_var][
                    :
                ]  # Element connectivity
                numelem_conn = mesh.variables[self._job_meta.numelemconn_var][
                    :
                ]  # Number of element connections
                mesh.close()

                mesh_edge_first_node = []
                mesh_edge_second_node = []

                # Loop through the elements to collect edge nodes
                for i in range(elem_conn.shape[0]):
                    loop = 0
                    while loop + 1 < numelem_conn[i]:
                        mesh_edge_first_node.append(elem_conn[i, loop])
                        mesh_edge_second_node.append(elem_conn[i, loop + 1])
                        loop += 1
                        if loop + 1 == numelem_conn[i]:
                            mesh_edge_first_node.append(
                                elem_conn[i, numelem_conn[i] - 1]
                            )
                            mesh_edge_second_node.append(elem_conn[i, 0])

                # Create edge node pairs
                edge_nodes = np.empty((len(mesh_edge_first_node), 2), dtype=int)
                edge_nodes[:, 0] = mesh_edge_first_node
                edge_nodes[:, 1] = mesh_edge_second_node

                edge_nodes = list(edge_nodes)
                seen = set()
                count = 1  # Initialize count to 1 to count unique edges

                # Count unique edges
                for item in edge_nodes:
                    t = tuple(item)
                    if t not in seen:
                        seen.add(t)
                        count += 1

                edge_count = count
                return edge_count
            else:
                # Raise NotImplementedError if grid_id is 1.
                raise NotImplementedError(
                    "get_grid_edge_count is not implemented for grid_id 1"
                )

        # If no valid grid is found, raise an exception or handle accordingly.
        raise ValueError("No valid grid found to calculate edge count.")

    # ------------------------------------------------------------
    def get_grid_edge_nodes(
        self, grid_id: int, edge_nodes: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Retrieve the edge nodes for the specified grid.

        This function retrieves the edge nodes for a grid by accessing the grid's
        element connectivity data and deduplicating the edges. It returns the edge
        nodes in the provided array.

        :param grid_id: The ID of the grid to retrieve edge nodes for.
        :param edge_nodes: A numpy array where the edge nodes will be stored.
        :return: The edge nodes of the specified grid.
        :raises NotImplementedError: If grid_id is 1, the function raises an error.
        :raises Exception: If an unexpected error occurs in retrieving the edge nodes.
        """
        for _ in self._grids:
            if grid_id != 1:
                mesh = nc.Dataset(self._job_meta.geogrid)
                elem_conn = mesh.variables[self._job_meta.elemconn_var][:]
                numelem_conn = mesh.variables[self._job_meta.numelemconn_var][:]
                mesh.close()

                mesh_edge_first_node = []
                mesh_edge_second_node = []
                for i in range(elem_conn.shape[0]):
                    loop = 0
                    while loop + 1 < numelem_conn[i]:
                        mesh_edge_first_node.append(elem_conn[i, loop])
                        mesh_edge_second_node.append(elem_conn[i, loop + 1])
                        loop += 1
                        if loop + 1 == numelem_conn[i]:
                            mesh_edge_first_node.append(
                                elem_conn[i, numelem_conn[i] - 1]
                            )
                            mesh_edge_second_node.append(elem_conn[i, 0])

                # Create a 2D numpy array for edge nodes with shape (N, 2)
                edge_nodes_ = np.empty((len(mesh_edge_first_node), 2), dtype=int)
                edge_nodes_[:, 0] = mesh_edge_first_node
                edge_nodes_[:, 1] = mesh_edge_second_node

                # Deduplicate edge nodes
                edge_nodes_ = list(edge_nodes_)
                seen = set()
                node_list = []
                for item in edge_nodes_:
                    t = tuple(item)
                    if t not in seen:
                        node_list.append(t)
                        seen.add(t)
                    else:
                        edge_data = list(seen)
                        node_list.append(edge_data[edge_data.index(t)])
                edge_nodes[:] = np.array(node_list).flatten()
                return edge_nodes
            else:
                # Raise NotImplementedError if grid_id is 1.
                raise NotImplementedError(
                    "get_grid_edge_nodes is not implemented for grid_id 1"
                )

        raise Exception("Unexpected error in retrieving edge nodes")

    # ------------------------------------------------------------
    def get_grid_face_count(self, grid_id: int) -> int:
        """Retrieve the number of faces for the specified grid.

        This function accesses the grid and counts the number of faces based on the face coordinates
        in the grid's data. The grid must not be of type 1, as this function is not implemented for that case.

        :param grid_id: The ID of the grid to retrieve face count for.
        :return: The number of faces in the grid.
        :raises NotImplementedError: If grid_id is 1, the function raises an error as it is not implemented for that case.
        :raises ValueError: If grid ID is not found in `_grids`, or if an unexpected error occurs.
        """
        for _ in self._grids:
            if grid_id != 1:
                mesh = nc.Dataset(self._job_meta.geogrid)
                face_count = len(mesh.variables[self._job_meta.elemcoords_var][:][:, 0])
                mesh.close()
                return face_count
            else:
                # Raise NotImplementedError if grid_id is 1.
                raise NotImplementedError(
                    "get_grid_face_count is not implemented for grid_id 1"
                )

        # If the loop doesn't return, raise an exception indicating grid ID not found.
        raise ValueError("Grid ID not found in _grids.")

    # ------------------------------------------------------------
    def get_grid_face_edges(
        self, grid_id: int, face_edges: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Retrieve the face edges for a specific grid, given its ID.

        This function checks the grid type and retrieves the face edges by reading the
        connectivity data from the geogrid. If grid_id is 1, this function raises a
        NotImplementedError.

        :param grid_id: The ID of the grid for which face edges are to be retrieved.
        :param grid_id: The ID of the grid for which face edges are to be retrieved.
        :param face_edges: A pre-allocated numpy array where the face edges will be stored.
        :return: The updated `face_edges` array with the retrieved face edge values.
        :raises NotImplementedError: If grid_id is 1, as this functionality is not implemented for that case.
        :raises Exception: If an unexpected error occurs during the retrieval of face edges.
        """
        for _ in self._grids:
            if grid_id != 1:
                mesh = nc.Dataset(self._job_meta.geogrid)
                elem_conn = mesh.variables[self._job_meta.elemconn_var][:]
                numelem_conn = mesh.variables[self._job_meta.numelemconn_var][:]
                mesh.close()

                mesh_edge_first_node = []
                mesh_edge_second_node = []
                for i in range(elem_conn.shape[0]):
                    loop = 0
                    while loop + 1 < numelem_conn[i]:
                        mesh_edge_first_node.append(elem_conn[i, loop])
                        mesh_edge_second_node.append(elem_conn[i, loop + 1])
                        loop += 1
                        if loop + 1 == numelem_conn[i]:
                            mesh_edge_first_node.append(
                                elem_conn[i, numelem_conn[i] - 1]
                            )
                            mesh_edge_second_node.append(elem_conn[i, 0])

                edge_nodes = np.empty((len(mesh_edge_first_node), 2), dtype=int)
                edge_nodes[:, 0] = mesh_edge_first_node
                edge_nodes[:, 1] = mesh_edge_second_node
                edge_nodes = list(edge_nodes)
                seen = set()
                edge_list = []
                count = 1
                for item in edge_nodes:
                    t = tuple(item)
                    if t not in seen:
                        seen.add(t)
                        edge_list.append(count)
                        count += 1
                    else:
                        edge_data = list(seen)
                        edge_list.append(edge_data.index(t))

                face_edges[:] = np.array(edge_list)
                return face_edges
            else:
                # Raise NotImplementedError if grid_id is 1.
                raise NotImplementedError(
                    "get_grid_face_edges is not implemented for grid_id 1"
                )

        # If the loop doesn't return, raise an exception indicating an unexpected error
        raise Exception("Unexpected error in retrieving face edges.")

    # ------------------------------------------------------------
    def get_grid_face_nodes(
        self, grid_id: int, face_nodes: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Retrieve the nodes connected to faces for a specific grid, given its ID.

        This function accesses the grid's connectivity data and retrieves the nodes connected
        to the faces of the grid. If grid_id is 1, this function raises a NotImplementedError.

        :param grid_id: The ID of the grid for which face nodes are to be retrieved.
        :param face_nodes: A pre-allocated numpy array where the face nodes will be stored.
        :return: The updated `face_nodes` array with the retrieved face node values.
        :raises NotImplementedError: If grid_id is 1, as this functionality is not implemented for that case.
        :raises Exception: If an unexpected error occurs during the retrieval of face nodes.
        """
        for _ in self._grids:
            if grid_id != 1:
                mesh = nc.Dataset(self._job_meta.geogrid)
                elem_conn = mesh.variables[self._job_meta.elemconn_var][:]
                numelem_conn = mesh.variables[self._job_meta.numelemconn_var][:]
                node_conn_num = 0
                for i in range(elem_conn.shape[0]):
                    node_conn_num += numelem_conn[i]

                face_nodes[:] = np.empty(node_conn_num, dtype=int)
                index = 0
                for i in range(elem_conn.shape[0]):
                    for j in range(numelem_conn[i]):
                        face_nodes[index] = elem_conn[i, j]
                        index += 1
                return face_nodes
            else:
                raise NotImplementedError(
                    "get_grid_face_nodes is not implemented for grid_id 1"
                )

        # If the loop doesn't return, raise an exception indicating an unexpected error
        raise Exception("Unexpected error in retrieving face nodes.")

    # ------------------------------------------------------------
    def get_grid_node_count(self, grid_id: int) -> int:
        """Retrieve the number of nodes for the specified grid.

        This function accesses the grid and counts the number of nodes based on the node coordinates
        in the grid's data. The grid must not be of type 1, as this function is not implemented for that case.

        :param grid_id: The ID of the grid to retrieve node count for.
        :return: The number of nodes in the grid.
        :raises NotImplementedError: If grid_id is 1, the function raises an error as it is not implemented for that case.
        :raises ValueError: If grid ID is not found in `_grids` or if an unexpected error occurs.
        """
        for _ in self._grids:
            if grid_id != 1:
                # Open the geogrid file and retrieve node coordinates.
                mesh = nc.Dataset(self._job_meta.geogrid)
                node_count = len(mesh.variables[self._job_meta.nodecoords_var][:][:, 0])
                mesh.close()
                return node_count
            else:
                # Raise NotImplementedError if grid_id is 1.
                raise NotImplementedError(
                    "get_grid_node_count is not implemented for grid_id 1"
                )

        # If the loop doesn't return within the for loop, raise an exception
        raise ValueError("Grid ID not found in _grids.")

    # ------------------------------------------------------------
    def get_grid_nodes_per_face(
        self, grid_id: int, nodes_per_face: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Retrieve the number of nodes connected to each face in the specified grid, given its ID.

        This function accesses the grid's connectivity data and retrieves the number of nodes connected to
        each face. If grid_id is 1, this function raises a NotImplementedError.

        :param grid_id: The ID of the grid for which the nodes per face are to be retrieved.
        :param nodes_per_face: A pre-allocated numpy array where the number of nodes per face will be stored.
        :return: The updated `nodes_per_face` array with the number of nodes per face.
        :raises NotImplementedError: If grid_id is 1, as this functionality is not implemented for that case.
        :raises Exception: If an unexpected error occurs during the retrieval of nodes per face.
        """
        for _ in self._grids:
            if grid_id != 1:
                mesh = nc.Dataset(self._job_meta.geogrid)
                elem_conn = mesh.variables[self._job_meta.elemconn_var][:]
                numelem_conn = mesh.variables[self._job_meta.numelemconn_var][:]
                nodes_per_face[:] = np.empty(elem_conn.shape[0], dtype=int)
                for i in range(elem_conn.shape[0]):
                    nodes_per_face[i] = numelem_conn[i]
                return nodes_per_face
            else:
                # Raise NotImplementedError if grid_id is 1.
                raise NotImplementedError(
                    "get_grid_nodes_per_face is not implemented for grid_id 1"
                )

        # If the loop doesn't return, raise an exception indicating an unexpected error
        raise Exception("Unexpected error in retrieving nodes per face.")

    # ------------------------------------------------------------
    def get_grid_origin(
        self, grid_id: int, origin: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Retrieve the origin coordinates for the specified grid.

        This function accesses the grid and returns its origin coordinates, such as the minimum x, y, and z values.

        :param grid_id: The ID of the grid to retrieve the origin for.
        :param origin: A pre-allocated numpy array to store the origin values.
        :return: The updated numpy array containing the origin coordinates.
        :raises ValueError: If grid ID is not found.
        """
        for grid in self._grids:
            if grid_id == grid.id:
                origin[:] = grid.origin
                return origin
        raise ValueError(f"get_grid_origin: grid_id {grid_id} unknown")

    # ------------------------------------------------------------
    def get_grid_rank(self, grid_id: int) -> int:
        """Retrieve the rank of the specified grid.

        This function accesses the grid and returns its rank, which typically represents the number of dimensions.

        :param grid_id: The ID of the grid to retrieve the rank for.
        :return: The rank (integer) of the grid.
        :raises ValueError: If grid ID is not found.
        """
        for grid in self._grids:
            if grid_id == grid.id:
                return grid.rank
        raise ValueError(f"get_grid_rank: grid_id {grid_id} unknown")

    # ------------------------------------------------------------
    def get_grid_shape(self, grid_id: int, shape: NDArray[np.int_]) -> NDArray[np.int_]:
        """Retrieve the shape (dimensions) of the specified grid.

        This function accesses the grid and returns its shape (size in each dimension) as a numpy array.

        :param grid_id: The ID of the grid to retrieve the shape for.
        :param shape: A pre-allocated numpy array to store the shape values.
        :return: The updated numpy array containing the shape of the grid.
        :raises ValueError: If grid ID is not found.
        """
        for grid in self._grids:
            if grid_id == grid.id:
                shape[:] = grid.shape
                return shape
        raise ValueError(f"get_grid_shape: grid_id {grid_id} unknown")

    # ------------------------------------------------------------
    def get_grid_size(self, grid_id: int) -> int:
        """Retrieve the size (total number of elements) of the specified grid.

        This function accesses the grid and returns its total number of elements.

        :param grid_id: The ID of the grid to retrieve the size for.
        :return: The total size (integer) of the grid.
        :raises ValueError: If grid ID is not found.
        """
        for grid in self._grids:
            if grid_id == grid.id:
                return grid.size
        raise ValueError(f"get_grid_size: grid_id {grid_id} unknown")

    # ------------------------------------------------------------
    def get_grid_spacing(
        self, grid_id: int, spacing: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Retrieve the spacing (distance between grid points) for the specified grid.

        This function accesses the grid and returns its spacing values, typically representing the distance between adjacent grid points.

        :param grid_id: The ID of the grid to retrieve the spacing for.
        :param spacing: A pre-allocated numpy array to store the spacing values.
        :return: The updated numpy array containing the spacing between grid points.
        :raises ValueError: If grid ID is not found.
        """
        for grid in self._grids:
            if grid_id == grid.id:
                spacing[:] = grid.spacing
                return spacing
        raise ValueError(f"get_grid_spacing: grid_id {grid_id} unknown")

        # ------------------------------------------------------------

    def get_grid_type(self, grid_id: int) -> str:
        """Retrieve the type of the specified grid.

        This function accesses the grid and returns its type, which could be 'gridded', 'unstructured', etc.

        :param grid_id: The ID of the grid to retrieve the type for.
        :return: A string representing the type of the grid (e.g., 'gridded', 'unstructured').
        :raises ValueError: If grid ID is not found.
        """
        for grid in self._grids:
            if grid_id == grid.id:
                return grid.type
        raise ValueError(f"get_grid_type: grid_id {grid_id} unknown")

    # ------------------------------------------------------------
    def get_grid_x(self, grid_id: int, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Retrieve the x-coordinates (longitude or grid points) for the specified grid.

        This function accesses the grid and returns its x-coordinates in a numpy array.

        :param grid_id: The ID of the grid to retrieve the x-coordinates for.
        :param x: A pre-allocated numpy array to store the x-coordinates.
        :return: The updated numpy array containing the x-coordinates of the grid.
        :raises ValueError: If grid ID is not found.
        """
        for grid in self._grids:
            if grid_id == grid.id:
                if self._grid_type == "gridded":
                    x[:] = np.unique(grid.grid_x)
                else:
                    x[:] = grid.grid_x
                return x
        raise ValueError(f"get_grid_x: grid_id {grid_id} unknown")

    # ------------------------------------------------------------
    def get_grid_y(self, grid_id: int, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Retrieve the y-coordinates (latitude or grid points) for the specified grid.

        This function accesses the grid and returns its y-coordinates in a numpy array.

        :param grid_id: The ID of the grid to retrieve the y-coordinates for.
        :param y: A pre-allocated numpy array to store the y-coordinates.
        :return: The updated numpy array containing the y-coordinates of the grid.
        :raises ValueError: If grid ID is not found.
        """
        for grid in self._grids:
            if grid_id == grid.id:
                if self._grid_type == "gridded":
                    y[:] = np.unique(grid.grid_y)
                else:
                    y[:] = grid.grid_y
                return y
        raise ValueError(f"get_grid_y: grid_id {grid_id} unknown")

    # ------------------------------------------------------------
    def get_grid_z(self, grid_id: int, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Retrieve the z-coordinates (depth or grid points) for the specified grid.

        This function accesses the grid and returns its z-coordinates in a numpy array.

        :param grid_id: The ID of the grid to retrieve the z-coordinates for.
        :param z: A pre-allocated numpy array to store the z-coordinates.
        :return: The updated numpy array containing the z-coordinates of the grid.
        :raises ValueError: If grid ID is not found.
        """
        for grid in self._grids:
            if grid_id == grid.id:
                if self._grid_type == "gridded":
                    z[:] = np.unique(grid.grid_z)
                else:
                    z[:] = grid.grid_z
                return z
        raise ValueError(f"get_grid_z: grid_id {grid_id} unknown")

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # -- Random utility functions
    # ------------------------------------------------------------
    # ------------------------------------------------------------


def parse_config(cfg: dict) -> dict:
    """Parse the provided configuration dictionary (`cfg`) and modifies it based on certain rules.

    This function processes specific keys in the configuration dictionary:
    - Converts path-like strings to `PosixPath` objects.
    - Converts date strings to `pandas` datetime objects.
    - Configures lists of integers or strings for specific variables in the configuration.

    The function updates the `cfg` dictionary directly, modifying values as needed to match expected formats and types.

    :param cfg: A dictionary containing the configuration settings. The dictionary may include paths, dates, and lists of values.
    :return: The updated configuration dictionary with appropriately parsed values.
    """
    # LOG.debug(f"Entering _parse_config with cfg type: {type(cfg)}")
    if isinstance(cfg, str):
        LOG.error(
            f"Received string data (raw CSV) instead of dictionary: {cfg[:200]}..."
        )
        raise TypeError(
            "Expected dictionary in _parse_config, but got a raw CSV string."
        )

    # if not isinstance(cfg, dict):
    #     raise TypeError(f"[ERROR] Expected dictionary in _parse_config, got {type(cfg)} with contents: {cfg}")

    for key, val in cfg.items():
        # LOG.debug(f"Processing key: {key}, value type: {type(val)}, value: {val}")
        # Convert all path strings to PosixPath objects
        if any([key.endswith(x) for x in ["_dir", "_path", "_file", "_files"]]):
            if (val is not None) and (val != "None"):
                if isinstance(val, list):
                    temp_list = []
                    for element in val:
                        temp_list.append(Path(element))
                    cfg[key] = temp_list
                else:
                    cfg[key] = Path(val)
            else:
                cfg[key] = None

        # Convert Dates to pandas Datetime indices
        elif key.endswith("_date"):
            if isinstance(val, list):
                temp_list = []
                for elem in val:
                    temp_list.append(pd.to_datetime(elem, format="%d/%m/%Y"))
                cfg[key] = temp_list
            else:
                cfg[key] = pd.to_datetime(val, format="%d/%m/%Y")

        # Configure NWMv3.0 input configurations to what the ConfigClass expects
        # Flag for variables that need a list of integers
        elif key in [
            "InputForcings",
            "InputMandatory",
            "ForecastInputHorizons",
            "ForecastInputOffsets",
            "IgnoredBorderWidths",
            "RegridOpt",
            "TemperatureDownscaling",
            "ShortwaveDownscaling",
            "PressureDownscaling",
            "PrecipDownscaling",
            "HumidityDownscaling",
            "TemperatureBiasCorrection",
            "PressureBiasCorrection",
            "HumidityBiasCorrection",
            "WindBiasCorrection",
            "SwBiasCorrection",
            "LwBiasCorrection",
            "PrecipBiasCorrection",
            "SuppPcp",
            "RegridOptSuppPcp",
            "SuppPcpTemporalInterpolation",
            "SuppPcpMandatory",
            "SuppPcpInputOffsets",
            "custom_input_fcst_freq",
        ]:
            cfg[key] = val

        # Flag for variables that need to be a list of strings
        elif key in [
            "InputForcingDirectories",
            "InputForcingTypes",
            "DownscalingParamDirs",
            "SuppPcpForcingTypes",
            "SuppPcpDirectories",
        ]:
            cfg[key] = val
        else:
            pass

    # Add more config parsing if necessary
    return cfg


class NWMv3_Forcing_Engine_BMI_model_Gridded(NWMv3_Forcing_Engine_BMI_model_Base):
    """Defines the BMI (Basic Model Interface) for the NWMv3.0 Forcings Engine model.

    It includes methods for initializing the model, updating it, accessing model variables,
    and managing model configuration. This class is responsible for interacting with
    geospatial data and forcing inputs for the model simulation.
    """

    def __init__(self):
        """Create a model that is ready for initialization.

        Initializes the model with default values for time, variables, and grid types.
        """
        super().__init__()
        self.GeoMeta = GriddedGeoMeta

    def grid_ranks(self) -> list[int]:
        """Get the grid ranks for the gridded domain."""
        return [self.grid_4.rank]

    def grid_ids(self) -> list[int]:
        """Get the grid IDs for the gridded domain."""
        return [self.grid_1.id]

    def get_size_of_arrays(self) -> None:
        """Get the size of the flattened 2D arrays from the gridded domain."""
        self._varsize = len(np.zeros(self.geo_meta.latitude_grid.shape).flatten())

        for model_output in self.get_output_var_names():
            self._values[model_output] = np.zeros(self._varsize, dtype=float)

    def set_var_names(self) -> None:
        """Set the variable names for the BMI model based on the geospatial metadata.

        Create a Python dictionary that maps CSDMS Standard
        Names to the model's internal variable names.
        This is going to get long,
            since the input variable names could come from any forcing...
        """
        # Flag here to indicate whether or not the NWM operational configuration
        # will support a BMI field for liquid fraction of precipitation
        self._output_var_names = BMI_MODEL["_output_var_names"]
        self._var_name_units_map = BMI_MODEL["_var_name_units_map"]
        if self.config_options.include_lqfrac == 1:
            self._output_var_names += ["LQFRAC_ELEMENT"]
            self._var_name_units_map |= {
                "LQFRAC_ELEMENT": ["Liquid Fraction of Precipitation", "%"]
            }
        self.grid_1 = Grid(
            1, 2, GridType.uniform_rectilinear
        )  # Grid 1 is a 2-dimensional grid
        self.grid_1._grid_y = self.geo_meta.latitude_grid.flatten()
        self.grid_1._grid_x = self.geo_meta.longitude_grid.flatten()
        self.grid_1._shape = self.geo_meta.latitude_grid.shape
        self.grid_1._size = len(self.geo_meta.latitude_grid.flatten())
        self.grid_1._spacing = (
            self.geo_meta.dx_meters,
            self.geo_meta.dy_meters,
        )
        self.grid_1._units = "m"
        self.grid_1._origin = None

        self._grids = [self.grid_1]
        self._grid_map = {var_name: self.grid_1 for var_name in self._output_var_names}


class NWMv3_Forcing_Engine_BMI_model_HydroFabric(NWMv3_Forcing_Engine_BMI_model_Base):
    """Defines the BMI (Basic Model Interface) for the NWMv3.0 Forcings Engine model.

    It includes methods for initializing the model, updating it, accessing model variables,
    and managing model configuration. This class is responsible for interacting with
    geospatial data and forcing inputs for the model simulation.
    """

    def __init__(self):
        """Create a model that is ready for initialization.

        Initializes the model with default values for time, variables, and grid types.
        """
        super().__init__()
        self.GeoMeta = HydrofabricGeoMeta

    def grid_ranks(self) -> list[int]:
        """Get the grid ranks for the hydrofabric domain."""
        return [self.grid_4.rank]

    def grid_ids(self) -> list[int]:
        """Get the grid IDs for the hydrofabric domain."""
        return [self.grid_4.id]

    def get_size_of_arrays(self):
        """Get the size of the flattened 1D arrays from the hydrofabric domain."""
        self._varsize = len(np.zeros(self.geo_meta.latitude_grid.shape).flatten())
        for model_output in self.get_output_var_names():
            self._values[model_output] = np.zeros(self._varsize, dtype=float)

    def set_var_names(self):
        """Set the variables for the hydrofabric geospatial metadata.

        Create a Python dictionary that maps CSDMS Standard
        Names to the model's internal variable names.
        This is going to get long,
            since the input variable names could come from any forcing...
        """
        # Flag here to indicate whether or not the NWM operational configuration
        # will support a BMI field for liquid fraction of precipitation
        self._output_var_names = ["CAT-ID"] + BMI_MODEL["_output_var_names"]
        self._var_name_units_map = {"CAT-ID": ["Catchment ID", ""]} | BMI_MODEL[
            "_var_name_units_map"
        ]

        if self._job_meta.include_lqfrac == 1:
            self._output_var_names += ["LQFRAC_ELEMENT"]
            self._var_name_units_map |= {
                "LQFRAC_ELEMENT": ["Liquid Fraction of Precipitation", "%"],
            }

        self.grid_4 = Grid(
            4, 2, GridType.unstructured
        )  # Grid 1 is a 2-dimensional grid

        self.grid_4._grid_y = self.geo_meta.latitude_grid
        self.grid_4._grid_x = self.geo_meta.longitude_grid
        self.grid_4._size = len(self.geo_meta.latitude_grid)
        self._grids = [self.grid_4]
        self._grid_map = {var_name: self.grid_4 for var_name in self._output_var_names}


class NWMv3_Forcing_Engine_BMI_model_Unstructured(NWMv3_Forcing_Engine_BMI_model_Base):
    """Defines the BMI (Basic Model Interface) for the NWMv3.0 Forcings Engine model.

    It includes methods for initializing the model, updating it, accessing model variables,
    and managing model configuration. This class is responsible for interacting with
    geospatial data and forcing inputs for the model simulation.
    """

    def __init__(self):
        """Create a model that is ready for initialization.

        Initializes the model with default values for time, variables, and grid types.
        """
        super().__init__()
        self.GeoMeta = UnstructuredGeoMeta

    def grid_ranks(self) -> list[int]:
        """Get the grid ranks for the unstructured domain."""
        return [self.grid_2.rank, self.grid_3.rank]

    def grid_ids(self) -> list[int]:
        """Get the grid IDs for the unstructured domain."""
        return [self.grid_2.id, self.grid_3.id]

    def get_size_of_arrays(self) -> None:
        """Get the size of the flattened 1D arrays for the unstructured domain."""
        self._varsize = len(np.zeros(self.geo_meta.latitude_grid.shape).flatten())
        self._varsize_elem = len(
            np.zeros(self.geo_meta.latitude_grid_elem.shape).flatten()
        )

        for model_output in self.get_output_var_names():
            if "ELEMENT" in model_output:
                self._values[model_output] = np.zeros(self._varsize_elem, dtype=float)
            else:
                self._values[model_output] = np.zeros(self._varsize, dtype=float)

    def set_var_names(self) -> None:
        """Set the variable names for the unstructured domain.

        Create a Python dictionary that maps CSDMS Standard
        Names to the model's internal variable names.
        This is going to get long,
        since the input variable names could come from any forcing...
        """
        # Flag here to indicate whether or not the NWM operational configuration
        # will support a BMI field for liquid fraction of precipitation
        if self._job_meta.include_lqfrac == 1:
            output_var_names_position1 = ["LQFRAC_NODE"]
            output_var_names_position3 = ["LQFRAC_ELEMENT"]
            var_name_units_map_position1 = {
                "LQFRAC_NODE": ["Liquid Fraction of Precipitation", "%"]
            }
            var_name_units_map_position3 = {
                "LQFRAC_ELEMENT": ["Liquid Fraction of Precipitation", "%"]
            }
            grid_map_position1 = {"LQFRAC_ELEMENT": self.grid_2}
            grid_map_position3 = {"LQFRAC_NODE": self.grid_3}

        else:
            (
                output_var_names_position1,
                var_name_units_map_position1,
                var_name_units_map_position3,
                grid_map_position1,
                grid_map_position3,
            ) = [[]] + [{}] * 4

        self._output_var_names = (
            BMI_MODEL["_output_var_names_unstructured"]
            + output_var_names_position1
            + BMI_MODEL["_output_var_names"]
            + output_var_names_position3
        )
        self._var_name_units_map = (
            BMI_MODEL["_var_name_units_map_unstructured"]
            | var_name_units_map_position1
            | BMI_MODEL["_var_name_units_map"]
            | var_name_units_map_position3
        )
        self._grid_map = (
            {var_name: self.grid_2 for var_name in BMI_MODEL["_output_var_names"]}
            | grid_map_position1
            | {
                var_name: self.grid_3
                for var_name in BMI_MODEL["_output_var_names_unstructured"]
            }
            | grid_map_position3
        )

        self.grid_2 = Grid(
            2, 2, GridType.unstructured
        )  # Grid 1 is a 2-dimensional grid
        self.grid_3 = Grid(
            3, 2, GridType.unstructured
        )  # Grid 1 is a 2-dimensional grid

        self.grid_2._grid_y = self.geo_meta.latitude_grid_elem
        self.grid_2._grid_x = self.geo_meta.longitude_grid_elem

        self.grid_3._grid_y = self.geo_meta.latitude_grid
        self.grid_3._grid_x = self.geo_meta.longitude_grid

        self.grid_2._size = len(self.geo_meta.latitude_grid_elem)
        self.grid_3._size = len(self.geo_meta.latitude_grid)
        self._grids = [self.grid_2, self.grid_3]


BMIMODEL = {
    "gridded": NWMv3_Forcing_Engine_BMI_model_Gridded,
    "unstructured": NWMv3_Forcing_Engine_BMI_model_Unstructured,
    "hydrofabric": NWMv3_Forcing_Engine_BMI_model_HydroFabric,
}

### NOTE patch so ngen always accesses the Hydrofabric child for now.
### Other discretization modes currently do not have a ngen workflow.
NWMv3_Forcing_Engine_BMI_model = NWMv3_Forcing_Engine_BMI_model_HydroFabric
