# Need these for BMI
# This is needed for get_var_bytes
from pathlib import Path

# import data_tools
# Basic utilities
import numpy as np
import pandas as pd
import os
import netCDF4 as nc
# Configuration file functionality
import yaml
from bmipy import Bmi

# Import BMI grid functions to advertise grid features
from .bmi_grid import Grid, GridType

# Here is the model we want to run
from .model import NWMv3_Forcing_Engine_model

# Import MPI Python module
from mpi4py import MPI

###### NWMv3.0 Forcings Engine modules ######
try:
    import esmpy as ESMF
except ImportError:
    import ESMF

from .core import config
from .core import err_handler
from .core import forcingInputMod
from .core import geoMod
from .core import ioMod
from .core import parallel
from .core import suppPrecipMod

from typing import Any
from numpy.typing import NDArray

# If less than 0, then ESMF.__version__ is greater than 8.7.0
if ESMF.version_compare('8.7.0', ESMF.__version__) < 0:
    manager = ESMF.api.esmpymanager.Manager(endFlag=ESMF.constants.EndAction.KEEP_MPI)


class UnknownBMIVariable(RuntimeError):
    pass

class NWMv3_Forcing_Engine_BMI_model(Bmi):

    def __init__(self):
        """Create a model that is ready for initialization."""
        super(NWMv3_Forcing_Engine_BMI_model, self).__init__()
        self._values = {}
        self._start_time = 0.0
        self._end_time = np.finfo(float).max
        self._model = None
        self._comm = None
        self.var_array_lengths = 1

    #----------------------------------------------
    # Required, static attributes of the model
    #----------------------------------------------
    _att_map = {
        'model_name':         'NWMv3.0 Forcings Engine BMI Python',
        'version':            '1.0',
        'author_name':        'Jason Ducker',
        'grid_type':          'unstructured&uniform_rectilinear',
        'time_units':         'seconds',
               }

    #---------------------------------------------
    # Input variable names (CSDMS standard names)
    #---------------------------------------------
    # Forcings engine requires no inputs currently
    # and only provides model output
    _input_var_names = []

    _input_var_types = {}

    #------------------------------------------------------
    # A list of static attributes/parameters.
    #------------------------------------------------------
    _model_parameters_list = []

    #------------------------------------------------------------
    #------------------------------------------------------------
    # BMI: Model Control Functions
    #------------------------------------------------------------ 
    #------------------------------------------------------------

    #-------------------------------------------------------------------
    def initialize( self, bmi_cfg_file_name: str ):

        # -------------- Read in the BMI configuration -------------------------#
        if not isinstance(bmi_cfg_file_name, str) or len(bmi_cfg_file_name) == 0:
            raise RuntimeError("No BMI initialize configuration provided, nothing to do...")

        bmi_cfg_file = Path(bmi_cfg_file_name).resolve()
        if not bmi_cfg_file.is_file():
            raise RuntimeError("No configuration provided, nothing to do...")

        with bmi_cfg_file.open('r') as fp:
            cfg = yaml.safe_load(fp)
        self.cfg_bmi = self._parse_config(cfg)

        # Initialize the configuration object that will contain all
        # user-specified options within Forcings Engine BMI config file.
        self._job_meta = config.ConfigOptions(bmi_cfg_file)


        # Parse the configuration options
        try:
            self._job_meta.read_config(self.cfg_bmi)
        except KeyboardInterrupt:
            err_handler.err_out_screen('User keyboard interrupt')
        except ImportError:
            err_handler.err_out_screen('Missing Python packages')
        except InterruptedError:
            err_handler.err_out_screen('External kill signal detected')

        # Place NWM version number (if provided by the user). This will be placed into the final
        # output files as a global attribute.
        if self.cfg_bmi['NWM_VERSION'] is not None:
            self._job_meta.nwmVersion = self.cfg_bmi['NWM_VERSION']

        # Place NWM configuration (if provided by the user). This will be placed into the final
        # output files as a global attribute.
        if self.cfg_bmi['NWM_CONFIG'] is not None:
            self._job_meta.nwmConfig = self.cfg_bmi['NWM_CONFIG']
        # Initialize our MPI communication
        self._mpi_meta = parallel.MpiConfig()
        try:
            comm = MPI.Comm.f2py(self._comm) if self._comm is not None else None
            self._mpi_meta.initialize_comm(self._job_meta, comm=comm)
        except:
            err_handler.err_out_screen(self._job_meta.errMsg)
        # Initialize our WRF-Hydro geospatial object, which contains
        # information about the modeling domain, local processor
        # grid boundaries, and ESMF grid objects/fields to be used
        # in regridding.
        self._WrfHydroGeoMeta = geoMod.GeoMetaWrfHydro()

        if(self._job_meta.grid_type=='gridded'):
            try:
                self._WrfHydroGeoMeta.initialize_destination_geo_gridded(self._job_meta, self._mpi_meta)
            except Exception as e:
                raise e
                err_handler.err_out_screen_para(self._job_meta.errMsg, self._mpi_meta)
        elif(self._job_meta.grid_type=='unstructured'):
            try:
                self._WrfHydroGeoMeta.initialize_destination_geo_unstructured(self._job_meta, self._mpi_meta)
            except Exception as e:
                raise e
                err_handler.err_out_screen_para(self._job_meta.errMsg, self._mpi_meta)
        elif(self._job_meta.grid_type=='hydrofabric'):
            try:
                self._WrfHydroGeoMeta.initialize_destination_geo_hydrofabric(self._job_meta, self._mpi_meta)
            except Exception as e:
                raise e
                err_handler.err_out_screen_para(self._job_meta.errMsg, self._mpi_meta)
        else:
            self._job_meta.errMsg = "You must specify a proper grid_type (gridded,unstructured) within the config.yml file."
            err_handler.err_out_screen_para(self._job_meta.errMsg, self._mpi_meta)

        # Assign grid type to BMI class for grid information
        self._grid_type = self._job_meta.grid_type.lower()

        if(self._grid_type == "gridded"):
            #---------------------------------------------
            # Output variable names (CSDMS standard names)
            #---------------------------------------------
            
            # Flag here to indicate whether or not the NWM operational configuration
            # will support a BMI field for liquid fraction of precipitation
            if(self._job_meta.include_lqfrac == 1):
                self._output_var_names = ['U2D_ELEMENT', 'V2D_ELEMENT', 'LWDOWN_ELEMENT','SWDOWN_ELEMENT','T2D_ELEMENT','Q2D_ELEMENT','PSFC_ELEMENT','RAINRATE_ELEMENT','LQFRAC_ELEMENT']

                #------------------------------------------------------
                # Create a Python dictionary that maps CSDMS Standard
                # Names to the model's internal variable names.
                # This is going to get long,
                #     since the input variable names could come from any forcing...
                #------------------------------------------------------
                self._var_name_units_map = {'U2D_ELEMENT':['10-m U-component of wind','m/s'],
                                            'V2D_ELEMENT':['10-m V-component of wind','m/s'],
                                            'T2D_ELEMENT':['2-m Air Temperature','K'],
                                            'Q2D_ELEMENT':['2-m Specific Humidity','kg/kg'],
                                            'LWDOWN_ELEMENT':['Surface downward long-wave radiation flux','W/m^2'],
                                            'SWDOWN_ELEMENT':['Surface downward short-wave radiation flux','W/m^2'],
                                            'PSFC_ELEMENT':['Surface Pressure','Pa'],
                                            'RAINRATE_ELEMENT':['Surface Precipitation Rate','mm/s'],
                                            'LQFRAC_ELEMENT':['Liquid Fraction of Precipitation','%']}

                self.grid_1: Grid = Grid(1, 2, GridType.uniform_rectilinear) #Grid 1 is a 2 dimensional grid
                self.grid_1._grid_y = self._WrfHydroGeoMeta.latitude_grid.flatten()
                self.grid_1._grid_x = self._WrfHydroGeoMeta.longitude_grid.flatten()
                self.grid_1._shape = self._WrfHydroGeoMeta.latitude_grid.shape
                self.grid_1._size = len(self._WrfHydroGeoMeta.latitude_grid.flatten())
                self.grid_1._spacing = (self._WrfHydroGeoMeta.dx_meters,self._WrfHydroGeoMeta.dy_meters)
                self.grid_1._units = 'm'
                self.grid_1._origin = None

                self._grids = [self.grid_1]

                self._grid_map = {'U2D_ELEMENT': self.grid_1, 'V2D_ELEMENT': self.grid_1, 'LWDOWN_ELEMENT': self.grid_1, 'SWDOWN_ELEMENT': self.grid_1, 'T2D_ELEMENT': self.grid_1, 'Q2D_ELEMENT': self.grid_1, 'PSFC_ELEMENT': self.grid_1, 'RAINRATE_ELEMENT': self.grid_1, 'LQFRAC_ELEMENT': self.grid_1}

            else:
                self._output_var_names = ['U2D_ELEMENT', 'V2D_ELEMENT', 'LWDOWN_ELEMENT','SWDOWN_ELEMENT','T2D_ELEMENT','Q2D_ELEMENT','PSFC_ELEMENT','RAINRATE_ELEMENT']

                #------------------------------------------------------
                # Create a Python dictionary that maps CSDMS Standard
                # Names to the model's internal variable names.
                # This is going to get long,
                #     since the input variable names could come from any forcing...
                #------------------------------------------------------
                self._var_name_units_map = {'U2D_ELEMENT':['10-m U-component of wind','m/s'],
                                            'V2D_ELEMENT':['10-m V-component of wind','m/s'],
                                            'T2D_ELEMENT':['2-m Air Temperature','K'],
                                            'Q2D_ELEMENT':['2-m Specific Humidity','kg/kg'],
                                            'LWDOWN_ELEMENT':['Surface downward long-wave radiation flux','W/m^2'],
                                            'SWDOWN_ELEMENT':['Surface downward short-wave radiation flux','W/m^2'],
                                            'PSFC_ELEMENT':['Surface Pressure','Pa'],
                                            'RAINRATE_ELEMENT':['Surface Precipitation Rate','mm/s']}

                self.grid_1: Grid = Grid(1, 2, GridType.uniform_rectilinear) #Grid 1 is a 2 dimensional grid
                self.grid_1._grid_y = self._WrfHydroGeoMeta.latitude_grid.flatten()
                self.grid_1._grid_x = self._WrfHydroGeoMeta.longitude_grid.flatten()
                self.grid_1._shape = self._WrfHydroGeoMeta.latitude_grid.shape
                self.grid_1._size = len(self._WrfHydroGeoMeta.latitude_grid.flatten())
                self.grid_1._spacing = (self._WrfHydroGeoMeta.dx_meters,self._WrfHydroGeoMeta.dy_meters)
                self.grid_1._units = 'm'
                self.grid_1._origin = None

                self._grids = [self.grid_1]

                self._grid_map = {'U2D_ELEMENT': self.grid_1, 'V2D_ELEMENT': self.grid_1, 'LWDOWN_ELEMENT': self.grid_1, 'SWDOWN_ELEMENT': self.grid_1, 'T2D_ELEMENT': self.grid_1, 'Q2D_ELEMENT': self.grid_1, 'PSFC_ELEMENT': self.grid_1, 'RAINRATE_ELEMENT': self.grid_1}

        elif(self._grid_type == "unstructured"):
            # Flag here to indicate whether or not the NWM operational configuration
            # will support a BMI field for liquid fraction of precipitation
            if(self._job_meta.include_lqfrac == 1):
                #---------------------------------------------
                # Output variable names (CSDMS standard names)
                #---------------------------------------------
                self._output_var_names = ['U2D_NODE', 'V2D_NODE', 'LWDOWN_NODE','SWDOWN_NODE','T2D_NODE','Q2D_NODE','PSFC_NODE','RAINRATE_NODE','LQFRAC_NODE','U2D_ELEMENT', 'V2D_ELEMENT', 'LWDOWN_ELEMENT','SWDOWN_ELEMENT','T2D_ELEMENT','Q2D_ELEMENT','PSFC_ELEMENT','RAINRATE_ELEMENT','LQFRAC_ELEMENT']

                #------------------------------------------------------
                # Create a Python dictionary that maps CSDMS Standard
                # Names to the model's internal variable names.
                # This is going to get long,
                #     since the input variable names could come from any forcing...
                #------------------------------------------------------
                self._var_name_units_map = {'U2D_NODE':['10-m U-component of wind','m/s'],
                                            'V2D_NODE':['10-m V-component of wind','m/s'],
                                            'T2D_NODE':['2-m Air Temperature','K'],
                                            'Q2D_NODE':['2-m Specific Humidity','kg/kg'],
                                            'LWDOWN_NODE':['Surface downward long-wave radiation flux','W/m^2'],
                                            'SWDOWN_NODE':['Surface downward short-wave radiation flux','W/m^2'],
                                            'PSFC_NODE':['Surface Pressure','Pa'],
                                            'RAINRATE_NODE':['Surface Precipitation Rate','mm/s'],
                                            'LQFRAC_NODE':['Liquid Fraction of Precipitation','%'],
                                            'U2D_ELEMENT':['10-m U-component of wind','m/s'],
                                            'V2D_ELEMENT':['10-m V-component of wind','m/s'],
                                            'T2D_ELEMENT':['2-m Air Temperature','K'],
                                            'Q2D_ELEMENT':['2-m Specific Humidity','kg/kg'],
                                            'LWDOWN_ELEMENT':['Surface downward long-wave radiation flux','W/m^2'],
                                            'SWDOWN_ELEMENT':['Surface downward short-wave radiation flux','W/m^2'],
                                            'PSFC_ELEMENT':['Surface Pressure','Pa'],
                                            'RAINRATE_ELEMENT':['Surface Precipitation Rate','mm/s'],
                                            'LQFRAC_ELEMENT':['Liquid Fraction of Precipitation','%']}

                self.grid_2: Grid = Grid(2, 2, GridType.unstructured) #Grid 1 is a 2 dimensional grid
                self.grid_3: Grid = Grid(3, 2, GridType.unstructured) #Grid 1 is a 2 dimensional grid

                self.grid_2._grid_y = self._WrfHydroGeoMeta.latitude_grid_elem
                self.grid_2._grid_x = self._WrfHydroGeoMeta.longitude_grid_elem

                self.grid_3._grid_y = self._WrfHydroGeoMeta.latitude_grid
                self.grid_3._grid_x = self._WrfHydroGeoMeta.longitude_grid

                self.grid_2._size = len(self._WrfHydroGeoMeta.latitude_grid_elem)
                self.grid_3._size = len(self._WrfHydroGeoMeta.latitude_grid)

                self._grids = [self.grid_2, self.grid_3]

                self._grid_map = {'U2D_ELEMENT': self.grid_2, 'V2D_ELEMENT': self.grid_2, 'LWDOWN_ELEMENT': self.grid_2, 'SWDOWN_ELEMENT': self.grid_2, 'T2D_ELEMENT': self.grid_2, 'Q2D_ELEMENT': self.grid_2, 'PSFC_ELEMENT': self.grid_2, 'RAINRATE_ELEMENT': self.grid_2, 'LQFRAC_ELEMENT': self.grid_2, 'U2D_NODE': self.grid_3, 'V2D_NODE': self.grid_3, 'LWDOWN_NODE': self.grid_3, 'SWDOWN_NODE': self.grid_3, 'T2D_NODE': self.grid_3, 'Q2D_NODE': self.grid_3, 'PSFC_NODE': self.grid_3, 'RAINRATE_NODE': self.grid_3, 'LQFRAC_NODE': self.grid_3}          
            else:
                #---------------------------------------------
                # Output variable names (CSDMS standard names)
                #---------------------------------------------
                self._output_var_names = ['U2D_NODE', 'V2D_NODE', 'LWDOWN_NODE','SWDOWN_NODE','T2D_NODE','Q2D_NODE','PSFC_NODE','RAINRATE_NODE','U2D_ELEMENT', 'V2D_ELEMENT', 'LWDOWN_ELEMENT','SWDOWN_ELEMENT','T2D_ELEMENT','Q2D_ELEMENT','PSFC_ELEMENT','RAINRATE_ELEMENT']

                #------------------------------------------------------
                # Create a Python dictionary that maps CSDMS Standard
                # Names to the model's internal variable names.
                # This is going to get long,
                #     since the input variable names could come from any forcing...
                #------------------------------------------------------
                self._var_name_units_map = {'U2D_NODE':['10-m U-component of wind','m/s'],
                                            'V2D_NODE':['10-m V-component of wind','m/s'],
                                            'T2D_NODE':['2-m Air Temperature','K'],
                                            'Q2D_NODE':['2-m Specific Humidity','kg/kg'],
                                            'LWDOWN_NODE':['Surface downward long-wave radiation flux','W/m^2'],
                                            'SWDOWN_NODE':['Surface downward short-wave radiation flux','W/m^2'],
                                            'PSFC_NODE':['Surface Pressure','Pa'],
                                            'RAINRATE_NODE':['Surface Precipitation Rate','mm/s'],
                                            'U2D_ELEMENT':['10-m U-component of wind','m/s'],
                                            'V2D_ELEMENT':['10-m V-component of wind','m/s'],
                                            'T2D_ELEMENT':['2-m Air Temperature','K'],
                                            'Q2D_ELEMENT':['2-m Specific Humidity','kg/kg'],
                                            'LWDOWN_ELEMENT':['Surface downward long-wave radiation flux','W/m^2'],
                                            'SWDOWN_ELEMENT':['Surface downward short-wave radiation flux','W/m^2'],
                                            'PSFC_ELEMENT':['Surface Pressure','Pa'],
                                            'RAINRATE_ELEMENT':['Surface Precipitation Rate','mm/s']}

                self.grid_2: Grid = Grid(2, 2, GridType.unstructured) #Grid 1 is a 2 dimensional grid
                self.grid_3: Grid = Grid(3, 2, GridType.unstructured) #Grid 1 is a 2 dimensional grid

                self.grid_2._grid_y = self._WrfHydroGeoMeta.latitude_grid_elem
                self.grid_2._grid_x = self._WrfHydroGeoMeta.longitude_grid_elem

                self.grid_3._grid_y = self._WrfHydroGeoMeta.latitude_grid
                self.grid_3._grid_x = self._WrfHydroGeoMeta.longitude_grid

                self.grid_2._size = len(self._WrfHydroGeoMeta.latitude_grid_elem)
                self.grid_3._size = len(self._WrfHydroGeoMeta.latitude_grid)

                self._grids = [self.grid_2, self.grid_3]

                self._grid_map = {'U2D_ELEMENT': self.grid_2, 'V2D_ELEMENT': self.grid_2, 'LWDOWN_ELEMENT': self.grid_2, 'SWDOWN_ELEMENT': self.grid_2, 'T2D_ELEMENT': self.grid_2, 'Q2D_ELEMENT': self.grid_2, 'PSFC_ELEMENT': self.grid_2, 'RAINRATE_ELEMENT': self.grid_2, 'U2D_NODE': self.grid_3, 'V2D_NODE': self.grid_3, 'LWDOWN_NODE': self.grid_3, 'SWDOWN_NODE': self.grid_3, 'T2D_NODE': self.grid_3, 'Q2D_NODE': self.grid_3, 'PSFC_NODE': self.grid_3, 'RAINRATE_NODE': self.grid_3}

        elif(self._grid_type == "hydrofabric"):
            # Flag here to indicate whether or not the NWM operational configuration
            # will support a BMI field for liquid fraction of precipitation
            if(self._job_meta.include_lqfrac == 1):
                #---------------------------------------------
                # Output variable names (CSDMS standard names)
                #---------------------------------------------
                self._output_var_names = ['CAT-ID','U2D_ELEMENT', 'V2D_ELEMENT', 'LWDOWN_ELEMENT','SWDOWN_ELEMENT','T2D_ELEMENT','Q2D_ELEMENT','PSFC_ELEMENT','RAINRATE_ELEMENT','LQFRAC_ELEMENT']

                #------------------------------------------------------
                # Create a Python dictionary that maps CSDMS Standard
                # Names to the model's internal variable names.
                # This is going to get long,
                #     since the input variable names could come from any forcing...
                #------------------------------------------------------
                self._var_name_units_map = {'CAT-ID':['Catchment ID',''],
                                            'U2D_ELEMENT':['10-m U-component of wind','m/s'],
                                            'V2D_ELEMENT':['10-m V-component of wind','m/s'],
                                            'T2D_ELEMENT':['2-m Air Temperature','K'],
                                            'Q2D_ELEMENT':['2-m Specific Humidity','kg/kg'],
                                            'LWDOWN_ELEMENT':['Surface downward long-wave radiation flux','W/m^2'],
                                            'SWDOWN_ELEMENT':['Surface downward short-wave radiation flux','W/m^2'],
                                            'PSFC_ELEMENT':['Surface Pressure','Pa'],
                                            'RAINRATE_ELEMENT':['Surface Precipitation Rate','mm/s'],
                                            'LQFRAC_ELEMENT':['Liquid Fraction of Precipitation','%']}

                self.grid_4: Grid = Grid(4, 2, GridType.unstructured) #Grid 1 is a 2 dimensional grid

                self.grid_4._grid_y = self._WrfHydroGeoMeta.latitude_grid
                self.grid_4._grid_x = self._WrfHydroGeoMeta.longitude_grid

                self.grid_4._size = len(self._WrfHydroGeoMeta.latitude_grid)

                self._grids = [self.grid_4]

                self._grid_map = {'CAT-ID': self.grid_4, 'U2D_ELEMENT': self.grid_4, 'V2D_ELEMENT': self.grid_4, 'LWDOWN_ELEMENT': self.grid_4, 'SWDOWN_ELEMENT': self.grid_4, 'T2D_ELEMENT': self.grid_4, 'Q2D_ELEMENT': self.grid_4, 'PSFC_ELEMENT': self.grid_4, 'RAINRATE_ELEMENT': self.grid_4, 'LQFRAC_ELEMENT': self.grid_4}
            else:
                #---------------------------------------------
                # Output variable names (CSDMS standard names)
                #---------------------------------------------
                self._output_var_names = ['CAT-ID','U2D_ELEMENT', 'V2D_ELEMENT', 'LWDOWN_ELEMENT','SWDOWN_ELEMENT','T2D_ELEMENT','Q2D_ELEMENT','PSFC_ELEMENT','RAINRATE_ELEMENT']

                #------------------------------------------------------
                # Create a Python dictionary that maps CSDMS Standard
                # Names to the model's internal variable names.
                # This is going to get long,
                #     since the input variable names could come from any forcing...
                #------------------------------------------------------
                self._var_name_units_map = {'CAT-ID':['Catchment ID',''],
                                            'U2D_ELEMENT':['10-m U-component of wind','m/s'],
                                            'V2D_ELEMENT':['10-m V-component of wind','m/s'],
                                            'T2D_ELEMENT':['2-m Air Temperature','K'],
                                            'Q2D_ELEMENT':['2-m Specific Humidity','kg/kg'],
                                            'LWDOWN_ELEMENT':['Surface downward long-wave radiation flux','W/m^2'],
                                            'SWDOWN_ELEMENT':['Surface downward short-wave radiation flux','W/m^2'],
                                            'PSFC_ELEMENT':['Surface Pressure','Pa'],
                                            'RAINRATE_ELEMENT':['Surface Precipitation Rate','mm/s']}

                self.grid_4: Grid = Grid(4, 2, GridType.unstructured) #Grid 1 is a 2 dimensional grid

                self.grid_4._grid_y = self._WrfHydroGeoMeta.latitude_grid
                self.grid_4._grid_x = self._WrfHydroGeoMeta.longitude_grid

                self.grid_4._size = len(self._WrfHydroGeoMeta.latitude_grid)

                self._grids = [self.grid_4]

                self._grid_map = {'CAT-ID': self.grid_4, 'U2D_ELEMENT': self.grid_4, 'V2D_ELEMENT': self.grid_4, 'LWDOWN_ELEMENT': self.grid_4, 'SWDOWN_ELEMENT': self.grid_4, 'T2D_ELEMENT': self.grid_4, 'Q2D_ELEMENT': self.grid_4, 'PSFC_ELEMENT': self.grid_4, 'RAINRATE_ELEMENT': self.grid_4}

        # ----- Create some lookup tabels from the long variable names --------#
        self._var_name_map_long_first = {long_name:self._var_name_units_map[long_name][0] for long_name in self._var_name_units_map.keys()}
        self._var_name_map_short_first = {self._var_name_units_map[long_name][0]:long_name for long_name in self._var_name_units_map.keys()}
        self._var_units_map = {long_name:self._var_name_units_map[long_name][1] for long_name in self._var_name_units_map.keys()}
    
        if self._job_meta.spatial_meta is not None:
            try:
                self._WrfHydroGeoMeta.initialize_geospatial_metadata(self._job_meta, self._mpi_meta)
            except Exception:
                err_handler.err_out_screen_para(self._job_meta.errMsg, self._mpi_meta)
        err_handler.check_program_status(self._job_meta, self._mpi_meta)
  
        # Check to make sure we have enough dimensionality to run regridding. ESMF requires both grids
        # to have a size of at least 2.
        if self._WrfHydroGeoMeta.nx_local < 2 or self._WrfHydroGeoMeta.ny_local < 2:
            self._job_meta.errMsg = "You have specified too many cores for your WRF-Hydro grid. " \
                             "Local grid Must have x/y dimension size of 2."
            err_handler.err_out_screen_para(self._job_meta.errMsg, self._mpi_meta)
        err_handler.check_program_status(self._job_meta, self._mpi_meta)
 
        # Initialize our output object, which includes local slabs from the output grid.
        try:
            self._OutputObj = ioMod.OutputObj(self._job_meta,self._WrfHydroGeoMeta)
        except Exception:
            err_handler.err_out_screen_para(self._job_meta, self._mpi_meta)
        err_handler.check_program_status(self._job_meta, self._mpi_meta)

        # If user requests output for given domain, then call
        # the I/O module to initialize netcdf file with given
        # geospatial fields of the domain
        if(self._job_meta.forcing_output == 1):
            # First, name the file based on domain configuration and start time requested
            # Compose the expected path to the output file. Check to see if the file exists,
            # if so, continue to the next time step. Also initialize our output arrays if necessary.
            if(self._job_meta.grid_type=='gridded'):
                ext = 'GRIDDED'
            elif(self._job_meta.grid_type=='hydrofabric'):
                ext = 'HYDROFABRIC'
            elif(self._job_meta.grid_type=='unstructured'):
                ext = 'MESH'
            self._OutputObj.outPath = self._job_meta.scratch_dir + "/NextGen_Forcings_Engine_" + ext + "_output_" + pd.Timestamp(self._job_meta.b_date_proc).strftime('%Y%m%d%H%M') + ".nc"

            self._OutputObj.init_forcing_file(self._job_meta,self._WrfHydroGeoMeta,self._mpi_meta)


        # Next, initialize our input forcing classes. These objects will contain
        # information about our source products (I.E. data type, grid sizes, etc).
        # Information will be mapped via the options specified by the user.
        # In addition, input ESMF grid objects will be created to hold data for
        # downscaling and regridding purposes.
        try:
            self._inputForcingMod = forcingInputMod.initDict(self._job_meta,self._WrfHydroGeoMeta, self._mpi_meta)
        except Exception:
            err_handler.err_out_screen_para(self._job_meta, self._mpi_meta)
        err_handler.check_program_status(self._job_meta, self._mpi_meta)

        # If we have specified supplemental precipitation products, initialize
        # the supp class.
        if self._job_meta.number_supp_pcp > 0:
            self._suppPcpMod = suppPrecipMod.initDict(self._job_meta,self._WrfHydroGeoMeta)
        else:
            self._suppPcpMod = None
        err_handler.check_program_status(self._job_meta, self._mpi_meta)

        # ------------- Initialize the parameters, inputs and outputs ----------#
        for parm in self._model_parameters_list:
            self._values[self._var_name_map_short_first[parm]] = self.cfg_bmi[parm]

        if(self._job_meta.grid_type=='gridded'):
            #-----------------------------------------------------------------------#
            # Get the size of the flattened 2D arrays from the gridded domain
            self._varsize = len(np.zeros(self._WrfHydroGeoMeta.latitude_grid.shape).flatten())

            for model_output in self.get_output_var_names():
                self._values[model_output] = np.zeros(self._varsize, dtype=float)

        elif(self._job_meta.grid_type=='unstructured'):
            #-----------------------------------------------------------------------#
            # Get the size of the flattened 1D arrays from the unstructured domain
            self._varsize = len(np.zeros(self._WrfHydroGeoMeta.latitude_grid.shape).flatten())
            self._varsize_elem = len(np.zeros(self._WrfHydroGeoMeta.latitude_grid_elem.shape).flatten())

            for model_output in self.get_output_var_names():
                if("ELEMENT" in model_output):
                    self._values[model_output] = np.zeros(self._varsize_elem, dtype=float)
                else:
                    self._values[model_output] = np.zeros(self._varsize, dtype=float)

        elif(self._job_meta.grid_type=='hydrofabric'):
            #-----------------------------------------------------------------------#
            # Get the size of the flattened 1D arrays from the hydrofabric domain
            self._varsize = len(np.zeros(self._WrfHydroGeoMeta.latitude_grid.shape).flatten())
            for model_output in self.get_output_var_names():
                self._values[model_output] = np.zeros(self._varsize, dtype=float)

        #for model_input in self.get_input_var_names():
        #    self._values[model_input] = np.zeros(self._varsize, dtype=float)

        # ------------- Set time to initial value -----------------------#
        self._values['current_model_time'] = self.cfg_bmi['initial_time']
        
        # ------------- Set time step size -----------------------#
        self._values['time_step_size'] = self.cfg_bmi['time_step_seconds']

        # ------------- Initialize a model ------------------------------#
        #self._model = ngen_model(self._values.keys())
        self._model = NWMv3_Forcing_Engine_model()

        # Now set the catchment ids to the BMI output field
        # so they're initialized for the model engine to reference
        if(self._grid_type == "hydrofabric"):
            self._values['CAT-ID'] = self._WrfHydroGeoMeta.element_ids

    #------------------------------------------------------------ 
    def update(self):
        """
        Update/advance the model by one time step.
        """
        self._values['current_model_time'] += self._values['time_step_size']
     
        self.update_until(self._values['current_model_time'])
    
    #------------------------------------------------------------ 
    def update_until(self, future_time: float):
        """
        Update the model to a particular time

        Parameters
        ----------
        future_time : float
            The future time to when the model would be advanced.
        """
        # Flag to see if update is just a single model time step
        # otherwise we must perform a time loop to iterate data until
        # requested time stamp
        if(future_time != self._values['current_model_time']):
            while(self._values['current_model_time'] < future_time):
                self._values['current_model_time'] += self._values['time_step_size']
                self._model.run(self._values, self._values['current_model_time'], self._job_meta, self._WrfHydroGeoMeta, self._inputForcingMod, self._suppPcpMod, self._mpi_meta, self._OutputObj)
        # This is just a single model time step (1 hour) update
        else:
            self._model.run(self._values, future_time, self._job_meta, self._WrfHydroGeoMeta, self._inputForcingMod, self._suppPcpMod, self._mpi_meta, self._OutputObj)
    #------------------------------------------------------------    
    def finalize( self ):
        """Finalize model."""
        # Remove scratch directory files once BMI is completed to avoid 
        # storage issues for NextGen formulation run, but only on root thread
        if(self._mpi_meta.rank == 0):
            for filename in os.listdir(self._job_meta.scratch_dir):
                file_path = os.path.join(self._job_meta.scratch_dir, filename)
                if(os.path.isfile(file_path) and filename[0:23] != "NextGen_Forcings_Engine"):
                    os.remove(file_path)
                elif(os.path.isdir(file_path)):
                    os.rmdir(file_path)

        # Force destruction of ESMF objects
        try:
            del self._WrfHydroGeoMeta
        except AttributeError:
            pass

        try:
            del self._inputForcingMod
        except AttributeError:
            pass

        try:
            del self._suppPcpMod
        except AttributeError:
            pass
    
        self._model = None
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # BMI: Model Information Functions
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    
    def get_attribute(self, att_name):
    
        try:
            return self._att_map[ att_name.lower() ]
        except:
            print(' ERROR: Could not find attribute: ' + att_name)

    #--------------------------------------------------------
    # Note: These are currently variables needed from other
    #       components vs. those read from files or GUI.
    #--------------------------------------------------------   
    def get_input_var_names(self):

        return self._input_var_names

    def get_output_var_names(self):
 
        return self._output_var_names

    #------------------------------------------------------------ 
    def get_component_name(self):
        """Name of the component."""
        return self.get_attribute( 'model_name' ) #JG Edit

    #------------------------------------------------------------ 
    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    #------------------------------------------------------------ 
    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    #------------------------------------------------------------ 
    def get_value(self, var_name: str, dest: NDArray[Any]) -> NDArray[Any]:
        """Copy of values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        Returns
        -------
        array_like
            Copy of values.
        """
        if var_name == "grid:count":
            if(self._job_meta.grid_type != 'unstructured'):
                dest[...] = 1
            else:
                dest[...] = 2
        elif var_name == "grid:ids":
            if(self._job_meta.grid_type == 'gridded'):
                dest[:] = [self.grid_1.id]
            elif(self._job_meta.grid_type == 'unstructured'):
                dest[:] = [self.grid_2.id,self.grid_3.id]
            elif(self._job_meta.grid_type == 'hydrofabric'):
                dest[:] = [self.grid_4.id]
        elif var_name == "grid:ranks":
            if(self._job_meta.grid_type == 'gridded'):
                dest[:] = [self.grid_1.rank]
            elif(self._job_meta.grid_type == 'unstructured'):
                dest[:] = [self.grid_2.rank,self.grid_3.rank]
            elif(self._job_meta.grid_type == 'hydrofabric'):
                dest[:] = [self.grid_4.rank]
        else:
            dest[:] = self.get_value_ptr(var_name)
        return dest

    #-------------------------------------------------------------------
    def get_value_ptr(self, var_name: str) -> NDArray[Any]:
        """Reference to values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        array_like
            Value array.
        """
        #Make sure to return a flattened array
        if(var_name == "grid_1_shape"): # FIXME cannot expose shape as ptr, because it has to side affect variable construction...
            return self.grid_1.shape
        if(var_name == "grid_1_spacing"):
            return self.grid_1.spacing
        if(var_name == "grid_1_origin"):
            return self.grid_1.origin
        if(var_name == "grid_1_units"):
            return self.grid_1.units
        if(var_name == "grid_2_shape"):
            return self.grid_2.shape
        if(var_name == "grid_2_spacing"):
            return self.grid_2.spacing
        if(var_name == "grid_2_origin"):
            return self.grid_2.origin
        if(var_name == "grid_2_units"):
            return self.grid_2.units 
        if(var_name == "grid_3_shape"):
            return self.grid_3.shape
        if(var_name == "grid_3_spacing"):
            return self.grid_3.spacing
        if(var_name == "grid_3_origin"):
            return self.grid_3.origin
        if(var_name == "grid_3_units"):
            return self.grid_3.units
        if(var_name == "grid_4_shape"):
            return self.grid_4.shape
        if(var_name == "grid_4_spacing"):
            return self.grid_4.spacing
        if(var_name == "grid_4_origin"):
            return self.grid_4.origin
        if(var_name == "grid_4_units"):
            return self.grid_4.units

        if var_name not in self._values.keys():
            raise(UnknownBMIVariable(f"No known variable in BMI model: {var_name}"))

        shape = self._values[var_name].shape
        try:
            #see if raveling is possible without a copy
            self._values[var_name].shape = (-1,)
            #reset original shape
            self._values[var_name].shape = shape
        except ValueError as e:
            raise RuntimeError("Cannot flatten array without copying -- "+str(e).split(": ")[-1])

        return self._values[var_name].ravel()#.reshape((-1,))

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # BMI: Variable Information Functions
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    def get_var_name(self, long_var_name):
                              
        return self._var_name_map_long_first[ long_var_name ]

    #-------------------------------------------------------------------
    def get_var_units(self, long_var_name):

        return self._var_units_map[ long_var_name ]
                                                             
    #-------------------------------------------------------------------
    def get_var_type(self, var_name: str) -> str:
        """Data type of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Data type.
        """
        return str(self.get_value_ptr(var_name).dtype)
    
    #------------------------------------------------------------ 
    def get_var_grid(self, name):
        
        # all vars have grid 0 but check if its in names list first
        if name in (self._output_var_names):
            if("ELEMENT" in name and self._job_meta.grid_type == "gridded"):
                return 1
            elif("ELEMENT" in name and self._job_meta.grid_type == "unstructured"):
                return 2
            elif("NODE" in name and self._job_meta.grid_type == "unstructured"):
                return 3
            elif("ELEMENT" in name and self._job_meta.grid_type == "hydrofabric"):
                return 4
            else:
                return self._var_grid_id
        raise(UnknownBMIVariable(f"No known variable in BMI model: {name}"))

    #------------------------------------------------------------ 
    def get_var_itemsize(self, name):
        return self.get_value_ptr(name).itemsize

    #------------------------------------------------------------ 
    def get_var_location(self, name):
        
        if("ELEMENT" in name):
            return "face"
        elif("NODE" in name):
            return "node"
        else:
            raise ValueError(f"get_var_location: grid_id {self._var_grid_id} unknown")

    #-------------------------------------------------------------------
    def get_var_rank(self, long_var_name):

        return np.int16(0)

    #-------------------------------------------------------------------
    def get_start_time(self) -> float:
    
        return self._start_time 

    #-------------------------------------------------------------------
    def get_end_time(self) -> float:

        return self._end_time 


    #-------------------------------------------------------------------
    def get_current_time(self) -> float:

        return self._values['current_model_time']

    #-------------------------------------------------------------------
    def get_time_step(self) -> float:

        return self._values['time_step_size']

    #-------------------------------------------------------------------
    def get_time_units(self) -> str:

        return self.get_attribute( 'time_units' ) 
       
    #-------------------------------------------------------------------
    def set_value(self, var_name: str, values: NDArray[Any]):
        """
        Set model values for the provided BMI variable.

        Parameters
        ----------
        var_name : str
            Name of model variable for which to set values.
        values : NDArray[Any]
              Array of new values.
        """
        if var_name == 'bmi_mpi_comm':
            self._comm = values[0]
        else:
            self._values[var_name][:] = values

    #------------------------------------------------------------ 
    def set_value_at_indices(self, var_name: str, indices: NDArray[np.int_], src: NDArray[Any]):
        """
        Set model values for the provided BMI variable at particular indices.

        Parameters
        ----------
        var_name : str
            Name of model variable for which to set values.
        indices : array_like
            Array of indices of the variable into which analogous provided values should be set.
        src : array_like
            Array of new values.
        """
        # This is not particularly efficient, but it is functionally correct.
        for i in range(indices.shape[0]):
            bmi_var_value_index = indices[i]
            self.get_value_ptr(var_name)[bmi_var_value_index] = src[i]

    #------------------------------------------------------------ 
    def get_var_nbytes(self, var_name) -> int:
        """
        Get the number of bytes required for a variable.
        Parameters
        ----------
        var_name : str
            Name of variable.
        Returns
        -------
        int
            Size of data array in bytes.
        """
        return self.get_value_ptr(var_name).nbytes

    #------------------------------------------------------------ 
    def get_value_at_indices(self, var_name: str, dest: NDArray[Any], indices: NDArray[np.int_]) -> NDArray[Any]:
        """Get values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : NDArray[Any]
            A numpy array into which to place the values.
        indices : NDArray[np.int_]
            Array of indices.
        Returns
        -------
        NDArray[Any]
            Values at indices.
        """
        original: NDArray[Any] = self.get_value_ptr(var_name)
        for i in range(indices.shape[0]):
            value_index = indices[i]
            dest[i] = original[value_index]
        return dest

    # JG Note: remaining grid funcs do not apply for type 'scalar'
    #   Yet all functions in the BMI must be implemented 
    #   See https://bmi.readthedocs.io/en/latest/bmi.best_practices.html          
    #------------------------------------------------------------ 
    def get_grid_edge_count(self, grid_id: int) -> int:
        for grid in self._grids:
            if grid_id != 1:
                mesh = nc.Dataset(self._job_meta.geogrid)
                elem_conn = mesh.variables[self._job_meta.elemconn_var][:]
                numelem_conn = mesh.variables[self._job_meta.numelemconn_var][:]
                mesh.close()
                mesh_edge_first_node = []
                mesh_edge_second_node = []
                for i in range(elem_conn.shape[0]):
                    loop = 0
                    while(loop+1 < numelem_conn[i]):
                        mesh_edge_first_node.append(elem_conn[i,loop])
                        mesh_edge_second_node.append(elem_conn[i,loop+1])
                        loop += 1
                        if(loop+1 == numelem_conn[i]):
                            mesh_edge_first_node.append(elem_conn[i,numelem_conn[i]-1])
                            mesh_edge_second_node.append(elem_conn[i,0])
                edge_nodes = np.empty((len(mesh_edge_first_node),2),dtype=int)
                edge_nodes[:,0] = mesh_edge_first_node
                edge_nodes[:,1] = mesh_edge_second_node
                edge_nodes = list(edge_nodes)
                seen = set()
                count = 1
                for item in edge_nodes:
                    t = tuple(item)
                    if t not in seen:
                        seen.add(t)
                        count += 1
                edge_count = count
                return edge_count
            else:
                raise NotImplementedError("get_grid_edge_count")

    #------------------------------------------------------------ 
    def get_grid_edge_nodes(self, grid_id: int, edge_nodes: NDArray[np.int_]) -> NDArray[np.int_]:
        for grid in self._grids:
            if grid_id != 1:
                mesh = nc.Dataset(self._job_meta.geogrid)
                elem_conn = mesh.variables[self._job_meta.elemconn_var][:]
                numelem_conn = mesh.variables[self._job_meta.numelemconn_var][:]
                mesh.close()
                mesh_edge_first_node = []
                mesh_edge_second_node = []
                for i in range(elem_conn.shape[0]):
                    loop = 0
                    while(loop+1 < numelem_conn[i]):
                        mesh_edge_first_node.append(elem_conn[i,loop])
                        mesh_edge_second_node.append(elem_conn[i,loop+1])
                        loop += 1
                        if(loop+1 == numelem_conn[i]):
                            mesh_edge_first_node.append(elem_conn[i,numelem_conn[i]-1])
                            mesh_edge_second_node.append(elem_conn[i,0])
                edge_nodes_ = np.empty((len(mesh_edge_first_node),2),dtype=int)
                edge_nodes_[:,0] = mesh_edge_first_node
                edge_nodes_[:,1] = mesh_edge_second_node
                edge_nodes_ = list(edge_nodes_)
                seen = set()
                node_list = []
                count = 1
                for item in edge_nodes_:
                    t = tuple(item)
                    if t not in seen:
                        node_list.append(t)
                        seen.add(t)
                        count += 1
                    else:
                        edge_data = list(seen)
                        node_list.append(edge_data[edge_data.index(t)])
                edge_nodes[:] = np.array(node_list).flatten()
                return edge_nodes
            else:
                raise NotImplementedError("get_grid_edge_nodes")

    #------------------------------------------------------------ 
    def get_grid_face_count(self, grid_id: int) -> int:
        for grid in self._grids:
            if grid_id != 1:
                mesh = nc.Dataset(self._job_meta.geogrid)
                face_count = len(mesh.variables[self._job_meta.elemcoords_var][:][:,0])
                mesh.close()
                return face_count
            else:
                raise NotImplementedError("get_grid_face_count")

    
    #------------------------------------------------------------ 
    def get_grid_face_edges(self, grid_id: int, face_edges: NDArray[np.int_]) -> NDArray[np.int_]:
        for grid in self._grids:
            if grid_id != 1:
                mesh = nc.Dataset(self._job_meta.geogrid)
                elem_conn = mesh.variables[self._job_meta.elemconn_var][:]
                numelem_conn = mesh.variables[self._job_meta.numelemconn_var][:]
                mesh.close()
                mesh_edge_first_node = []
                mesh_edge_second_node = []
                for i in range(elem_conn.shape[0]):
                    loop = 0
                    while(loop+1 < numelem_conn[i]):
                        mesh_edge_first_node.append(elem_conn[i,loop])
                        mesh_edge_second_node.append(elem_conn[i,loop+1])
                        loop += 1
                        if(loop+1 == numelem_conn[i]):
                            mesh_edge_first_node.append(elem_conn[i,numelem_conn[i]-1])
                            mesh_edge_second_node.append(elem_conn[i,0])
                edge_nodes = np.empty((len(mesh_edge_first_node),2),dtype=int)
                edge_nodes[:,0] = mesh_edge_first_node
                edge_nodes[:,1] = mesh_edge_second_node
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
                raise NotImplementedError("get_grid_face_edges")

    #------------------------------------------------------------ 
    def get_grid_face_nodes(self, grid_id: int, face_nodes: NDArray[np.int_]) -> NDArray[np.int_]:
        for grid in self._grids:
            if grid_id != 1:
                mesh = nc.Dataset(self._job_meta.geogrid)
                elem_conn = mesh.variables[self._job_meta.elemconn_var][:]
                numelem_conn = mesh.variables[self._job_meta.numelemconn_var][:]
                node_conn_num = 0
                for i in range(elem_conn.shape[0]):
                    node_conn_num += numelem_conn[i]
                face_nodes[:] = np.empty(node_conn_num,dtype=int)
                index = 0
                for i in range(elem_conn.shape[0]):
                    for j in range(numelem_conn[i]):
                        face_nodes[index] = elem_conn[i,j]
                        index +=1
                return face_nodes
            else:
                raise NotImplementedError("get_grid_face_nodes")

    
    #------------------------------------------------------------
    def get_grid_node_count(self, grid_id: int) -> int:
        for grid in self._grids:
            if grid_id != 1:
                mesh = nc.Dataset(self._job_meta.geogrid)
                node_count = len(mesh.variables[self._job_meta.nodecoords_var][:][:,0])
                mesh.close()
                return node_count
            else:
                raise NotImplementedError("get_grid_node_count")


    #------------------------------------------------------------ 
    def get_grid_nodes_per_face(self, grid_id: int, nodes_per_face: NDArray[np.int_]) -> NDArray[np.int_]:
        for grid in self._grids:
            if grid_id != 1:
                mesh = nc.Dataset(self._job_meta.geogrid)
                elem_conn = mesh.variables[self._job_meta.elemconn_var][:]
                numelem_conn = mesh.variables[self._job_meta.numelemconn_var][:]
                nodes_per_face[:] = np.empty(elem_conn.shape[0],dtype=int)
                for i in range(elem_conn.shape[0]):
                    nodes_per_face[i] = numelem_conn[i]
                return nodes_per_face
            else:
                raise NotImplementedError("get_grid_nodes_per_face")
    
    #------------------------------------------------------------ 
    def get_grid_origin(self, grid_id: int, origin: NDArray[np.float64]) -> NDArray[np.float64]:
        for grid in self._grids:
            if grid_id == grid.id: 
                origin[:] = grid.origin
                return origin
        raise ValueError(f"get_grid_origin: grid_id {grid_id} unknown")

    #------------------------------------------------------------ 
    def get_grid_rank(self, grid_id: int) -> int:
        for grid in self._grids:
            if grid_id == grid.id: 
                return grid.rank
        raise ValueError(f"get_grid_rank: grid_id {grid_id} unknown")

    #------------------------------------------------------------ 
    def get_grid_shape(self, grid_id: int, shape: NDArray[np.int_]) -> NDArray[np.int_]:
        for grid in self._grids:
            if grid_id == grid.id:
                shape[:] = grid.shape
                return shape
        raise ValueError(f"get_grid_shape: grid_id {grid_id} unknown")

    #------------------------------------------------------------ 
    def get_grid_size(self, grid_id: int) -> int:
        for grid in self._grids:
            if grid_id == grid.id: 
                return grid.size
        raise ValueError(f"get_grid_size: grid_id {grid_id} unknown")

    #------------------------------------------------------------ 
    def get_grid_spacing(self, grid_id: int, spacing: NDArray[np.float64]) -> NDArray[np.float64]:
        for grid in self._grids:
            if grid_id == grid.id: 
                spacing[:] = grid.spacing
                return spacing
        raise ValueError(f"get_grid_spacing: grid_id {grid_id} unknown")  

    #------------------------------------------------------------ 
    def get_grid_type(self, grid_id: int) -> str:
        for grid in self._grids:
            if grid_id == grid.id: 
                return grid.type
        raise ValueError(f"get_grid_type: grid_id {grid_id} unknown")

    #------------------------------------------------------------ 
    def get_grid_x(self, grid_id: int, x: NDArray[np.float64]) -> NDArray[np.float64]:
        for grid in self._grids:
            if grid_id == grid.id:
                x[:] = grid.grid_x
                return x
        raise ValueError(f"get_grid_x: grid_id {grid_id} unknown")

    #------------------------------------------------------------ 
    def get_grid_y(self, grid_id: int, y: NDArray[np.float64]) -> NDArray[np.float64]:
        for grid in self._grids:
            if grid_id == grid.id: 
                y[:] = grid.grid_y
                return y
        raise ValueError(f"get_grid_y: grid_id {grid_id} unknown")

    #------------------------------------------------------------ 
    def get_grid_z(self, grid_id: int, z: NDArray[np.float64]) -> NDArray[np.float64]:
        for grid in self._grids:
            if grid_id == grid.id: 
                z[:] = grid.grid_z
                return z
        raise ValueError(f"get_grid_z: grid_id {grid_id} unknown")

    #------------------------------------------------------------ 
    #------------------------------------------------------------ 
    #-- Random utility functions
    #------------------------------------------------------------ 
    #------------------------------------------------------------ 

    def _parse_config(self, cfg):
        for key, val in cfg.items():
            # convert all path strings to PosixPath objects
            if any([key.endswith(x) for x in ['_dir', '_path', '_file', '_files']]):
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

            # convert Dates to pandas Datetime indexs
            elif key.endswith('_date'):
                if isinstance(val, list):
                    temp_list = []
                    for elem in val:
                        temp_list.append(pd.to_datetime(elem, format='%d/%m/%Y'))
                    cfg[key] = temp_list
                else:
                    cfg[key] = pd.to_datetime(val, format='%d/%m/%Y')
            # Configure NWMv3.0 input configurations to 
            # what the ConfigClass is expecting

            # Flag for variables that need a list of integers
            elif(key in ['InputForcings','InputMandatory','ForecastInputHorizons','ForecastInputOffsets','IgnoredBorderWidths','RegridOpt','TemperatureDownscaling','ShortwaveDownscaling','PressureDownscaling','PrecipDownscaling','HumidityDownscaling','TemperatureBiasCorrection','PressureBiasCorrection','HumidityBiasCorrection','WindBiasCorrection','SwBiasCorrection','LwBiasCorrection','PrecipBiasCorrection','SuppPcp','RegridOptSuppPcp','SuppPcpTemporalInterpolation','SuppPcpMandatory','SuppPcpInputOffsets','custom_input_fcst_freq']):
                cfg[key] = val
            # Flag for variables that need to be a list of strings
            elif(key in ['InputForcingDirectories','InputForcingTypes','DownscalingParamDirs','SuppPcpForcingTypes','SuppPcpDirectories']):
                cfg[key] = val
            else:
                pass

        # Add more config parsing if necessary
        return cfg
