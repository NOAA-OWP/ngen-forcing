"""Module file to handle both reading in input files.

1.) GRIB2 format
2.) NetCDF format
Also, creating output files.
"""

import datetime
import gzip
import math
import os
import shutil
import subprocess
import sys
from typing import Optional

import ewts
import numpy as np
from netCDF4 import Dataset

from . import err_handler

LOG = ewts.get_logger(ewts.FORCING_ID)


if "WGRIB2" not in os.environ:
    WGRIB2_env = False
    import pywgrib2_s
else:
    WGRIB2_env = True


class OutputObj:
    """Abstract class to hold local "slabs" of final output grids."""

    def __init__(self, ConfigOptions, GeoMetaWrfHydro):
        """Initialize OutputObj class variables."""
        self.output_local = None
        self.output_global = None
        self.outPath = None
        self.outDate = None
        self.idOut = None
        self.out_ndv = -9999

        # TODO should the length of these depend on `if ConfigOptions.include_lqfrac`?
        # Create local "slabs" to hold final output grids. These
        # will be collected during the output routine below.
        if ConfigOptions.grid_type == "unstructured":
            self.output_local = np.empty([9, GeoMetaWrfHydro.ny_local])
            self.output_local_elem = np.empty([9, GeoMetaWrfHydro.ny_local_elem])
        elif ConfigOptions.grid_type == "gridded":
            self.output_local = np.empty(
                [9, GeoMetaWrfHydro.ny_local, GeoMetaWrfHydro.nx_local]
            )
        elif ConfigOptions.grid_type == "hydrofabric":
            self.output_local = np.empty([9, GeoMetaWrfHydro.ny_local])
            self.output_global = np.empty([9, GeoMetaWrfHydro.ny_global])
        # self.output_local[:,:,:] = self.out_ndv

    def init_forcing_file(self, ConfigOptions, geoMetaWrfHydro, MpiConfig):
        """Initialize the forcing file output for the WRF-Hydro model.

        Initializes the forcing file output for the WRF-Hydro model. This function assumes that all necessary
        preprocessing steps, such as regridding, interpolation, downscaling, and bias correction, have already
        been completed on the input forcing data. This function collects the data from the local "slabs" of each
        processor and combines them into the final output grid.

        Since this function runs in parallel, the work is done on local data slabs for each processor to ensure
        efficiency. Once the data is processed, it is aggregated into a final grid and written into the output files.
        Additionally, detailed geospatial metadata from the input geogrid file is translated into the output file.

        :param ConfigOptions: Configuration options for the output and model settings.
        :param geoMetaWrfHydro: Geospatial metadata related to the WRF-Hydro grid.
        :param MpiConfig: MPI configuration for parallel processing.
        :return: None
        """
        # Dictionary of output variable attributes: includes variables like wind components, temperature, etc.
        output_variable_attribute_dict = {
            "U2D": [
                0,
                "m s-1",
                "x_wind",
                "10-m U-component of wind",
                "time: point",
                0.001,
                0.0,
                3,
            ],
            "V2D": [
                1,
                "m s-1",
                "y_wind",
                "10-m V-component of wind",
                "time: point",
                0.001,
                0.0,
                3,
            ],
            "LWDOWN": [
                2,
                "W m-2",
                "surface_downward_longwave_flux",
                "Surface downward long-wave radiation flux",
                "time: point",
                0.001,
                0.0,
                3,
            ],
            "RAINRATE": [
                3,
                "mm s^-1",
                "precipitation_flux",
                "Surface Precipitation Rate",
                "time: mean",
                1.0,
                0.0,
                0,
            ],
            "T2D": [
                4,
                "K",
                "air_temperature",
                "2-m Air Temperature",
                "time: point",
                0.01,
                100.0,
                2,
            ],
            "Q2D": [
                5,
                "kg kg-1",
                "surface_specific_humidity",
                "2-m Specific Humidity",
                "time: point",
                0.000001,
                0.0,
                6,
            ],
            "PSFC": [
                6,
                "Pa",
                "air_pressure",
                "Surface Pressure",
                "time: point",
                0.1,
                0.0,
                1,
            ],
            "SWDOWN": [
                7,
                "W m-2",
                "surface_downward_shortwave_flux",
                "Surface downward short-wave radiation flux",
                "time: point",
                0.001,
                0.0,
                3,
            ],
        }

        # Add Liquid Water Fraction variable if it is included in the configuration
        if ConfigOptions.include_lqfrac:
            output_variable_attribute_dict["LQFRAC"] = [
                8,
                "%",
                "liquid_water_fraction",
                "Fraction of precipitation that is liquid vs. frozen",
                "time: point",
                0.01,
                0.0,
                3,
            ]

        # Compose the ESMF remapped string attribute based on the regridding option.
        # We will default to the regridding method chosen for the first input forcing selected.
        if ConfigOptions.regrid_opt[0] == 1:
            regrid_att = "remapped via ESMF regrid_with_weights: Bilinear"
        elif ConfigOptions.regrid_opt[0] == 2:
            regrid_att = "remapped via ESMF regrid_with_weights: Nearest Neighbor"
        elif ConfigOptions.regrid_opt[0] == 3:
            regrid_att = "remapped via ESMF regrid_with_weights: Conservative Bilinear"
        else:
            regrid_att = None

        # Ensure all processors are synchronized before starting the output process
        if MpiConfig.rank == 0:
            while True:
                # Only output data on the master processor
                try:
                    self.idOut = Dataset(
                        self.outPath, "w"
                    )  # Open the output file for writing
                except Exception as e:
                    ConfigOptions.errMsg = (
                        f"Unable to create output file: {self.outPath} - {e}"
                    )
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                # Create dimensions (time and grid-related dimensions)
                try:
                    self.idOut.createDimension(
                        "time", None
                    )  # Time dimension is variable-length
                except Exception as e:
                    ConfigOptions.errMsg = (
                        f"Unable to create time dimension in: {self.outPath} - {e}"
                    )
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                # Handle grid-specific dimensions based on grid type
                if ConfigOptions.grid_type == "gridded":
                    try:
                        self.idOut.createDimension(
                            "y", geoMetaWrfHydro.ny_global
                        )  # Latitude dimension
                    except Exception as e:
                        ConfigOptions.errMsg = (
                            f"Unable to create y dimension in: {self.outPath} - {e}"
                        )
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        self.idOut.createDimension(
                            "x", geoMetaWrfHydro.nx_global
                        )  # Longitude dimension
                    except Exception as e:
                        ConfigOptions.errMsg = (
                            f"Unable to create x dimension in: {self.outPath} - {e}"
                        )
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                elif ConfigOptions.grid_type == "hydrofabric":
                    try:
                        self.idOut.createDimension(
                            "catchment-id", len(geoMetaWrfHydro.element_ids_global)
                        )  # Catchment ID dimension
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to create catchment id dimension in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                elif ConfigOptions.grid_type == "unstructured":
                    try:
                        self.idOut.createDimension(
                            "element-id", geoMetaWrfHydro.ny_global_elem
                        )  # Element ID dimension for unstructured grid
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to create element id dimension in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        self.idOut.createDimension(
                            "nodeCount", geoMetaWrfHydro.ny_global
                        )  # Node ID dimension for unstructured grid
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to create nodeCount dimension in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        self.idOut.createDimension(
                            "coordDim", 2
                        )  # Node ID dimension for unstructured grid
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to create coordDim dimension in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                # Set global attributes for the output file (model initialization time, version, etc.)
                try:
                    self.idOut.model_output_valid_time = (
                        ConfigOptions.b_date_proc.strftime("%Y-%m-%d_%H:%M:00")
                    )
                except Exception as e:
                    ConfigOptions.errMsg = f"Unable to set the model_output_valid_time attribute in: {self.outPath} - {e}"
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break
                try:
                    if ConfigOptions.ana_flag:
                        model_init = ConfigOptions.b_date_proc - datetime.timedelta(
                            minutes=ConfigOptions.output_freq
                        )
                    else:
                        model_init = ConfigOptions.b_date_proc
                    self.idOut.model_initialization_time = model_init.strftime(
                        "%Y-%m-%d_%H:%M:00"
                    )
                except Exception as e:
                    ConfigOptions.errMsg = f"Unable to set the model_initialization_time global attribute in: {self.outPath} - {e}"
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                # Set NWM version if provided
                if ConfigOptions.nwmVersion is not None:
                    try:
                        self.idOut.NWM_version_number = "v" + str(
                            ConfigOptions.nwmVersion
                        )
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to set the NWM_version_number global attribute in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                # Set model configuration if provided
                if ConfigOptions.nwmConfig is not None:
                    try:
                        self.idOut.model_configuration = ConfigOptions.nwmConfig
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to set the model_configuration global attribute in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                # Set the output type
                try:
                    self.idOut.model_output_type = "forcing"
                except Exception as e:
                    ConfigOptions.errMsg = f"Unable to put model_output_type global attribute in: {self.outPath} - {e}"
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                # Set the total number of valid times for the model output
                try:
                    self.idOut.model_total_valid_times = float(
                        ConfigOptions.actual_output_steps
                    )
                except Exception as e:
                    ConfigOptions.errMsg = f"Unable to create total_valid_times global attribute in: {self.outPath} - {e}"
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                # Create a time variable
                try:
                    self.idOut.createVariable("Time", "double", "time")
                except Exception as e:
                    ConfigOptions.errMsg = (
                        f"Unable to create time variable in: {self.outPath} - {e}"
                    )
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                # Set attributes for the time variable
                try:
                    self.idOut.variables[
                        "Time"
                    ].units = "minutes since 1970-01-01 00:00:00 UTC"
                except Exception as e:
                    ConfigOptions.errMsg = f"Unable to create time units attribute in: {self.outPath} - {e}"
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                try:
                    self.idOut.variables["Time"].standard_name = "time"
                except Exception as e:
                    ConfigOptions.errMsg = f"Unable to create time standard_name attribute in: {self.outPath} - {e}"
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                try:
                    self.idOut.variables["Time"].long_name = "valid output time"
                except Exception as e:
                    ConfigOptions.errMsg = f"Unable to create time long_name attribute in: {self.outPath} - {e}"
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                # Set dimension fields for different grid types
                if ConfigOptions.grid_type == "gridded":
                    dim_x = "x"
                    dim_y = "y"
                elif ConfigOptions.grid_type == "hydrofabric":
                    dim_x = "catchment-id"
                    dim_y = "catchment-id"
                elif ConfigOptions.grid_type == "unstructured":
                    dim_x = "element-id"
                    dim_y = "element-id"
                    dim_node = "nodeCount"
                    dim_coord = "coordDim"
                else:
                    raise ValueError(f"Invalid grid_type: {ConfigOptions.grid_type}")

                # Handle spatial metadata if available
                if ConfigOptions.spatial_meta is not None:
                    # Create coordinate variables (x and y) for spatial metadata
                    try:
                        if ConfigOptions.useCompression == 1:
                            self.idOut.createVariable(
                                "x", "f8", "x", zlib=True, complevel=2
                            )
                        else:
                            self.idOut.createVariable("x", "f8", "x")
                    except Exception as e:
                        ConfigOptions.errMsg = (
                            f"Unable to create x variable in: {self.outPath} - {e}"
                        )
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                    try:
                        self.idOut.variables["x"].setncatts(
                            geoMetaWrfHydro.x_coord_atts
                        )
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to establish x coordinate attributes in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        self.idOut.variables["x"][:] = geoMetaWrfHydro.x_coords
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to place x coordinate values into output variable for output file: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                    try:
                        if ConfigOptions.useCompression == 1:
                            self.idOut.createVariable(
                                "y", "f8", "y", zlib=True, complevel=2
                            )
                        else:
                            self.idOut.createVariable("y", "f8", "y")
                    except Exception as e:
                        ConfigOptions.errMsg = (
                            f"Unable to create y variable in: {self.outPath} - {e}"
                        )
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                    try:
                        self.idOut.variables["y"].setncatts(
                            geoMetaWrfHydro.y_coord_atts
                        )
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to establish y coordinate attributes in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                    try:
                        self.idOut.variables["y"][:] = geoMetaWrfHydro.y_coords
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to place y coordinate values into output variable for output file: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                    try:
                        self.idOut.createVariable("crs", "S1")
                    except Exception as e:
                        ConfigOptions.errMsg = (
                            f"Unable to create crs in: {self.outPath} - {e}"
                        )
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                    try:
                        self.idOut.variables["crs"].setncatts(geoMetaWrfHydro.crs_atts)
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to establish crs attributes in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                else:
                    # Handle geospatial information from ESMF mesh/grid files
                    if (
                        geoMetaWrfHydro.esmf_grid.coord_sys is None
                        or geoMetaWrfHydro.esmf_grid.coord_sys.name == "SPH_DEG"
                    ):
                        transform_name = "GCS_WGS_1984"
                        proj = "+proj=latlong +datum=WGS84"
                        units = "degrees"
                    else:
                        # Cartesian coordinates in ESMF
                        transform_name = "GCS_WGS_1984"
                        proj = "+proj=geocent +datum=WGS84"
                        units = "m"

                    # Temporary open the geogrid file to assign geospatial data to netcdf file
                    idTmp = Dataset(ConfigOptions.geogrid, "r")
                    # Create coordinate variables and populate with attributes read in.
                    try:
                        if ConfigOptions.useCompression == 1:
                            self.idOut.createVariable(
                                "x", "f8", dim_x, zlib=True, complevel=2
                            )
                        else:
                            self.idOut.createVariable("x", "f8", dim_x)
                    except Exception as e:
                        ConfigOptions.errMsg = (
                            f"Unable to create x variable in: {self.outPath} - {e}"
                        )
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        self.idOut.variables["x"].setncattr(
                            "standard_name", "Longitude"
                        )
                        self.idOut.variables["x"].setncattr(
                            "long_name", "x coordinate of projection"
                        )
                        self.idOut.variables["x"].setncattr("units", units)
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to establish x coordinate attributes in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                    if ConfigOptions.grid_type != "gridded":
                        try:
                            self.idOut.variables["x"][:] = idTmp.variables[
                                ConfigOptions.elemcoords_var
                            ][:].data[:, 0]
                        except Exception as e:
                            ConfigOptions.errMsg = f"Unable to place x coordinate values into output variable for output file: {self.outPath} - {e}"
                            err_handler.log_critical(ConfigOptions, MpiConfig)
                            break
                    else:
                        try:
                            self.idOut.variables["x"][:] = idTmp.variables[
                                ConfigOptions.lon_var
                            ][:].data
                        except Exception as e:
                            ConfigOptions.errMsg = f"Unable to place x coordinate values into output variable for output file: {self.outPath} - {e}"
                            err_handler.log_critical(ConfigOptions, MpiConfig)
                            break

                    # Repeat for y coordinate (latitude)
                    try:
                        if ConfigOptions.useCompression == 1:
                            self.idOut.createVariable(
                                "y", "f8", dim_y, zlib=True, complevel=2
                            )
                        else:
                            self.idOut.createVariable("y", "f8", dim_y)
                    except Exception as e:
                        ConfigOptions.errMsg = (
                            f"Unable to create y variable in: {self.outPath} - {e}"
                        )
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                    try:
                        self.idOut.variables["y"].setncattr("standard_name", "Latitude")
                        self.idOut.variables["y"].setncattr(
                            "long_name", "y coordinate of projection"
                        )
                        self.idOut.variables["y"].setncattr("units", units)
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to establish y coordinate attributes in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                    if ConfigOptions.grid_type != "gridded":
                        try:
                            self.idOut.variables["y"][:] = idTmp.variables[
                                ConfigOptions.elemcoords_var
                            ][:].data[:, 1]
                        except Exception as e:
                            ConfigOptions.errMsg = f"Unable to place y coordinate values into output variable for output file: {self.outPath} - {e}"
                            err_handler.log_critical(ConfigOptions, MpiConfig)
                            break
                    else:
                        try:
                            self.idOut.variables["y"][:] = idTmp.variables[
                                ConfigOptions.lat_var
                            ][:].data
                        except Exception as e:
                            ConfigOptions.errMsg = f"Unable to place y coordinate values into output variable for output file: {self.outPath} - {e}"
                            err_handler.log_critical(ConfigOptions, MpiConfig)
                            break

                    # added node dimension and coordinates for 'unstructured', necessary for schism
                    # the SCHISM BMI uses elements for RAINTATE but nodes for all other variables
                    if ConfigOptions.grid_type == "unstructured":
                        try:
                            if ConfigOptions.useCompression == 1:
                                self.idOut.createVariable(
                                    "nodeCoords",
                                    "f8",
                                    (dim_node, dim_coord),
                                    zlib=True,
                                    complevel=2,
                                )
                            else:
                                self.idOut.createVariable(
                                    "nodeCoords", "f8", (dim_node, dim_coord)
                                )
                        except Exception as e:
                            ConfigOptions.errMsg = f"Unable to create node variable in: {self.outPath} - {e}"
                            err_handler.log_critical(ConfigOptions, MpiConfig)
                            break

                        try:
                            self.idOut.variables["nodeCoords"].setncattr(
                                "standard_name", "Longitude/Latitude"
                            )
                            self.idOut.variables["nodeCoords"].setncattr(
                                "long_name",
                                "Longitude and latitude coordinate of projection",
                            )
                            self.idOut.variables["nodeCoords"].setncattr("units", units)
                        except Exception as e:
                            ConfigOptions.errMsg = f"Unable to establish node coordinate attributes in: {self.outPath} - {e}"
                            err_handler.log_critical(ConfigOptions, MpiConfig)
                            break

                        try:
                            self.idOut.variables["nodeCoords"][:, :] = idTmp.variables[
                                ConfigOptions.nodecoords_var
                            ][:, :]
                        except Exception as e:
                            ConfigOptions.errMsg = f"Unable to place node coordinate values into output variable for output file: {self.outPath} - {e}"
                            err_handler.log_critical(ConfigOptions, MpiConfig)
                            break

                    if ConfigOptions.grid_type == "hydrofabric":
                        try:
                            self.idOut.createVariable("ids", "str", dim_y)
                        except Exception as e:
                            ConfigOptions.errMsg = f"Unable to create catchment id variable in: {self.outPath} - {e}"
                            err_handler.log_critical(ConfigOptions, MpiConfig)
                            break
                        try:
                            self.idOut.variables["ids"].setncattr(
                                "standard_name", "catchment_ids"
                            )
                            self.idOut.variables["ids"].setncattr(
                                "long_name", "Catchment ID for NextGen hydrofabric"
                            )
                        except Exception as e:
                            ConfigOptions.errMsg = f"Unable to establish catchment id attributes in: {self.outPath} - {e}"
                            err_handler.log_critical(ConfigOptions, MpiConfig)
                            break
                        try:
                            self.idOut.variables["ids"][:] = np.array(
                                [
                                    "cat-" + str(x)
                                    for x in np.array(
                                        geoMetaWrfHydro.element_ids_global, dtype=int
                                    )
                                ]
                            )
                        except Exception as e:
                            ConfigOptions.errMsg = f"Unable to place catchment id string values into output variable for output file: {self.outPath} - {e}"
                            err_handler.log_critical(ConfigOptions, MpiConfig)
                            break

                    try:
                        self.idOut.createVariable("crs", "S1")
                    except Exception as e:
                        ConfigOptions.errMsg = (
                            f"Unable to create crs in: {self.outPath} - {e}"
                        )
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        self.idOut.variables["crs"].setncattr(
                            "transform_name", transform_name
                        )
                        self.idOut.variables["crs"].setncattr(
                            "grid_mapping_name", transform_name
                        )
                        self.idOut.variables["crs"].setncattr("esri_pe_string", proj)
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to establish crs attributes in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break

                    # Close the geogrid netcdf file after assigning coordinates
                    idTmp.close()

                # Continue with variable creation and attribute assignment...
                # Loop through and create each variable, along with expected attributes.
                for varTmp in output_variable_attribute_dict:
                    try:
                        if ConfigOptions.useCompression:
                            zlib = True
                            complevel = 2
                            least_significant_digit = (
                                None
                                if varTmp == "RAINRATE"
                                else output_variable_attribute_dict[varTmp][7]
                            )  # use all digits in RAINRATE
                        else:
                            zlib = False
                            complevel = 0
                            least_significant_digit = None

                        if (
                            ConfigOptions.useFloats or varTmp == "RAINRATE"
                        ):  # RAINRATE always a float
                            fill_value = ConfigOptions.globalNdv
                            dtype = "f4"
                        else:
                            fill_value = int(ConfigOptions.globalNdv)
                            # fill_value = int((ConfigOptions.globalNdv - output_variable_attribute_dict[varTmp][6]) /
                            #                 output_variable_attribute_dict[varTmp][5])
                            dtype = "i4"

                        if ConfigOptions.grid_type == "gridded":
                            self.idOut.createVariable(
                                varTmp,
                                dtype,
                                ("time", dim_y, dim_x),
                                fill_value=fill_value,
                                zlib=zlib,
                                complevel=complevel,
                                least_significant_digit=least_significant_digit,
                            )
                        elif ConfigOptions.grid_type == "unstructured":
                            if varTmp == "RAINRATE":
                                # use elements for RAINATE only
                                self.idOut.createVariable(
                                    varTmp,
                                    dtype,
                                    ("time", dim_y),
                                    fill_value=fill_value,
                                    zlib=zlib,
                                    complevel=complevel,
                                    least_significant_digit=least_significant_digit,
                                )
                            else:
                                # use nodes for other variables
                                self.idOut.createVariable(
                                    varTmp,
                                    dtype,
                                    ("time", dim_node),
                                    fill_value=fill_value,
                                    zlib=zlib,
                                    complevel=complevel,
                                    least_significant_digit=least_significant_digit,
                                )
                        else:
                            self.idOut.createVariable(
                                varTmp,
                                dtype,
                                ("time", dim_y),
                                fill_value=fill_value,
                                zlib=zlib,
                                complevel=complevel,
                                least_significant_digit=least_significant_digit,
                            )

                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to create {varTmp} variable in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        self.idOut.variables[
                            varTmp
                        ].cell_methods = output_variable_attribute_dict[varTmp][4]
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to create cell_methods attribute for: {varTmp} in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        self.idOut.variables[varTmp].remap = regrid_att
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to create remap attribute for: {varTmp} in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    # Place geospatial metadata attributes in if we have them.
                    if ConfigOptions.spatial_meta is not None:
                        try:
                            self.idOut.variables[varTmp].grid_mapping = "crs"
                        except Exception as e:
                            ConfigOptions.errMsg = f"Unable to create grid_mapping attribute for: {varTmp} in: {self.outPath} - {e}"
                            err_handler.log_critical(ConfigOptions, MpiConfig)
                            break
                        if "esri_pe_string" in geoMetaWrfHydro.crs_atts.keys():
                            try:
                                self.idOut.variables[
                                    varTmp
                                ].esri_pe_string = geoMetaWrfHydro.crs_atts[
                                    "esri_pe_string"
                                ]
                            except Exception as e:
                                ConfigOptions.errMsg = f"Unable to create esri_pe_string attribute for: {varTmp} in: {self.outPath} - {e}"
                                err_handler.log_critical(ConfigOptions, MpiConfig)
                                break
                        if "proj4" in geoMetaWrfHydro.spatial_global_atts.keys():
                            try:
                                self.idOut.variables[
                                    varTmp
                                ].proj4 = geoMetaWrfHydro.spatial_global_atts["proj4"]
                            except Exception as e:
                                ConfigOptions.errMsg = f"Unable to create proj4 attribute for: {varTmp} in: {self.outPath} - {e}"
                                err_handler.log_critical(ConfigOptions, MpiConfig)
                                break

                    try:
                        self.idOut.variables[
                            varTmp
                        ].units = output_variable_attribute_dict[varTmp][1]
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to create units attribute for: {varTmp} in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        self.idOut.variables[
                            varTmp
                        ].standard_name = output_variable_attribute_dict[varTmp][2]
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to create standard_name attribute for: {varTmp} in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    try:
                        self.idOut.variables[
                            varTmp
                        ].long_name = output_variable_attribute_dict[varTmp][3]
                    except Exception as e:
                        ConfigOptions.errMsg = f"Unable to create long_name attribute for: {varTmp} in: {self.outPath} - {e}"
                        err_handler.log_critical(ConfigOptions, MpiConfig)
                        break
                    # If we are using scale_factor / add_offset, create here.
                    if not ConfigOptions.useFloats:
                        if varTmp != "RAINRATE":
                            try:
                                self.idOut.variables[
                                    varTmp
                                ].scale_factor = output_variable_attribute_dict[varTmp][
                                    5
                                ]
                            except (ValueError, IOError) as e:
                                ConfigOptions.errMsg = f"Unable to create scale_factor attribute for: {varTmp} in: {self.outPath} - {e}"
                                err_handler.log_critical(ConfigOptions, MpiConfig)
                                break
                            try:
                                self.idOut.variables[
                                    varTmp
                                ].add_offset = output_variable_attribute_dict[varTmp][6]
                            except (ValueError, IOError) as e:
                                ConfigOptions.errMsg = f"Unable to create add_offset attribute for: {varTmp} in: {self.outPath} - {e}"
                                err_handler.log_critical(ConfigOptions, MpiConfig)
                                break

                # Close the NetCDF file
                try:
                    self.idOut.close()
                except (ValueError, IOError) as e:
                    ConfigOptions.errMsg = (
                        f"Unable to close output file: {self.outPath} - {e}"
                    )
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    break

                break

        err_handler.check_program_status(ConfigOptions, MpiConfig)

    def gather_global_outputs(self, ConfigOptions, geoMetaWrfHydro, MpiConfig):
        """
        Updates the output NetCDF file with the regridded or processed forcing data from each processor.

        This function will collect data from the different processors, assemble it into a final grid based on the grid type (gridded, hydrofabric, unstructured), and write the result into the output NetCDF file.

        It also handles setting various global and variable attributes in the output file, including the creation of dimensions, variables, and geospatial metadata.

        Originally this method had one purpose: to write global results (gathered from all processors) to an output NetCDF file.
        This method was later updated to support another purpose, the case of ngen's intentional hydrofabric catchment partitioning differing from
        ESMF's arbitrary mesh/grid partitioning. For that case, MPI Allgatherv is used instead of Gatherv, and the results are stored in
        an attribute that stores global outputs, s.t. all processors have access to results from all other processors. This way, the ngen rank
        can acquire hydrofabric forcing results from any of the ngen-forcing ranks.

        The update described above introduced checks for ConfigOptions.forcing_output. Previously, this method would always write to the NetCDF file,
        but with the update, this method only writes conditionally on that configuration option.

        :param ConfigOptions: Configuration options object containing various settings and parameters.
                              It includes globalNdv (NoDataValue), bmi_time_index, regrid_opt, grid_type, etc.
        :param geoMetaWrfHydro: Geospatial metadata for WRF-Hydro, containing information about grid dimensions and coordinate values.
        :param MpiConfig: MPI configuration object for parallel execution. It manages communication and data distribution between processors.
        :return: None
        """
        # Ensure idOut is explicitly declared as Optional[Dataset]
        idOut: Optional[Dataset] = None

        output_variable_attribute_dict = {
            "U2D": [
                0,
                "m s-1",
                "x_wind",
                "10-m U-component of wind",
                "time: point",
                0.001,
                0.0,
                3,
            ],
            "V2D": [
                1,
                "m s-1",
                "y_wind",
                "10-m V-component of wind",
                "time: point",
                0.001,
                0.0,
                3,
            ],
            "LWDOWN": [
                2,
                "W m-2",
                "surface_downward_longwave_flux",
                "Surface downward long-wave radiation flux",
                "time: point",
                0.001,
                0.0,
                3,
            ],
            "RAINRATE": [
                3,
                "mm s^-1",
                "precipitation_flux",
                "Surface Precipitation Rate",
                "time: mean",
                1.0,
                0.0,
                0,
            ],
            "T2D": [
                4,
                "K",
                "air_temperature",
                "2-m Air Temperature",
                "time: point",
                0.01,
                100.0,
                2,
            ],
            "Q2D": [
                5,
                "kg kg-1",
                "surface_specific_humidity",
                "2-m Specific Humidity",
                "time: point",
                0.000001,
                0.0,
                6,
            ],
            "PSFC": [
                6,
                "Pa",
                "air_pressure",
                "Surface Pressure",
                "time: point",
                0.1,
                0.0,
                1,
            ],
            "SWDOWN": [
                7,
                "W m-2",
                "surface_downward_shortwave_flux",
                "Surface downward short-wave radiation flux",
                "time: point",
                0.001,
                0.0,
                3,
            ],
        }

        if ConfigOptions.include_lqfrac:
            output_variable_attribute_dict["LQFRAC"] = [
                8,
                "%",
                "liquid_water_fraction",
                "Fraction of precipitation that is liquid vs. frozen",
                "time: point",
                0.01,
                0.0,
                3,
            ]

        if ConfigOptions.forcing_output == 1 and MpiConfig.rank == 0:
            LOG.debug(
                f"Writing output forcing file for timestamp {self.outDate.strftime('%Y-%m-%d %H:%M')}: {self.outPath}"
            )
            # Only output on the master processor.
            try:
                # Try opening the output file and populating the time variable
                idOut = Dataset(self.outPath, "a")

                # Populate time variables
                dEpoch = datetime.datetime(1970, 1, 1)
                dtValid = ConfigOptions.current_time - dEpoch
                idOut.variables["Time"][int(ConfigOptions.bmi_time_index)] = int(
                    dtValid.days * 24.0 * 60
                ) + int(math.floor(dtValid.seconds / 60.0))

            except Exception as e:
                ConfigOptions.errMsg = (
                    f"Error processing output file: {self.outPath} - {e}"
                )
                err_handler.log_critical(ConfigOptions, MpiConfig)
                idOut = None

        # Now loop through each variable, collect the data (call on each processor), assemble into the final
        # output grid, and place into the output file (if on processor 0).
        for i_var, varTmp in enumerate(output_variable_attribute_dict):
            # Collect data from the various processors, and place into the output file.
            try:
                if ConfigOptions.grid_type == "gridded":
                    dataOutTmp = MpiConfig.merge_slabs_gatherv(
                        self.output_local[
                            output_variable_attribute_dict[varTmp][0], :, :
                        ],
                        ConfigOptions,
                    )
                elif ConfigOptions.grid_type == "hydrofabric":
                    dataOutTmp = MpiConfig.merge_slabs_gatherv(
                        self.output_local[output_variable_attribute_dict[varTmp][0], :],
                        ConfigOptions,
                        allgather=True,
                    )
                    # NOTE this assumes that the var order here matches var order elsewhere.
                    self.output_global[i_var, :] = dataOutTmp.flatten()
                elif ConfigOptions.grid_type == "unstructured":
                    if varTmp == "RAINRATE":
                        dataOutTmp = MpiConfig.merge_slabs_gatherv(
                            self.output_local_elem[
                                output_variable_attribute_dict[varTmp][0], :
                            ],
                            ConfigOptions,
                        )
                    else:
                        dataOutTmp = MpiConfig.merge_slabs_gatherv(
                            self.output_local[
                                output_variable_attribute_dict[varTmp][0], :
                            ],
                            ConfigOptions,
                        )
                else:
                    raise ValueError(f"Invalid grid_type: {ConfigOptions.grid_type}")
            except Exception as e:
                ConfigOptions.errMsg = (
                    f"Unable to gather final grids for: {varTmp} - {e}"
                )
                err_handler.log_critical(ConfigOptions, MpiConfig)
                continue

            if ConfigOptions.forcing_output == 1 and MpiConfig.rank == 0:
                # Only process on the master processor
                if idOut is not None:
                    try:
                        if ConfigOptions.grid_type == "gridded":
                            idOut.variables[varTmp][
                                int(ConfigOptions.bmi_time_index), :, :
                            ] = dataOutTmp
                        else:
                            idOut.variables[varTmp][
                                int(ConfigOptions.bmi_time_index), :
                            ] = dataOutTmp
                    except (ValueError, IOError) as e:
                        ConfigOptions.errMsg = (
                            f"Unable to place final output grid for: {varTmp} - {e}"
                        )
                        err_handler.log_critical(ConfigOptions, MpiConfig)
            # Reset temporary data objects to keep memory usage down.
            dataOutTmp = None

            err_handler.check_program_status(ConfigOptions, MpiConfig)

        if (
            ConfigOptions.forcing_output == 1
            and MpiConfig.rank == 0
            and idOut is not None
        ):
            # Close the NetCDF file
            try:
                idOut.close()
            except (ValueError, IOError) as e:
                ConfigOptions.errMsg = (
                    f"Unable to close output file: {self.outPath} - {e}"
                )
                err_handler.log_critical(ConfigOptions, MpiConfig)

            # Reset memory
            del idOut

        err_handler.check_program_status(ConfigOptions, MpiConfig)


def open_grib2(
    GribFileIn,
    NetCdfFileOut,
    Wgrib2Cmd,
    ConfigOptions,
    MpiConfig,
    inputVar,
    special_case,
):
    """Convert a GRIB2 file to a NetCDF file using the wgrib2 utility.

    This function checks if the necessary GRIB2 input file exists and if a NetCDF output file
    is already present (deletes the old file if necessary). It uses the wgrib2 command to convert
    the GRIB2 file into NetCDF format, handling special cases when necessary. The function also performs
    error checking, including verifying the existence of variables in the NetCDF output file.

    :param GribFileIn: str
        Path to the input GRIB2 file to be converted.
    :param NetCdfFileOut: str
        Path to the output NetCDF file.
    :param Wgrib2Cmd: list
        The command to be passed to wgrib2 to perform the conversion.
    :param ConfigOptions: object
        Configuration object containing various settings and parameters.
    :param MpiConfig: object
        MPI configuration for parallel execution.
    :param inputVar: str
        The variable to check in the NetCDF file after conversion.
    :param special_case: bool
        Flag indicating whether the conversion should handle special cases.

    :return: Dataset or None
        The NetCDF Dataset object if the conversion is successful,
        or None if an error occurs.

    :raises: Exception
        If any errors occur during file conversion or if the expected
        NetCDF file is not found after conversion.
    """
    # Ensure all processors are synced up before outputting.
    # MpiConfig.comm.barrier()

    # Run wgrib2 command to convert GRIB2 file to NetCDF.
    if MpiConfig.rank == 0:
        # Check to see if output file already exists. If so, delete it and
        # override.
        ConfigOptions.statusMsg = "Reading in GRIB2 file: " + GribFileIn
        err_handler.log_msg(ConfigOptions, MpiConfig, True)  # log at debug level
        if os.path.isfile(NetCdfFileOut):
            ConfigOptions.statusMsg = (
                "Overwriting temporary NetCDF file: " + NetCdfFileOut
            )
            err_handler.log_msg(ConfigOptions, MpiConfig, True)  # log at debug level
        try:
            # WCOSS fix for WGRIB2 crashing when called on the same file twice in python
            if not os.environ.get("MFE_SILENT") and not special_case:
                LOG.debug(f"Wgrib2 command: {Wgrib2Cmd}")  # log at debug level

            # set up GRIB2TABLE if needed:
            if not os.environ.get("GRIB2TABLE"):
                g2path = os.path.join(ConfigOptions.scratch_dir, "grib2.tbl")
                with open(g2path, "wt") as g2t:
                    g2t.write(
                        "209:1:0:0:161:1:6:30:MultiSensorQPE01H:"
                        "Multi-sensor estimated precipitation accumulation 1-hour:mm\n"
                        "209:1:0:0:161:1:6:37:MultiSensorQPE01H:"
                        "Multi-sensor estimated precipitation accumulation 1-hour:mm\n"
                    )
                os.environ["GRIB2TABLE"] = g2path
            if WGRIB2_env:
                result = subprocess.run(
                    Wgrib2Cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                if (exitcode := result.returncode) != 0:
                    ConfigOptions.errMsg = f"wgrib2 failed with exit code {exitcode}. Output:\n{result.stdout}"
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                else:
                    LOG.debug("wgrib2 output:")
                    for line in result.stdout.splitlines():  # Log each line separately
                        LOG.debug(line)

            else:
                sys.stdout.flush()  # Native code uses unbuffered output, so flush our buffer first
                if special_case:
                    # National Blended Model Supplementary precipitation
                    if len(Wgrib2Cmd) == 3:
                        pywgrib2_s.wgrib2(
                            [
                                GribFileIn,
                                "-rewind_init",
                                GribFileIn,
                                "-match",
                                Wgrib2Cmd[0],
                                "-match",
                                Wgrib2Cmd[1],
                                "-not",
                                Wgrib2Cmd[2],
                                "-netcdf",
                                NetCdfFileOut,
                            ]
                        )
                    # National Digital Forecast Database
                    if len(Wgrib2Cmd) == 1:
                        # Extract table contents of the forecast hour data matching the forecast cycle starttime
                        # table = pywgrib2_s.inq(GribFileIn,Wgrib2Cmd[0],Matched=True)#pywgrib2_s.wgrib2([GribFileIn,'-rewind_init',GribFileIn,'-match',Wgrib2Cmd[0],'-vt',GribFileIn])
                        # wgrib2_table = []
                        # append the time date stamps from pwgrib2 inquiry
                        # for i in range(table):
                        #    wgrib2_table.append( pywgrib2_s.matched[i])
                        # Only extract the hourly forecast time stamps for NDFD
                        # wgrib2_table = np.array(wgrib2_table[0:48])

                        pywgrib2_s.wgrib2(
                            [
                                GribFileIn,
                                "-rewind_init",
                                GribFileIn,
                                "-match",
                                Wgrib2Cmd[0],
                                "-netcdf",
                                NetCdfFileOut,
                            ]
                        )
                    # Just specify the entire grib2 file to be converted
                    else:
                        pywgrib2_s.wgrib2(
                            [
                                GribFileIn,
                                "-rewind_init",
                                GribFileIn,
                                "-netcdf",
                                NetCdfFileOut,
                            ]
                        )
                else:
                    pywgrib2_s.wgrib2(
                        [
                            GribFileIn,
                            "-rewind_init",
                            GribFileIn,
                            "-match",
                            Wgrib2Cmd,
                            "-netcdf",
                            NetCdfFileOut,
                        ]
                    )
        except Exception as e:
            ConfigOptions.errMsg = (
                f"Unable to convert: {GribFileIn} to {NetCdfFileOut} - {e}"
            )
            err_handler.log_critical(ConfigOptions, MpiConfig)

        # Ensure file exists.
        if not os.path.isfile(NetCdfFileOut):
            ConfigOptions.errMsg = f"Expected NetCDF file: {NetCdfFileOut} not found. It's possible the GRIB2 variable was not found."
            err_handler.log_critical(ConfigOptions, MpiConfig)

        # Open the NetCDF file.
        try:
            idTmp = Dataset(NetCdfFileOut, "r")
        except Exception as e:
            ConfigOptions.errMsg = (
                f"Unable to open input NetCDF file: {NetCdfFileOut} - {e}"
            )
            err_handler.log_critical(ConfigOptions, MpiConfig)
            idTmp = None
            pass

        if idTmp is not None:
            # Check for expected lat/lon variables.
            if "latitude" not in idTmp.variables.keys():
                ConfigOptions.statusMsg = (
                    f"Unable to locate latitude from: {GribFileIn}"
                )
                err_handler.log_warning(ConfigOptions, MpiConfig)
                # idTmp = None
                pass
        if idTmp is not None:
            if "longitude" not in idTmp.variables.keys():
                ConfigOptions.statusMsg = (
                    f"Unable to locate longitude from: {GribFileIn}"
                )
                err_handler.log_warning(ConfigOptions, MpiConfig)

        if idTmp is not None and inputVar is not None:
            # Loop through all the expected variables.
            if inputVar not in idTmp.variables.keys():
                ConfigOptions.errMsg = f"Unable to locate expected variable: {inputVar} in: {NetCdfFileOut}"
                err_handler.log_critical(ConfigOptions, MpiConfig)
                idTmp = None
    else:
        idTmp = None

    # Ensure all processors are synced up before outputting.
    # MpiConfig.comm.barrier()  ## THIS HAPPENS IN check_program_status

    err_handler.check_program_status(ConfigOptions, MpiConfig)

    # Return the NetCDF file handle back to the user.
    return idTmp


def open_netcdf_forcing(
    NetCdfFileIn,
    ConfigOptions,
    MpiConfig,
    open_on_all_procs=False,
    lat_var="latitude",
    lon_var="longitude",
):
    """Open a NetCDF forcing file, optionally on all processors.

    Generic function to convert a NetCDF forcing file given a list of input forcing variables.
    :param NetCdfFileIn: Path to the NetCDF file.
    :param ConfigOptions: Configuration options.
    :param MpiConfig: MPI configuration.
    :param open_on_all_procs: Boolean to specify whether to open the file on all processors.
    :param lat_var: Name of the latitude variable.
    :param lon_var: Name of the longitude variable.
    :return: The opened NetCDF file handle.
    """
    # Ensure all processors are synced up before outputting.
    # MpiConfig.comm.barrier()

    # Open the NetCDF file on the master processor and read in data.
    if MpiConfig.rank == 0 or open_on_all_procs:
        # Ensure file exists.
        if not os.path.isfile(NetCdfFileIn):
            ConfigOptions.errMsg = f"Expected NetCDF file: {NetCdfFileIn} not found."
            err_handler.log_critical(ConfigOptions, MpiConfig)

        # Open the NetCDF file.
        try:
            idTmp = Dataset(NetCdfFileIn, "r")
        except Exception as e:
            ConfigOptions.errMsg = (
                f"Unable to open input NetCDF file: {NetCdfFileIn} - {e}"
            )
            err_handler.log_critical(ConfigOptions, MpiConfig)
            idTmp = None

        if ConfigOptions.nwm_geogrid is None:
            if idTmp is not None:
                # Check for expected lat/lon variables.
                if lat_var not in idTmp.variables.keys():
                    ConfigOptions.errMsg = (
                        f"Unable to locate {lat_var} from: {NetCdfFileIn}"
                    )
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    idTmp = None

            if idTmp is not None:
                if lon_var not in idTmp.variables.keys():
                    ConfigOptions.errMsg = (
                        f"Unable to locate {lon_var} from: {NetCdfFileIn}"
                    )
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    idTmp = None
    else:
        idTmp = None

    err_handler.check_program_status(ConfigOptions, MpiConfig)
    # Return the NetCDF file handle back to the user.
    return idTmp


def unzip_file(GzFileIn: str, FileOut: str, ConfigOptions, MpiConfig):
    """Unzip a .gz file to a new location.

    Generic I/O function to unzip a .gz file to a new location.
    :param GzFileIn: Path to the input gzipped file.
    :param FileOut: Path to the output file after unzipping.
    :param ConfigOptions: Configuration options.
    :param MpiConfig: MPI configuration.
    :return: None
    """
    # Ensure all processors are synced up before outputting.
    # MpiConfig.comm.barrier()

    if MpiConfig.rank == 0:
        # Unzip the file in place.
        try:
            ConfigOptions.statusMsg = f"Unzipping file: {GzFileIn}"
            err_handler.log_msg(ConfigOptions, MpiConfig, True)  # log at debug level
            with gzip.open(GzFileIn, "rb") as fTmpGz:  # fTmpGz is of type BinaryIO
                with open(FileOut, "wb") as fTmp:  # fTmp is also of type BinaryIO
                    # noinspection PyUnresolvedReferences
                    shutil.copyfileobj(fTmpGz, fTmp)  # type: ignore
        except Exception as e:
            ConfigOptions.errMsg = f"Unable to unzip: {GzFileIn} to {FileOut} - {e}"
            err_handler.log_critical(ConfigOptions, MpiConfig)
            return

        if not os.path.isfile(FileOut):
            ConfigOptions.errMsg = f"Unable to locate expected unzipped file: {FileOut}"
            err_handler.log_critical(ConfigOptions, MpiConfig)
            return
    else:
        return


def read_rqi_monthly_climo(
    ConfigOptions, MpiConfig, supplemental_precip, GeoMetaWrfHydro
):
    """Read in monthly RQI grids on the NWM grid. This is an NWM ONLY option.

    Function to read in monthly RQI grids on the NWM grid. This is an NWM ONLY option.
    Please do not activate if not executing on the NWM conus grid.

    The function reads a NetCDF file that contains RQI data for the NWM (National Water Model)
    grid. If the file does not already exist, it will attempt to open and read the necessary
    RQI grids. If the grid size is incorrect, an error will be raised. It then scatters the
    RQI grid data to all processors for parallel processing.

    :param ConfigOptions: Configuration options containing global settings, such as `globalNdv` and `supp_precip_param_dir`.
    :param MpiConfig: MPI configuration used to manage parallel execution.
    :param supplemental_precip: Object containing the supplemental precipitation data.
    :param GeoMetaWrfHydro: Geospatial metadata object containing global grid dimensions (`ny_global`, `nx_global`) and other spatial attributes.
    :return: None
    """
    # Ensure all processors are synchronized before proceeding
    # MpiConfig.comm.barrier()

    # Check if the RQI grids have valid data (i.e., no NDV values)
    indTmp = np.where(supplemental_precip.regridded_rqi2 != ConfigOptions.globalNdv)

    # Path to the RQI parameter file based on the date
    rqiPath = str(
        os.path.join(
            ConfigOptions.supp_precip_param_dir,
            "MRMS_WGT_RQI0.9_m",
            supplemental_precip.pcp_date2.strftime("%m"),
            "_v1.1_geosmth.nc",
        )
    )

    # Initialize idTmp and varTmp to None to avoid referencing before assignment
    idTmp = None
    varTmp = None

    # If no valid RQI data, we need to read the RQI parameter file
    if len(indTmp[0]) == 0:
        # Only read the file if the rank is 0 (master processor)
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = f"Reading in RQI Parameter File: {rqiPath}"
            err_handler.log_msg(ConfigOptions, MpiConfig, True)  # log at debug level

            # Ensure the RQI file exists before proceeding
            if not os.path.isfile(rqiPath):
                ConfigOptions.errMsg = (
                    f"Expected RQI parameter file: {rqiPath} not found."
                )
                err_handler.log_critical(ConfigOptions, MpiConfig)
                pass

            # Attempt to open the RQI parameter file
            try:
                # Open the Parameter file.
                idTmp = Dataset(rqiPath, "r")
                # Extract the RQI grid from the NetCDF file
                varTmp = idTmp.variables["POP_0mabovemeansealevel"][0, :, :]
            except Exception as e:
                ConfigOptions.errMsg = (
                    f"Error processing parameter file: {rqiPath} - {e}"
                )
                err_handler.log_critical(ConfigOptions, MpiConfig)
                idTmp = None
                varTmp = None

            # Check if the RQI grid size matches the expected dimensions
            if varTmp is not None and (
                varTmp.shape[0] != GeoMetaWrfHydro.ny_global
                or varTmp.shape[1] != GeoMetaWrfHydro.nx_global
            ):
                ConfigOptions.errMsg = f"Improper dimension sizes for POP_0mabovemeansealevel in parameter file: {rqiPath}"
                err_handler.log_critical(ConfigOptions, MpiConfig)
                pass
        else:
            # Set variables to None for non-master processors
            idTmp = None
            varTmp = None

        err_handler.check_program_status(ConfigOptions, MpiConfig)

        # Scatter the RQI data to local processors
        if varTmp is not None:
            varSubTmp = MpiConfig.scatter_array(GeoMetaWrfHydro, varTmp, ConfigOptions)
            err_handler.check_program_status(ConfigOptions, MpiConfig)

            # Assign the processed data to the regridded RQI grid
            supplemental_precip.regridded_rqi2[:, :] = varSubTmp

            # Reset temporary variables for memory management
            varSubTmp = None
            varTmp = None

            # Close the RQI NetCDF file
            if MpiConfig.rank == 0 and idTmp is not None:
                try:
                    idTmp.close()
                except Exception as e:
                    ConfigOptions.errMsg = (
                        f"Unable to close parameter file: {rqiPath} - {e}"
                    )
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    pass
            err_handler.check_program_status(ConfigOptions, MpiConfig)

    # Check if the month has changed and if a new RQI monthly grid should be read
    if supplemental_precip.pcp_date2.month != supplemental_precip.pcp_date1.month:
        # We need to read in a new RQI monthly grid.
        if MpiConfig.rank == 0:
            ConfigOptions.statusMsg = f"Reading in RQI Parameter File: {rqiPath}"
            err_handler.log_msg(ConfigOptions, MpiConfig, True)  # log at debug level

            # Ensure the RQI file exists
            if not os.path.isfile(rqiPath):
                ConfigOptions.errMsg = (
                    f"Expected RQI parameter file: {rqiPath} not found."
                )
                err_handler.log_critical(ConfigOptions, MpiConfig)
                pass

            # Attempt to open the RQI parameter file and extract the grid
            try:
                idTmp = Dataset(rqiPath, "r")
                varTmp = idTmp.variables["POP_0mabovemeansealevel"][0, :, :]

                # Check if the grid dimensions are valid
                if (
                    varTmp.shape[0] != GeoMetaWrfHydro.ny_global
                    or varTmp.shape[1] != GeoMetaWrfHydro.nx_global
                ):
                    ConfigOptions.errMsg = f"Improper dimension sizes for POP_0mabovemeansealevel in parameter file: {rqiPath}"
                    err_handler.log_critical(ConfigOptions, MpiConfig)

            except Exception as e:
                ConfigOptions.errMsg = (
                    f"Unable to process parameter file: {rqiPath} - {e}"
                )

        else:
            idTmp = None
            varTmp = None

        err_handler.check_program_status(ConfigOptions, MpiConfig)

        # Scatter the RQI data again after the month switch
        if varTmp is not None:
            varSubTmp = MpiConfig.scatter_array(GeoMetaWrfHydro, varTmp, ConfigOptions)
            err_handler.check_program_status(ConfigOptions, MpiConfig)

            supplemental_precip.regridded_rqi2[:, :] = varSubTmp

            # Close the RQI NetCDF file
            if MpiConfig.rank == 0 and idTmp is not None:
                try:
                    idTmp.close()
                except Exception as e:
                    ConfigOptions.errMsg = (
                        f"Unable to close parameter file: {rqiPath} - {e}"
                    )
                    err_handler.log_critical(ConfigOptions, MpiConfig)
                    pass
            err_handler.check_program_status(ConfigOptions, MpiConfig)
