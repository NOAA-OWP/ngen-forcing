"""Module will guide the forcing engine in defining parameters in all input forcing products.

These parameters include things such as file types, grid definitions (including
initializing ESMF grids and regrid objects), etc
"""

import logging

import numpy as np

from .config import (
    ConfigOptions,
)
from .geoMod import (
    GeoMetaWrfHydro,
)
from .parallel import MpiConfig
from nextgen_forcings_ewts import MODULE_NAME

from . import regrid, time_handling, timeInterpMod

LOG = logging.getLogger(MODULE_NAME)


class InputForcings:
    """Abstract class defining parameters of a single input forcing product.

    This is an abstract class that will define all the parameters
    of a single input forcing product.
    """

    def __init__(self):
        """Initialize all attributes and objects to None."""
        self.inDir = None
        self.enforce = None
        self.paramDir = None
        self.userFcstHorizon = None
        self.userCycleOffset = None
        self.file_type = None
        self.nx_global = None
        self.ny_global = None
        self.nx_local = None
        self.ny_local = None
        self.nx_local_corner = None
        self.ny_local_corner = None
        self.x_lower_bound = None
        self.x_upper_bound = None
        self.y_lower_bound = None
        self.y_upper_bound = None
        self.x_lower_bound_corner = None
        self.x_upper_bound_corner = None
        self.y_lower_bound_corner = None
        self.y_upper_bound_corner = None
        self.outFreq = None
        self.regridOpt = None
        self.timeInterpOpt = None
        self.t2dDownscaleOpt = None
        self.lapseGrid = None
        self.rqiClimoGrid = None
        self.swDowscaleOpt = None
        self.precipDownscaleOpt = None
        self.nwmPRISM_numGrid = None
        self.nwmPRISM_denGrid = None
        self.q2dDownscaleOpt = None
        self.psfcDownscaleOpt = None
        self.t2dBiasCorrectOpt = None
        self.swBiasCorrectOpt = None
        self.precipBiasCorrectOpt = None
        self.q2dBiasCorrectOpt = None
        self.windBiasCorrectOpt = None
        self.psfcBiasCorrectOpt = None
        self.lwBiasCorrectOpt = None
        self.esmf_lats = None
        self.esmf_lons = None
        self.esmf_grid_in = None
        self.esmf_grid_in_elem = None
        self.regridComplete = False
        self.regridObj = None
        self.regridObj_elem = None
        self.esmf_field_in = None
        self.esmf_field_in_elem = None
        self.esmf_field_out = None
        self.esmf_field_out_elem = None
        # --------------------------------
        # Only used for CFSv2 bias correction
        # as bias correction needs to take
        # place prior to regridding.
        self.coarse_input_forcings1 = None
        self.coarse_input_forcings2 = None
        # --------------------------------
        self.regridded_forcings1 = None
        self.regridded_forcings2 = None
        self.globalPcpRate1 = None
        self.globalPcpRate2 = None
        self.regridded_mask = None
        self.regridded_mask_AORC = None
        self.final_forcings = None
        self.regridded_forcings1_elem = None
        self.regridded_forcings2_elem = None
        self.globalPcpRate1_elem = None
        self.globalPcpRate2_elem = None
        self.regridded_mask_elem = None
        self.regridded_mask_elem_AORC = None
        self.final_forcings_elem = None
        self.ndv = None
        self.file_in1 = None
        self.file_in2 = None
        self.fcst_hour1 = None
        self.fcst_hour2 = None
        self.fcst_date1 = None
        self.fcst_date2 = None
        self.height = None
        self.height_elem = None
        self.tmpFile = None
        self.tmpFile2 = None
        self.tmpFileHeight = None
        self.psfcTmp = None
        self.t2dTmp = None
        self.psfcTmp_elem = None
        self.t2dTmp_elem = None
        self.rstFlag = 0
        self.regridded_precip1 = None
        self.regridded_precip2 = None
        self.regridded_precip1_elem = None
        self.regridded_precip2_elem = None
        self.border = None
        self.skip = False

        # Private attrs that have associated @property setter/getter
        self._keyValue = None
        self._file_ext = None
        self._cycle_freq = None
        self._grib_vars = None

    @property
    def product_name(self):
        """Map the forcing key value to the product name."""
        return {
            1: "NLDAS2_GRIB1",
            2: "NARR_GRIB1",
            3: "GFS_Production_GRIB2",
            4: "NAM_Conus_Nest_GRIB2",
            5: "HRRR_Conus_GRIB2",
            6: "RAP_Conus_GRIB2",
            7: "CFSv2_6Hr_Global_GRIB2",
            8: "WRF_ARW_Hawaii_GRIB2",
            9: "GFS_Production_025d_GRIB2",
            10: "Custom_NetCDF_Hourly",
            11: "Custom_NetCDF_Hourly",
            12: "AORC",
            13: "NAM_Nest_3km_Hawaii",
            14: "NAM_Nest_3km_PuertoRico",
            15: "NAM_Nest_3km_Alaska",
            16: "NAM_Nest_3km_Hawaii_Radiation-Only",
            17: "NAM_Nest_3km_PuertoRico_Radiation-Only",
            18: "WRF_ARW_PuertoRico_GRIB2",
            19: "HRRR_Alaska_GRIB2",
            20: "Alaska_AnA",
            21: "AORC_Alaska",
            22: "Alaska_ExtAnA",
            23: "ERA5",
            24: "NBM",
            25: "NDFD",
            26: "HRRR_15min",
            27: "NWM",
        }[self.keyValue]

    @property
    def keyValue(self):
        if self._keyValue is None:
            raise RuntimeError(f"keyValue has not yet been set")
        return self._keyValue

    @keyValue.setter
    def keyValue(self, val):
        if self._keyValue is not None:
            raise RuntimeError(f"keyValue has already been set (to {self._keyValue}).")
        self._keyValue = val

    @property
    def file_ext(self) -> str:
        """Map the forcing file type to the file extension."""
        if self._file_ext is None:
            # First call to getter, initialize
            if self.file_type == "GRIB1":
                ext = ".grb"
            elif self.file_type == "GRIB2":
                ext = ".grib2"
            elif self.file_type == "NETCDF":
                ext = ".nc"
            elif self.file_type == "NETCDF4":
                ext = ".nc4"
            elif self.file_type == "NWM":
                ext = ".LDASIN_DOMAIN1"
            elif self.file_type == "ZARR":
                ext = ".zarr"
            else:
                raise ValueError(f"Unexpected file_type: {self.file_type}")
            self._file_ext = ext

        return self._file_ext

    @file_ext.setter
    def file_ext(self, val):
        if val is None:
            raise TypeError(
                "Cannot set file_ext to None since that value indicates an uninitialized state"
            )
        self._file_ext = val

    @property
    def cycle_freq(self) -> int:
        """Map the forcing key value to the cycle frequency in minutes."""
        if self._cycle_freq is None:
            # First call to getter, initialize
            self._cycle_freq = {
                1: 60,
                2: 180,
                3: 360,
                4: 360,
                5: 60,
                6: 60,
                7: 360,
                8: 1440,
                9: 360,
                10: -9999,
                11: -9999,
                12: -9999,
                13: 360,
                14: 360,
                15: 360,
                16: 360,
                17: 360,
                18: 1440,
                19: 180,
                20: 180,
                21: -9999,
                22: 180,
                23: -9999,
                24: 60,
                25: 1440,
                26: 15,
                27: -9999,
            }[self.keyValue]
        return self._cycle_freq

    @cycle_freq.setter
    def cycle_freq(self, val):
        if val is None:
            raise TypeError(
                "Cannot set cycle_freq to None since that value indicates an uninitialized state"
            )
        self._cycle_freq = val

    @property
    def grib_vars(self) -> list[str] | None:
        """Map the forcing key value to the required GRIB variable names."""
        if self._grib_vars is None:
            # First call to getter, initialize
            self._grib_vars = {
                1: ["TMP", "SPFH", "UGRD", "VGRD", "PRATE", "DSWRF", "DLWRF", "PRES"],
                2: None,
                3: [
                    "TMP",
                    "SPFH",
                    "UGRD",
                    "VGRD",
                    "PRATE",
                    "DSWRF",
                    "DLWRF",
                    "PRES",
                    "CPOFP",
                ],
                4: None,
                5: [
                    "TMP",
                    "SPFH",
                    "UGRD",
                    "VGRD",
                    "APCP",
                    "DSWRF",
                    "DLWRF",
                    "PRES",
                    "CPOFP",
                ],
                6: [
                    "TMP",
                    "SPFH",
                    "UGRD",
                    "VGRD",
                    "APCP",
                    "DSWRF",
                    "DLWRF",
                    "PRES",
                    "LQFRAC",
                ],
                7: ["TMP", "SPFH", "UGRD", "VGRD", "PRATE", "DSWRF", "DLWRF", "PRES"],
                8: ["TMP", "SPFH", "UGRD", "VGRD", "APCP", "PRES"],
                9: ["TMP", "SPFH", "UGRD", "VGRD", "PRATE", "DSWRF", "DLWRF", "PRES"],
                10: None,
                11: None,
                12: ["TMP", "SPFH", "UGRD", "VGRD", "APCP", "DSWRF", "DLWRF", "PRES"],
                13: ["TMP", "SPFH", "UGRD", "VGRD", "PRATE", "DSWRF", "DLWRF", "PRES"],
                14: ["TMP", "SPFH", "UGRD", "VGRD", "PRATE", "DSWRF", "DLWRF", "PRES"],
                15: ["TMP", "SPFH", "UGRD", "VGRD", "PRATE", "DSWRF", "DLWRF", "PRES"],
                16: ["DSWRF", "DLWRF"],
                17: ["DSWRF", "DLWRF"],
                18: ["TMP", "SPFH", "UGRD", "VGRD", "APCP", "PRES"],
                19: ["TMP", "SPFH", "UGRD", "VGRD", "APCP", "DSWRF", "DLWRF", "PRES"],
                20: ["TMP", "SPFH", "UGRD", "VGRD", "APCP", "DSWRF", "DLWRF", "PRES"],
                21: ["TMP", "SPFH", "UGRD", "VGRD", "APCP", "DSWRF", "DLWRF", "PRES"],
                22: ["TMP", "SPFH", "UGRD", "VGRD", "APCP", "DSWRF", "DLWRF", "PRES"],
                23: ["TMP", "SPFH", "UGRD", "VGRD", "APCP", "DSWRF", "DLWRF", "PRES"],
                24: ["TMP", "APCP"],
                25: ["TMP", "WDIR", "WSPD", "APCP"],
                26: ["TMP", "SPFH", "UGRD", "VGRD", "APCP", "DSWRF", "DLWRF", "PRES","CPOFP"],
                27: [
                    "T2D",
                    "Q2D",
                    "U2D",
                    "V2D",
                    "RAINRATE",
                    "SWDOWN",
                    "LWDOWN",
                    "PSFC",
                ],
            }[self.keyValue]
        return self._grib_vars

    @grib_vars.setter
    def grib_vars(self, val):
        if val is None:
            raise TypeError(
                "Cannot set grib_vars to None since that value indicates an uninitialized state"
            )
        self._grib_vars = val

    @property
    def grib_levels(self):
        """Map the forcing key value to the required GRIB variable levels."""
        return {
            1: [
                "2 m above ground",
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
                "surface",
                "surface",
                "surface",
            ],
            2: None,
            3: [
                "2 m above ground",
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
                "surface",
                "surface",
                "surface",
                "surface",
            ],
            4: None,
            5: [
                "2 m above ground",
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
                "surface",
                "surface",
                "surface",
                "surface",
            ],
            6: [
                "2 m above ground",
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
                "surface",
                "surface",
                "surface",
                "surface",
            ],
            7: [
                "2 m above ground",
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
                "surface",
                "surface",
                "surface",
            ],
            8: [
                "80 m above ground",
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
                "surface",
            ],
            9: [
                "2 m above ground",
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
                "surface",
                "surface",
                "surface",
            ],
            10: None,
            11: None,
            12: None,
            13: [
                "2 m above ground",
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
                "surface",
                "surface",
                "surface",
            ],
            14: [
                "2 m above ground",
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
                "surface",
                "surface",
                "surface",
            ],
            15: [
                "2 m above ground",
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
                "surface",
                "surface",
                "surface",
            ],
            16: ["surface", "surface"],
            17: ["surface", "surface"],
            18: [
                "80 m above ground",
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
                "surface",
            ],
            19: [
                "2 m above ground",
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
                "surface",
                "surface",
                "surface",
            ],
            20: [
                "2 m above ground",
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
                "surface",
                "surface",
                "surface",
            ],
            21: None,
            22: [
                "2 m above ground",
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
                "surface",
                "surface",
                "surface",
            ],
            23: None,
            24: ["2 m above ground", "surface"],
            25: [
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
            ],
            26: [
                "2 m above ground",
                "2 m above ground",
                "10 m above ground",
                "10 m above ground",
                "surface",
                "surface",
                "surface",
                "surface",
                "surface",
            ],
            27: None,
        }[self.keyValue]

    @property
    def netcdf_var_names(self):
        """Map the forcing key value to the required NetCDF variable names."""
        return {
            1: [
                "TMP_2maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "APCP_surface",
                "DSWRF_surface",
                "DLWRF_surface",
                "PRES_surface",
            ],
            2: None,
            3: [
                "TMP_2maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "PRATE_surface",
                "DSWRF_surface",
                "DLWRF_surface",
                "PRES_surface",
                "CPOFP_surface",
            ],
            4: None,
            5: [
                "TMP_2maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "APCP_surface",
                "DSWRF_surface",
                "DLWRF_surface",
                "PRES_surface",
                "CPOFP_surface",
            ],
            6: [
                "TMP_2maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "APCP_surface",
                "DSWRF_surface",
                "DLWRF_surface",
                "PRES_surface",
                "LQFRAC_surface",
            ],
            7: [
                "TMP_2maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "PRATE_surface",
                "DSWRF_surface",
                "DLWRF_surface",
                "PRES_surface",
            ],
            8: [
                "TMP_80maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "APCP_surface",
                "PRES_surface",
            ],
            9: [
                "TMP_2maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "PRATE_surface",
                "DSWRF_surface",
                "DLWRF_surface",
                "PRES_surface",
            ],
            10: ["T2D", "Q2D", "U10", "V10", "RAINRATE", "DSWRF", "DLWRF", "PRES"],
            11: ["T2D", "Q2D", "U10", "V10", "RAINRATE", "DSWRF", "DLWRF", "PRES"],
            12: [
                "TMP_2maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "APCP_surface",
                "DSWRF_surface",
                "DLWRF_surface",
                "PRES_surface",
            ],
            13: [
                "TMP_2maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "PRATE_surface",
                "DSWRF_surface",
                "DLWRF_surface",
                "PRES_surface",
            ],
            14: [
                "TMP_2maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "PRATE_surface",
                "DSWRF_surface",
                "DLWRF_surface",
                "PRES_surface",
            ],
            15: [
                "TMP_2maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "PRATE_surface",
                "DSWRF_surface",
                "DLWRF_surface",
                "PRES_surface",
            ],
            16: ["DSWRF_surface", "DLWRF_surface"],
            17: ["DSWRF_surface", "DLWRF_surface"],
            18: [
                "TMP_80maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "APCP_surface",
                "PRES_surface",
            ],
            19: [
                "TMP_2maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "APCP_surface",
                "DSWRF_surface",
                "DLWRF_surface",
                "PRES_surface",
            ],
            20: [
                "TMP_2maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "APCP_surface",
                "DSWRF_surface",
                "DLWRF_surface",
                "PRES_surface",
            ],
            21: [
                "TMP_2maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "APCP_surface",
                "DSWRF_surface",
                "DLWRF_surface",
                "PRES_surface",
            ],
            22: [
                "TMP_2maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "APCP_surface",
                "DSWRF_surface",
                "DLWRF_surface",
                "PRES_surface",
            ],
            23: ["t2m", "d2m", "u10", "v10", "mtpr", "msdwswrf", "msdwlwrf", "sp"],
            24: ["TMP_2maboveground", "APCP_surface"],
            25: [
                "TMP_2maboveground",
                "WDIR_10maboveground",
                "WIND_10maboveground",
                "APCP_surface",
            ],
            26: [
                "TMP_2maboveground",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
                "APCP_surface",
                "DSWRF_surface",
                "DLWRF_surface",
                "PRES_surface",
                "CPOFP_surface",
            ],
            27: ["T2D", "Q2D", "U2D", "V2D", "RAINRATE", "SWDOWN", "LWDOWN", "PSFC"],
        }[self.keyValue]

    @property
    def grib_mes_idx(self):
        """Map the forcing key value to the required GRIB message ids.

        arrays that store the message ids of required forcing variables for each forcing type
        TODO fill these arrays for forcing types other than GFS
        """
        return {
            1: None,
            2: None,
            3: None,
            4: None,
            5: None,
            6: None,
            7: None,
            8: None,
            9: [33, 34, 39, 40, 43, 88, 91, 6],
            10: None,
            11: None,
            12: None,
            13: None,
            14: None,
            15: None,
            16: None,
            17: None,
            18: None,
            19: None,
            20: None,
            21: None,
            22: None,
            23: None,
            24: None,
            25: None,
            26: None,
            27: None,
        }[self.keyValue]

    @property
    def input_map_output(self):
        """Map the forcing key value to the input to output variable mapping."""
        return {
            1: [4, 5, 0, 1, 3, 7, 2, 6],
            2: None,
            3: [4, 5, 0, 1, 3, 7, 2, 6, 8],
            4: None,
            5: [4, 5, 0, 1, 3, 7, 2, 6, 8],
            6: [4, 5, 0, 1, 3, 7, 2, 6, 8],
            7: [4, 5, 0, 1, 3, 7, 2, 6],
            8: [4, 5, 0, 1, 3, 6],
            9: [4, 5, 0, 1, 3, 7, 2, 6],
            10: [4, 5, 0, 1, 3, 7, 2, 6],
            11: [4, 5, 0, 1, 3, 7, 2, 6],
            12: [4, 5, 0, 1, 3, 7, 2, 6],
            13: [4, 5, 0, 1, 3, 7, 2, 6],
            14: [4, 5, 0, 1, 3, 7, 2, 6],
            15: [4, 5, 0, 1, 3, 7, 2, 6],
            16: [7, 2],
            17: [7, 2],
            18: [4, 5, 0, 1, 3, 6],
            19: [4, 5, 0, 1, 3, 7, 2, 6],
            20: [4, 5, 0, 1, 3, 7, 2, 6],
            21: [4, 5, 0, 1, 3, 7, 2, 6],
            22: [4, 5, 0, 1, 3, 7, 2, 6],
            23: [4, 5, 0, 1, 3, 7, 2, 6],
            24: [4, 3],
            25: [4, 0, 1, 3],
            26: [4, 5, 0, 1, 3, 7, 2, 6, 8],
            27: [4, 5, 0, 1, 3, 7, 2, 6],
        }[self.keyValue]

    @property
    def forecast_horizons(self):
        """Map the forcing key value to the forecast horizons list."""
        return {
            1: None,
            2: None,
            3: None,
            4: None,
            5: [
                18,
                18,
                18,
                18,
                18,
                18,
                36,
                18,
                18,
                18,
                18,
                18,
                36,
                18,
                18,
                18,
                18,
                18,
                36,
                18,
                18,
                18,
                18,
                18,
            ],
            6: [
                21,
                21,
                21,
                39,
                21,
                21,
                21,
                21,
                21,
                39,
                21,
                21,
                21,
                21,
                21,
                39,
                21,
                21,
                21,
                21,
                21,
                39,
                21,
                21,
            ],
            7: None,
            8: None,
            9: None,
            10: None,
            11: None,
            12: None,
            13: None,
            14: None,
            15: None,
            16: None,
            17: None,
            18: None,
            19: None,
            20: None,
            21: None,
            22: None,
            23: None,
            24: None,
            25: None,
            26: [
                18,
                18,
                18,
                18,
                18,
                18,
                36,
                18,
                18,
                18,
                18,
                18,
                36,
                18,
                18,
                18,
                18,
                18,
                36,
                18,
                18,
                18,
                18,
                18,
            ],
            27: None,
        }[self.keyValue]

    @property
    def find_neighbor_files_map(self):
        """Map the forcing key value to the neighbor file finding function."""
        return {
            1: time_handling.find_nldas_neighbors,
            3: time_handling.find_gfs_neighbors,
            5: time_handling.find_conus_hrrr_neighbors,
            6: time_handling.find_conus_rap_neighbors,
            7: time_handling.find_cfsv2_neighbors,
            8: time_handling.find_hourly_wrf_arw_neighbors,
            9: time_handling.find_gfs_neighbors,
            10: time_handling.find_custom_hourly_neighbors,
            11: time_handling.find_custom_hourly_neighbors,
            12: time_handling.find_aorc_neighbors,
            13: time_handling.find_nam_nest_neighbors,
            14: time_handling.find_nam_nest_neighbors,
            15: time_handling.find_nam_nest_neighbors,
            16: time_handling.find_nam_nest_neighbors,
            17: time_handling.find_nam_nest_neighbors,
            18: time_handling.find_hourly_wrf_arw_neighbors,
            19: time_handling.find_ak_hrrr_neighbors,
            20: time_handling.find_ak_hrrr_neighbors,
            21: time_handling.find_aorc_neighbors,
            22: time_handling.find_ak_hrrr_neighbors,
            23: time_handling.find_era5_neighbors,
            24: time_handling.find_hourly_nbm_neighbors,
            25: time_handling.find_ndfd_neighbors,
            26: time_handling.find_input_neighbors,
            27: time_handling.find_nwm_neighbors,
        }

    def calc_neighbor_files(
        self, config_options: ConfigOptions, dcurrent, mpi_config: MpiConfig
    ):
        """Calculate the last/next expected input forcing file based on the current time step.

        Function that will calculate the last/next expected
        input forcing file based on the current time step that
        is being processed.
        :param config_options:
        :param dCurrent:
        :return:
        """
        # First calculate the current input cycle date this
        # WRF-Hydro output timestep corresponds to.

        LOG.debug(
            f"keyValue: {self.keyValue}, {self.find_neighbor_files_map[self.keyValue].__name__}"
        )
        self.find_neighbor_files_map[self.keyValue](
            self, config_options, dcurrent, mpi_config
        )

    @property
    def regrid_map(self):
        """Map the forcing key value to the regridding function."""
        return {
            1: regrid.regrid_conus_rap,
            3: regrid.regrid_gfs,
            5: regrid.regrid_conus_hrrr,
            6: regrid.regrid_conus_rap,
            7: regrid.regrid_cfsv2,
            8: regrid.regrid_hourly_wrf_arw,
            9: regrid.regrid_gfs,
            10: regrid.regrid_custom_hourly_netcdf,
            11: regrid.regrid_custom_hourly_netcdf,
            12: regrid.regrid_custom_hourly_netcdf,
            13: regrid.regrid_nam_nest,
            14: regrid.regrid_nam_nest,
            15: regrid.regrid_nam_nest,
            16: regrid.regrid_nam_nest,
            17: regrid.regrid_nam_nest,
            18: regrid.regrid_hourly_wrf_arw,
            19: regrid.regrid_conus_hrrr,
            20: regrid.regrid_conus_hrrr,
            21: regrid.regrid_custom_hourly_netcdf,
            22: regrid.regrid_conus_hrrr,
            23: regrid.regrid_era5,
            24: regrid.regrid_hourly_nbm,
            25: regrid.regrid_ndfd,
            26: regrid.regrid_conus_hrrr,
            27: regrid.regrid_nwm,
        }

    def regrid_inputs(
        self,
        config_options: ConfigOptions,
        wrf_hyro_geo_meta: GeoMetaWrfHydro,
        mpi_config: MpiConfig,
    ):
        """Regrid input forcings to the final output grids for this timestep.

        Polymorphic function that will regrid input forcings to the
        final output grids for this particular timestep. For
        timesteps that require interpolation, two sets of input
        forcing grids will be regridded IF we have come across new
        files and the process flag has been reset.
        :param config_options:
        :return:
        """
        # Establish a mapping dictionary that will point the
        # code to the functions to that will regrid the data.
        self.regrid_map[self.keyValue](
            self, config_options, wrf_hyro_geo_meta, mpi_config
        )

    @property
    def temporal_interpolate_inputs_map(self):
        """Map the temporal interpolation options to the functions."""
        return {
            0: timeInterpMod.no_interpolation,
            1: timeInterpMod.nearest_neighbor,
            2: timeInterpMod.weighted_average,
        }

    def temporal_interpolate_inputs(
        self, config_options: ConfigOptions, mpi_config: MpiConfig
    ):
        """Run temporal interpolation of the input forcing grids that have been regridded.

        Polymorphic function that will run temporal interpolation of
        the input forcing grids that have been regridded. This is
        especially important for forcings that have large output
        frequencies. This is also important for frequent WRF-Hydro
        input timesteps.
        :param config_options:
        :param mpi_config:
        :return:
        """
        self.temporal_interpolate_inputs_map[self.timeInterpOpt](
            self, config_options, mpi_config
        )


def init_dict(
    config_options: ConfigOptions,
    geo_meta_wrf_hydro: GeoMetaWrfHydro,
    mpi_config: MpiConfig,
) -> dict:
    """Initialize the input forcing dictionary.

    Initial function to create an input forcing dictionary, which
    will contain an abstract class for each input forcing product.
    This gets called one time by the parent calling program.
    :param config_options:
    :return: input_dict - A dictionary defining our inputs.
    """
    # Initialize an empty dictionary
    input_dict = {}

    if config_options.precip_only_flag:
        return input_dict

    # Loop through and initialize the empty class for each product.
    custom_count = 0
    for force_tmp in range(0, config_options.number_inputs):
        force_key = config_options.input_forcings[force_tmp]
        input_dict[force_key] = InputForcings()
        input_dict[force_key].keyValue = force_key
        input_dict[force_key].regridOpt = config_options.regrid_opt[force_tmp]
        input_dict[force_key].enforce = config_options.input_force_mandatory[force_tmp]
        input_dict[force_key].timeInterpOpt = config_options.forceTemoralInterp[
            force_tmp
        ]
        input_dict[force_key].q2dDownscaleOpt = config_options.q2dDownscaleOpt[
            force_tmp
        ]
        input_dict[force_key].t2dDownscaleOpt = config_options.t2dDownscaleOpt[
            force_tmp
        ]
        input_dict[force_key].precipDownscaleOpt = config_options.precipDownscaleOpt[
            force_tmp
        ]
        input_dict[force_key].swDowscaleOpt = config_options.swDownscaleOpt[force_tmp]
        input_dict[force_key].psfcDownscaleOpt = config_options.psfcDownscaleOpt[
            force_tmp
        ]
        # Check to make sure the necessary input files for downscaling are present.
        # if input_dict[force_key].t2dDownscaleOpt == 2:
        #    # We are using a pre-calculated lapse rate on the WRF-Hydro grid.
        #    pathCheck = config_options.downscaleParamDir = "/T2M_Lapse_Rate_" + \
        #        input_dict[force_key].product_name + ".nc"
        #    if not os.path.isfile(pathCheck):
        #        config_options.errMsg = "Expected temperature lapse rate grid: " + \
        #            pathCheck + " not found."
        #        raise Exception

        input_dict[force_key].t2dBiasCorrectOpt = config_options.t2BiasCorrectOpt[
            force_tmp
        ]
        input_dict[force_key].q2dBiasCorrectOpt = config_options.q2BiasCorrectOpt[
            force_tmp
        ]
        input_dict[
            force_key
        ].precipBiasCorrectOpt = config_options.precipBiasCorrectOpt[force_tmp]
        input_dict[force_key].swBiasCorrectOpt = config_options.swBiasCorrectOpt[
            force_tmp
        ]
        input_dict[force_key].lwBiasCorrectOpt = config_options.lwBiasCorrectOpt[
            force_tmp
        ]
        input_dict[force_key].windBiasCorrectOpt = config_options.windBiasCorrect[
            force_tmp
        ]
        input_dict[force_key].psfcBiasCorrectOpt = config_options.psfcBiasCorrectOpt[
            force_tmp
        ]

        input_dict[force_key].inDir = config_options.input_force_dirs[force_tmp]
        input_dict[force_key].paramDir = config_options.dScaleParamDirs[force_tmp]
        input_dict[force_key].file_type = config_options.input_force_types[force_tmp]
        input_dict[force_key].userFcstHorizon = config_options.fcst_input_horizons[
            force_tmp
        ]
        input_dict[force_key].userCycleOffset = config_options.fcst_input_offsets[
            force_tmp
        ]

        input_dict[force_key].border = config_options.ignored_border_widths[force_tmp]

        # If we have specified specific humidity downscaling, establish arrays to hold
        # temporary temperature arrays that are un-downscaled.
        if input_dict[force_key].q2dDownscaleOpt > 0:
            if config_options.grid_type == "gridded":
                input_dict[force_key].t2dTmp = np.empty(
                    [geo_meta_wrf_hydro.ny_local, geo_meta_wrf_hydro.nx_local],
                    np.float32,
                )
                input_dict[force_key].psfcTmp = np.empty(
                    [geo_meta_wrf_hydro.ny_local, geo_meta_wrf_hydro.nx_local],
                    np.float32,
                )
            elif config_options.grid_type == "unstructured":
                input_dict[force_key].t2dTmp = np.empty(
                    [geo_meta_wrf_hydro.ny_local], np.float32
                )
                input_dict[force_key].psfcTmp = np.empty(
                    [geo_meta_wrf_hydro.ny_local], np.float32
                )
                input_dict[force_key].t2dTmp_elem = np.empty(
                    [geo_meta_wrf_hydro.ny_local_elem], np.float32
                )
                input_dict[force_key].psfcTmp_elem = np.empty(
                    [geo_meta_wrf_hydro.ny_local_elem], np.float32
                )
            elif config_options.grid_type == "hydrofabric":
                input_dict[force_key].t2dTmp = np.empty(
                    [geo_meta_wrf_hydro.ny_local], np.float32
                )
                input_dict[force_key].psfcTmp = np.empty(
                    [geo_meta_wrf_hydro.ny_local], np.float32
                )
        # Initialize the local final grid of values. This is represntative
        # of the local grid for this forcing, for a specific output timesetp.
        # This grid will be updated from one output timestep to another, and
        # also through downscaling and bias correction.
        force_count = 9 if config_options.include_lqfrac else 8
        if force_count == 8 and 8 in input_dict[force_key].input_map_output:
            # TODO: this assumes that LQFRAC (8) is always the last grib var
            input_dict[force_key].grib_vars = input_dict[force_key].grib_vars[:-1]

        if config_options.grid_type == "gridded":
            input_dict[force_key].final_forcings = np.empty(
                [force_count, geo_meta_wrf_hydro.ny_local, geo_meta_wrf_hydro.nx_local],
                np.float64,
            )
            input_dict[force_key].height = np.empty(
                [geo_meta_wrf_hydro.ny_local, geo_meta_wrf_hydro.nx_local], np.float32
            )
            input_dict[force_key].regridded_mask = np.empty(
                [geo_meta_wrf_hydro.ny_local, geo_meta_wrf_hydro.nx_local], np.float32
            )
            input_dict[force_key].regridded_mask_AORC = np.empty(
                [geo_meta_wrf_hydro.ny_local, geo_meta_wrf_hydro.nx_local], np.float32
            )
        elif config_options.grid_type == "unstructured":
            input_dict[force_key].final_forcings = np.empty(
                [force_count, geo_meta_wrf_hydro.ny_local], np.float64
            )
            input_dict[force_key].height = np.empty(
                [geo_meta_wrf_hydro.ny_local], np.float32
            )
            input_dict[force_key].regridded_mask = np.empty(
                [geo_meta_wrf_hydro.ny_local], np.float32
            )
            input_dict[force_key].regridded_mask_AORC = np.empty(
                [geo_meta_wrf_hydro.ny_local], np.float32
            )
            input_dict[force_key].final_forcings_elem = np.empty(
                [force_count, geo_meta_wrf_hydro.ny_local_elem], np.float64
            )
            input_dict[force_key].height_elem = np.empty(
                [geo_meta_wrf_hydro.ny_local_elem], np.float32
            )
            input_dict[force_key].regridded_mask_elem = np.empty(
                [geo_meta_wrf_hydro.ny_local_elem], np.float32
            )
            input_dict[force_key].regridded_mask_elem_AORC = np.empty(
                [geo_meta_wrf_hydro.ny_local_elem], np.float32
            )
        elif config_options.grid_type == "hydrofabric":
            input_dict[force_key].final_forcings = np.empty(
                [force_count, geo_meta_wrf_hydro.ny_local], np.float64
            )
            input_dict[force_key].height = np.empty(
                [geo_meta_wrf_hydro.ny_local], np.float32
            )
            input_dict[force_key].regridded_mask = np.empty(
                [geo_meta_wrf_hydro.ny_local], np.float32
            )
            input_dict[force_key].regridded_mask_AORC = np.empty(
                [geo_meta_wrf_hydro.ny_local], np.float32
            )
        # Obtain custom input cycle frequencies
        if force_key == 10 or force_key == 11:
            input_dict[force_key].cycle_freq = config_options.customFcstFreq[
                custom_count
            ]
            custom_count = custom_count + 1

    return input_dict
