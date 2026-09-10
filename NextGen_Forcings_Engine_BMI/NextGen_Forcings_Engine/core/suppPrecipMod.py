"""High-level module file that will handle supplemental analysis/observed precipitation grids that will replace precipitation in the final output files."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.consts import (
    SUPPPRECIPMOD,
)

if TYPE_CHECKING:
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.config import (
        ConfigOptions,
    )
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.geoMod import (
        GeoMeta,
    )
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.parallel import (
        MpiConfig,
    )

LOG = logging.getLogger("FORCING")


class SupplementalPrecip:
    """Supplemental precipitation class.

    This class defines all the parameters of a single supplemental precipitation product.

    Three-tier attr initialization:
    1. Attrs set during init (keyValue, geo_meta, etc.) come from constructor params — must NOT be in SUPPPRECIPMOD.
    2. Attrs in SUPPPRECIPMOD[base class name] are then set to None as an "unset" sentinel.
    3. _initialize_config_options then sets list-valued attrs from config_options;
       remaining attrs lazy-initialize via property getters on first access.

    NOTE: Lists are treated specially in config_options. When an attribute value in config_options
    is a list, the idx of this instance is used to extract the corresponding element from that list.
    This allows each SupplementalPrecip instance to reference its own value within a shared list structure.
    """

    def __init__(self, idx: int, config_options: ConfigOptions, geo_meta: GeoMeta):
        """Initializie all attributes and objects to None."""
        self.regridComplete = False
        self.has_cache = False
        self._keyValue = config_options.supp_precip_forcings[idx]
        self.idx = idx
        self.config_options = config_options
        self.geo_meta = geo_meta

        for attr in SUPPPRECIPMOD[__class__.__name__]:
            setattr(self, attr, None)

        self._initialize_config_options()

    @property
    def keyValue(self) -> int:
        """Get the forcing key value."""
        if self._keyValue is None:
            raise RuntimeError("keyValue has not yet been set")
        return self._keyValue

    @keyValue.setter
    def keyValue(self, val: int) -> int:
        """Set the forcing key value."""
        self._keyValue = val

    def _initialize_config_options(self) -> None:
        """Initialize configuration options from the config_options attribute.

        For each attribute in SUPPPRECIPMOD["SupplementalPrecip"], check if the
        same-named attribute exists in config_options as a list and set it on self.
        """
        for attr in SUPPPRECIPMOD[__class__.__name__]:
            if hasattr(self.config_options, attr):
                val = getattr(self.config_options, attr)
                if isinstance(val, list) and len(val) > 0:
                    setattr(self, attr, val[self.idx])

    @property
    def rqiMethod(self) -> int | float:
        """Get the RQI method for this supplemental precipitation product."""
        if self._rqiMethod is None:
            # config_options stores each product's values as a list (one entry per supp precip product).
            # A non-list value means RQI is not configured (default rqiMethod to 0).
            val = self.config_options.rqiMethod
            if isinstance(val, list):
                self._rqiMethod = val[self.idx]
            elif val is None:
                self._rqiMethod = 0
            else:
                raise TypeError(
                    f"Unexpected type for config_options.rqiMethod: {type(val)}"
                )
        return self._rqiMethod

    @rqiMethod.setter
    def rqiMethod(self, val: int | float) -> None:
        """Setter for grib_vars."""
        self._rqiMethod = val

    @property
    def rqiThresh(self) -> int | float:
        """Get the RQI threshold for this supplemental precipitation product."""
        if self._rqiThresh is None:
            # config_options stores each product's values as a list (one entry per supp precip product).
            # A non-list value means RQI is not configured (default rqiMethod to 1.0).
            val = self.config_options.rqiThresh
            if isinstance(val, list):
                self._rqiThresh = val[self.idx]
            elif val is None or isinstance(val, (int, float)):
                # config.py initializes rqiThresh=1.0 as the no-RQI default.
                # When RQI is configured, a scalar gets expanded to a list before reaching here.
                self._rqiThresh = float(val) if val is not None else 1.0
            else:
                raise TypeError(
                    f"Unexpected type for config_options.rqiThresh: {type(val)}"
                )
        return self._rqiThresh

    @rqiThresh.setter
    def rqiThresh(self, val: int | float) -> None:
        """Setter for rqiThresh."""
        self._rqiThresh = val

    @property
    def product_name(self) -> str:
        """Get the product name for this supplemental precipitation product."""
        if self._product_name is None:
            self._product_name = SUPPPRECIPMOD["PRODUCT_NAMES"][self.keyValue]
        return self._product_name

        ## DEFINED IN CONFIG
        # product_types = {
        #     1: "GRIB2",
        #     2: "GRIB2",
        #     3: "GRIB2",
        #     4: "GRIB2",
        #     5: "GRIB2"
        # }
        # self.file_type = product_types[self.keyValue]

    @product_name.setter
    def product_name(self, val: str) -> None:
        """Setter for product_name."""
        self._product_name = val

    @property
    def file_type(self) -> str:
        """Get the file type; aliases supp_precip_file_types set by _initialize_config_options."""
        return self.supp_precip_file_types

    @file_type.setter
    def file_type(self, val: str) -> None:
        """Setter for file_type; writes through to supp_precip_file_types."""
        self.supp_precip_file_types = val

    # TODO: remove these aliases once time_handling.py and regrid.py are refactored to use new attribute names
    @property
    def inDir(self):
        return self.supp_precip_dirs

    @property
    def regridOpt(self):
        return self.regrid_opt_supp_pcp

    @property
    def enforce(self):
        return self.supp_precip_mandatory

    @property
    def timeInterpOpt(self):
        return self.suppTemporalInterp

    @property
    def userCycleOffset(self):
        return self.supp_input_offsets

    @property
    def file_ext(self) -> str:
        """Get the file extension for this supplemental precipitation product."""
        return SUPPPRECIPMOD["FILE_EXT"][self.file_type]

    @property
    def grib_vars(self) -> list[str]:
        """Get the GRIB variable names for this supplemental precipitation product."""
        if self._grib_vars is None:
            self._grib_vars = SUPPPRECIPMOD["GRIB_VARS"][self.keyValue]
        return self._grib_vars

    @grib_vars.setter
    def grib_vars(self, val: list[str]) -> None:
        """Setter for grib_vars."""
        self._grib_vars = val

    @property
    def grib_levels(self) -> list[str]:
        """Get the GRIB levels for this supplemental precipitation product."""
        if self._grib_levels is None:
            self._grib_levels = SUPPPRECIPMOD["GRIB_LEVELS"][self.keyValue]
        return self._grib_levels

    @grib_levels.setter
    def grib_levels(self, val: list[str]) -> None:
        """Setter for grib_levels."""
        self._grib_levels = val

    @property
    def netcdf_var_names(self) -> list[str]:
        """Get the NetCDF variable names for this supplemental precipitation product."""
        if self._netcdf_var_names is None:
            self._netcdf_var_names = SUPPPRECIPMOD["NET_CDF_VARS_NAMES"][self.keyValue]
        return self._netcdf_var_names

    @netcdf_var_names.setter
    def netcdf_var_names(self, val: list[str]) -> None:
        """Setter for netcdf_var_names."""
        self._netcdf_var_names = val

    @property
    def rqi_netcdf_var_names(self) -> list[str] | None:
        """Get the RQI NetCDF variable names for this supplemental precipitation product."""
        if self._rqi_netcdf_var_names is None:
            self._rqi_netcdf_var_names = SUPPPRECIPMOD["RQI_NETCDF_VAR_NAMES"][
                self.keyValue
            ]
        return self._rqi_netcdf_var_names

    @rqi_netcdf_var_names.setter
    def rqi_netcdf_var_names(self, val: list[str] | None) -> None:
        """Setter for rqi_netcdf_var_names."""
        self._rqi_netcdf_var_names = val

    @property
    def output_var_idx(self) -> int:
        """Get the output variable index for this supplemental precipitation product."""
        return SUPPPRECIPMOD["OUTPUT_VAR_IDX"][self.keyValue]

    @property
    def find_neighbor_files(self) -> dict:
        """Get the function to find neighbor supplemental precipitation files for this supplemental precipitation product."""
        return SUPPPRECIPMOD["FIND_NEIGHBOR_FILES_MAP"]

    def calc_neighbor_files(
        self, config_options: ConfigOptions, dcurrent, mpi_config: MpiConfig
    ) -> None:
        """Calculate neighbor supplemental precipitation files.

        Function that will calculate the last/next expected
        supplemental precipitation file based on the current time step that
        is being processed.
        :param ConfigOptions:
        :param dCurrent:
        :return:
        """
        self.find_neighbor_files[self.keyValue](
            self, config_options, dcurrent, mpi_config
        )

    @property
    def regrid_map(self) -> dict:
        """Get the function to regrid input forcings to the supplemental precipitation grids for this supplemental precipitation product."""
        return SUPPPRECIPMOD["REGRID_MAP"]

    def regrid_inputs(
        self, config_options: ConfigOptions, geo_meta: GeoMeta, mpi_config: MpiConfig
    ) -> None:
        """Polymorphic function that will regrid input forcings to the supplemental precipitation grids for this particular timestep.

        Polymorphic function that will regrid input forcings to the
        supplemental precipitation grids for this particular timestep. For
        timesteps that require interpolation, two sets of input
        forcing grids will be regridded IF we have come across new
        files and the process flag has been reset.
        :param ConfigOptions:
        :return:
        """
        # Establish a mapping dictionary that will point the
        # code to the functions to that will regrid the data.
        self.regrid_map[self.keyValue](self, config_options, geo_meta, mpi_config)

    @property
    def temporal_interpolate_inputs_map(self) -> dict:
        """Get the function to temporal interpolate input forcings to the supplemental precipitation grids for this supplemental precipitation product."""
        return SUPPPRECIPMOD["TEMPORAL_INTERPOLATE_INPUTS_MAP"]

    def temporal_interpolate_inputs(
        self, config_options: ConfigOptions, mpi_config: MpiConfig
    ):
        """Polymorphic function that will run temporal interpolation of the supplemental precipitation grids that have been regridded.

        Polymorphic function that will run temporal interpolation of
        the supplemental precipitation grids that have been regridded. This is
        especially important for supplemental precips that have large output
        frequencies. This is also important for frequent WRF-Hydro
        input timesteps.
        :param ConfigOptions:
        :param MpiConfig:
        :return:
        """
        self.temporal_interpolate_inputs_map[self.timeInterpOpt](
            self, config_options, mpi_config
        )


class SupplementalPrecipGridded(SupplementalPrecip):
    """Supplemental precipitation class for gridded products."""

    def __init__(
        self,
        idx: int = None,
        config_options: ConfigOptions = None,
        geo_meta: GeoMeta = None,
    ) -> None:
        """Initialize SupplementalPrecipGridded.  Any subclass-specific attr names are sourced from SUPPPRECIPMOD[classname] in consts.py."""
        super().__init__(idx, config_options, geo_meta)
        for attr in SUPPPRECIPMOD[__class__.__name__]:
            setattr(self, attr, None)

    @property
    def final_supp_precip(self) -> np.ndarray | Any:
        """Get the final supplemental precipitation grid after regridding and temporal interpolation."""
        if self._final_supp_precip is None:
            self._final_supp_precip = np.full(
                [self.geo_meta.ny_local, self.geo_meta.nx_local],
                np.nan,
                dtype=np.float64,
            )
        return self._final_supp_precip

    @final_supp_precip.setter
    def final_supp_precip(self, value: Any) -> Any:
        """Setter for final_supp_precip."""
        self._final_supp_precip = value

    @property
    def regridded_mask(self) -> np.ndarray | Any:
        """Get the regridded mask after regridding input forcings to the supplemental precipitation grids."""
        if self._regridded_mask is None:
            self._regridded_mask = np.full(
                [self.geo_meta.ny_local, self.geo_meta.nx_local], np.nan, np.float32
            )
        return self._regridded_mask

    @regridded_mask.setter
    def regridded_mask(self, value: Any) -> Any:
        """Setter for regridded_mask."""
        self._regridded_mask = value


class SupplementalPrecipHydrofabric(SupplementalPrecip):
    """Supplemental precipitation class for hydrofabric grids."""

    def __init__(
        self,
        idx: int = None,
        config_options: ConfigOptions = None,
        geo_meta: GeoMeta = None,
    ) -> None:
        """Initialize SupplementalPrecipHydrofabric.  Any subclass-specific attr names are sourced from SUPPPRECIPMOD[classname] in consts.py."""
        super().__init__(idx, config_options, geo_meta)
        for attr in SUPPPRECIPMOD[__class__.__name__]:
            setattr(self, attr, None)

    @property
    def final_supp_precip(self) -> np.ndarray | Any:
        """Get the final supplemental precipitation grid after regridding and temporal interpolation."""
        if self._final_supp_precip is None:
            self._final_supp_precip = np.full(
                [self.geo_meta.ny_local], np.nan, dtype=np.float64
            )
        return self._final_supp_precip

    @final_supp_precip.setter
    def final_supp_precip(self, value: Any) -> Any:
        """Setter for final_supp_precip."""
        self._final_supp_precip = value

    @property
    def regridded_mask(self) -> np.ndarray | Any:
        """Get the regridded mask after regridding input forcings to the supplemental precipitation grids."""
        if self._regridded_mask is None:
            self._regridded_mask = np.full(
                [self.geo_meta.ny_local], np.nan, dtype=np.float32
            )
        return self._regridded_mask

    @regridded_mask.setter
    def regridded_mask(self, value: Any) -> Any:
        """Setter for regridded_mask."""
        self._regridded_mask = value


class SupplementalPrecipUnstructured(SupplementalPrecip):
    """Supplemental precipitation class for unstructured grids."""

    def __init__(
        self,
        idx: int = None,
        config_options: ConfigOptions = None,
        geo_meta: GeoMeta = None,
    ) -> None:
        """Initialize SupplementalPrecipUnstructured.  Any subclass-specific attr names are sourced from SUPPPRECIPMOD[classname] in consts.py."""
        super().__init__(idx, config_options, geo_meta)
        for attr in SUPPPRECIPMOD[__class__.__name__]:
            setattr(self, attr, None)

    @property
    def final_supp_precip(self) -> np.ndarray | Any:
        """Get the final supplemental precipitation grid after regridding and temporal interpolation."""
        if self._final_supp_precip is None:
            self._final_supp_precip = np.full(
                [self.geo_meta.ny_local], np.nan, dtype=np.float64
            )
        return self._final_supp_precip

    @final_supp_precip.setter
    def final_supp_precip(self, value: Any) -> Any:
        """Setter for final_supp_precip."""
        self._final_supp_precip = value

    @property
    def regridded_mask(self) -> np.ndarray | Any:
        """Get the regridded mask after regridding input forcings to the supplemental precipitation grids."""
        if self._regridded_mask is None:
            self._regridded_mask = np.full(
                [self.geo_meta.ny_local], np.nan, dtype=np.float32
            )
        return self._regridded_mask

    @regridded_mask.setter
    def regridded_mask(self, value: Any) -> Any:
        """Setter for regridded_mask."""
        self._regridded_mask = value

    @property
    def final_supp_precip_elem(self) -> np.ndarray | Any:
        """Get the final supplemental precipitation grid after regridding and temporal interpolation for unstructured grids."""
        if self._final_supp_precip_elem is None:
            self._final_supp_precip_elem = np.full(
                [self.geo_meta.ny_local_elem], np.nan, dtype=np.float64
            )
        return self._final_supp_precip_elem

    @final_supp_precip_elem.setter
    def final_supp_precip_elem(self, value: Any) -> Any:
        """Setter for final_supp_precip_elem."""
        self._final_supp_precip_elem = value

    @property
    def regridded_mask_elem(self) -> np.ndarray | Any:
        """Get the regridded mask after regridding input forcings to the supplemental precipitation grids for unstructured grids."""
        if self._regridded_mask_elem is None:
            self._regridded_mask_elem = np.full(
                [self.geo_meta.ny_local_elem], np.nan, dtype=np.float32
            )
        return self._regridded_mask_elem

    @regridded_mask_elem.setter
    def regridded_mask_elem(self, value: Any) -> Any:
        """Setter for regridded_mask_elem."""
        self._regridded_mask_elem = value


SUPPPRECIP = {
    "gridded": SupplementalPrecipGridded,
    "unstructured": SupplementalPrecipUnstructured,
    "hydrofabric": SupplementalPrecipHydrofabric,
}


def init_dict(config_options: ConfigOptions, geo_meta: GeoMeta) -> dict:
    """Initialize the supplemental precipitation input dictionary.

    Initial function to create an supplemental dictionary, which
    will contain an abstract class for each supplemental precip product.
    This gets called one time by the parent calling program.
    :param ConfigOptions:
    :return: input_dict - A dictionary defining our inputs.
    """
    input_dict = {}
    for idx in range(0, config_options.number_supp_pcp):
        supp_pcp_key = config_options.supp_precip_forcings[idx]
        input_dict[supp_pcp_key] = SUPPPRECIP[config_options.grid_type](
            idx, config_options, geo_meta
        )
    return input_dict
