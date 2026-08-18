from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# For ESMF + shapely 2.x, shapely must be imported first, to avoid segfault "address not mapped to object" stemming from calls such as:
# /usr/local/esmf/lib/libO/Linux.gfortran.64.openmpi.default/libesmf_fullylinked.so(get_geom+0x36)
import shapely
from scipy import spatial

try:
    import esmpy as ESMF
except ImportError:
    import ESMF

from functools import cached_property, wraps
from typing import Any

import ewts
import xarray as xr

from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.config import (
    ConfigOptions,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.consts import GEOMOD
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.err_handler import (
    log_critical,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.parallel import MpiConfig

LOG = ewts.get_logger(ewts.FORCING_ID)


def set_none(func) -> Any:
    """Set the output of a function to None if spatial_metadata_exists is false."""

    @wraps(func)
    def wrapper(self) -> Any:
        """Set the output of a function to None if spatial_metadata_exists is false."""
        if self.spatial_metadata_exists:
            return func(self)
        else:
            return None

    return wrapper


def broadcast(prop) -> Any:
    """Broadcast the output of a cached_property to all processors."""

    @wraps(prop)
    def wrapper(self) -> Any:
        """Broadcast the output of a cached_property to all processors."""
        result = prop.func(self)
        return self.mpi_config.comm.bcast(result, root=0)

    return cached_property(wrapper)


def barrier(prop) -> Any:
    """Synchronize all cached_property at a barrier."""

    @wraps(prop)
    def wrapper(self) -> Any:
        """Synchronize all cached_property at a barrier."""
        result = prop.func(self)
        self.mpi_config.comm.barrier()
        return result

    return cached_property(wrapper)


def scatter(prop) -> Any:
    """Scatter the output of a cached_property to all processors."""

    @wraps(prop)
    def wrapper(self) -> Any:
        """Scatter the output of a cached_property to all processors.

        Scatter the var array with the var array and config_options.
        If the post_slice boolean is True, then slice the array before returning.
        pass the variable name in to raise an informative error if the scatter fails.
        """
        try:
            var, name, config_options, post_slice = prop.func(self)
            assert isinstance(post_slice, bool)
            assert isinstance(name, str)
            assert isinstance(config_options, ConfigOptions)
            assert isinstance(var, np.ndarray)

            var = self.mpi_config.scatter_array(self, var, config_options)
            if post_slice:
                return var[:, :]
            else:
                return var
        except Exception as e:
            self.config_options.errMsg = (
                f"Unable to subset {name} from geogrid file into ESMF object"
            )
            log_critical(self.config_options, self.mpi_config)
            raise e

    return cached_property(wrapper)


class GeoMeta:
    """GeoMeta class for handling information about the geometry metadata.

    Extract names of variable attributes from each of the input geospatial variables. These
    can change, so we are making this as flexible as possible to accomodate future changes.
    """

    def __init__(self, config_options: ConfigOptions, mpi_config: MpiConfig) -> None:
        """Initialize GeoMeta class variables."""
        self.config_options = config_options
        self.mpi_config = mpi_config
        for attr in GEOMOD[self.__class__.__base__.__name__]:
            setattr(self, attr, None)

    @cached_property
    def spatial_metadata_exists(self) -> bool:
        """Check to make sure the geospatial metadata file exists in the config_options."""
        if self.config_options.spatial_meta is None:
            return False
        else:
            return True

    @cached_property
    def geogrid_ds(self) -> xr.Dataset:
        """Open the geogrid file and return the xarray dataset object."""
        try:
            with xr.open_dataset(self.config_options.geogrid) as ds:
                return ds.load()
        except Exception as e:
            self.config_options.errMsg = "Unable to open geogrid file with xarray"
            log_critical(self.config_options, self.mpi_config)
            raise e

    @cached_property
    @set_none
    def esmf_ds(self) -> xr.Dataset:
        """Open the geospatial metadata file and return the xarray dataset object."""
        try:
            with xr.open_dataset(self.config_options.spatial_meta) as ds:
                esmf_ds = ds.load()
        except Exception as e:
            self.config_options.errMsg = (
                f"Unable to open esmf file: {self.config_options.spatial_meta}"
            )
            log_critical(self.config_options, self.mpi_config)
            raise e
        self._check_variables_exist(esmf_ds)
        return esmf_ds

    def _check_variables_exist(self, esmf_ds: xr.Dataset):
        """Check to make sure the expected variables are present in the geospatial metadata file."""
        if self.mpi_config.rank == 0:
            for var in ["crs", "x", "y"]:
                if var not in esmf_ds.variables.keys():
                    self.config_options.errMsg = f"Unable to locate {var} variable in: {self.config_options.spatial_meta}"
                    log_critical(self.config_options, self.mpi_config)
                    raise Exception

    def ncattrs(self, var: str) -> list:
        """Extract variable attribute names from the geospatial metadata file."""
        return self.get_esmf_var(var).attrs

    def get_var(self, ds: xr.Dataset, var: str) -> xr.DataArray:
        """Get a variable from a xr.Dataset."""
        if var is not None:
            try:
                return ds.variables[var]
            except Exception as e:
                self.config_options.errMsg = f"Unable to extract {var} variable from: {self.config_options.spatial_meta} due to {str(e)}"
                log_critical(self.config_options, self.mpi_config)
                raise e

    def get_geogrid_var(self, var: str) -> xr.DataArray:
        """Get a variable from the geogrid file."""
        return self.get_var(self.geogrid_ds, var)

    def get_esmf_var(self, var: str) -> xr.DataArray:
        """Get a variable from the geospatial metadata file."""
        return self.get_var(self.esmf_ds, var)

    @cached_property
    @set_none
    def x_coord_atts(self) -> dict:
        """Extract x coordinate attribute values from the geospatial metadata file."""
        return self.ncattrs("x")

    @cached_property
    @set_none
    def y_coord_atts(self) -> dict:
        """Extract y coordinate attribute values from the geospatial metadata file."""
        return self.ncattrs("y")

    @cached_property
    @set_none
    def crs_atts(self) -> dict:
        """Extract crs coordinate attribute values from the geospatial metadata file."""
        return self.ncattrs("crs")

    @cached_property
    @set_none
    def spatial_global_atts(self) -> dict:
        """Extract global attribute values from the geospatial metadata file."""
        if self.mpi_config.rank == 0:
            try:
                return self.esmf_ds.attrs
            except Exception as e:
                self.config_options.errMsg = f"Unable to extract global attributes from: {self.config_options.spatial_meta}"
                log_critical(self.config_options, self.mpi_config)
                raise e

    def extract_coords(self, dimension: str) -> np.ndarray:
        """Extract coordinate values from the geospatial metadata file."""
        if self.mpi_config.rank == 0:
            if len(self.get_esmf_var(dimension).shape) == 1:
                return self.get_esmf_var(dimension)[:].data
            elif len(self.get_esmf_var(dimension).shape) == 2:
                return self.get_esmf_var(dimension)[:, :].data

    @cached_property
    @set_none
    def x_coords(self) -> np.ndarray:
        """Extract x coordinate values from the geospatial metadata file."""
        return self.extract_coords("x")

    @cached_property
    @set_none
    def y_coords(self) -> np.ndarray:
        """Extract y coordinate values from the geospatial metadata file.

        Check to see if the Y coordinates are North-South. If so, flip them.
        """
        if self.mpi_config.rank == 0:
            y_coords = self.extract_coords("y")
            if len(self.get_esmf_var("y").shape) == 1:
                if y_coords[1] < y_coords[0]:
                    y_coords[:] = np.flip(y_coords[:], axis=0)
            elif len(self.get_esmf_var("y").shape) == 2:
                if y_coords[1, 0] > y_coords[0, 0]:
                    y_coords[:, :] = np.flipud(y_coords[:, :])
            return y_coords


class GriddedGeoMeta(GeoMeta):
    """Class for handling information about the gridded domains for forcing."""

    def __init__(self, config_options: ConfigOptions, mpi_config: MpiConfig) -> None:
        """Initialize GriddedGeoMeta class variables.

        Initialization function to initialize ESMF through ESMPy,
        calculate the global parameters of the WRF-Hydro grid
        being processed to, along with the local parameters
        for this particular processor.
        :return:
        """
        super().__init__(config_options, mpi_config)
        for attr in GEOMOD[self.__class__.__name__]:
            setattr(self, attr, None)

    @broadcast
    @cached_property
    def nx_global(self) -> int:
        """Get the global x dimension size for the gridded domain."""
        if self.mpi_config.rank == 0:
            try:
                if self.ndim_lat == 3:
                    return self.lat_var.shape[2]
                elif self.ndim_lat == 2:
                    return self.lat_var.shape[1]
                else:
                    # NOTE Is this correct? using lon_var
                    return self.lon_var.shape[0]
            except Exception as e:
                self.config_options.errMsg = f"Unable to extract X dimension size from {self.config_options.lon_var} in: {self.config_options.geogrid}"
                log_critical(self.config_options, self.mpi_config)
                raise e

    @broadcast
    @cached_property
    def ny_global(self) -> int:
        """Get the global y dimension size for the gridded domain."""
        if self.mpi_config.rank == 0:
            try:
                if self.ndim_lat == 3:
                    return self.lat_var.shape[1]
                else:
                    return self.lat_var.shape[0]
            except Exception as e:
                self.config_options.errMsg = f"Unable to extract Y dimension size from {self.config_options.lat_var} in: {self.config_options.geogrid}"
                log_critical(self.config_options, self.mpi_config)
                raise e

    @cached_property
    def ndim_lat(self) -> int:
        """Get the number of dimensions for the latitude variable."""
        return self.lat_var.ndim

    @cached_property
    def ndim_lon(self) -> int:
        """Get the number of dimensions for the longitude variable."""
        return self.lon_var.ndim

    @broadcast
    @cached_property
    def dy_meters(self) -> float:
        """Get the DY distance in meters for the latitude variable."""
        if self.mpi_config.rank == 0:
            try:
                if self.ndim_lat == 3:
                    return self.geogrid_ds.DY
                elif self.ndim_lat == 2:
                    return self.lat_var.dy
                else:
                    if self.config_options.input_forcings[0] != 23:
                        return self.lat_var.dy
                    else:
                        # Manually input the grid spacing since ERA5-Interim does not
                        # internally have this geospatial information within the netcdf file
                        return 31000
            except Exception as e:
                self.config_options.errMsg = f"Unable to extract DY global attribute in: {self.config_options.geogrid}"
                log_critical(self.config_options, self.mpi_config)
                raise e

    @broadcast
    @cached_property
    def dx_meters(self) -> float:
        """Get the DX distance in meters for the longitude variable."""
        if self.mpi_config.rank == 0:
            try:
                if self.ndim_lat == 3:
                    return self.geogrid_ds.DX
                elif self.ndim_lat == 2:
                    return self.lon_var.dx
                else:
                    if self.config_options.input_forcings[0] != 23:
                        return self.lon_var.dx
                    else:
                        # Manually input the grid spacing since ERA5-Interim does not
                        # internally have this geospatial information within the netcdf file
                        return 31000
            except Exception as e:
                self.config_options.errMsg = f"Unable to extract dx metadata attribute in: {self.config_options.geogrid}"
                log_critical(self.config_options, self.mpi_config)
                raise e

    @cached_property
    def esmf_grid(self) -> ESMF.Grid:
        """Create the ESMF grid object for the gridded domain."""
        try:
            return ESMF.Grid(
                np.array([self.ny_global, self.nx_global]),
                staggerloc=ESMF.StaggerLoc.CENTER,
                coord_sys=ESMF.CoordSys.SPH_DEG,
            )
        except Exception as e:
            self.config_options.errMsg = f"Unable to create ESMF grid for WRF-Hydro geogrid: {self.config_options.geogrid}"
            log_critical(self.config_options, self.mpi_config)
            raise e

    @cached_property
    def esmf_lat(self) -> np.ndarray:
        """Get the ESMF latitude grid."""
        esmf_lat = self.esmf_grid.get_coords(1)
        esmf_lat[:, :] = self.latitude_grid
        return esmf_lat

    @cached_property
    def esmf_lon(self) -> np.ndarray:
        """Get the ESMF longitude grid."""
        esmf_lon = self.esmf_grid.get_coords(0)
        esmf_lon[:, :] = self.longitude_grid
        return esmf_lon

    @scatter
    @cached_property
    def latitude_grid(self) -> np.ndarray:
        """Get the latitude grid for the gridded domain."""
        # Scatter global XLAT_M grid to processors..
        if self.mpi_config.rank == 0:
            if self.ndim_lat == 3:
                var_tmp = self.lat_var[0, :, :]
            elif self.ndim_lat == 2:
                var_tmp = self.lat_var[:, :]
            elif self.ndim_lat == 1:
                lat = self.lat_var[:]
                lon = self.lon_var[:]
                var_tmp = np.meshgrid(lon, lat)[1]

            # Flag to grab entire array for AWS slicing
            if self.config_options.aws:
                self.lat_bounds = var_tmp
        else:
            var_tmp = None
        return var_tmp, "latitude_grid", self.config_options, False

    @cached_property
    def lon_var(self) -> xr.DataArray:
        """Get the longitude variable from the geospatial metadata file."""
        return self.get_geogrid_var(self.config_options.lon_var)

    @cached_property
    def lat_var(self) -> xr.DataArray:
        """Get the latitude variable from the geospatial metadata file."""
        return self.get_geogrid_var(self.config_options.lat_var)

    @scatter
    @cached_property
    def longitude_grid(self) -> np.ndarray:
        """Get the longitude grid for the gridded domain."""
        # Scatter global XLONG_M grid to processors..
        if self.mpi_config.rank == 0:
            if (
                self.ndim_lat == 3
            ):  # NOTE The original code has lat here... should it maybe be lon instead?
                var_tmp = self.lon_var[0, :, :]
            elif self.ndim_lon == 2:
                var_tmp = self.lon_var[:, :]
            elif self.ndim_lon == 1:
                lat = self.lat_var[:]
                lon = self.lon_var[:]
                var_tmp = np.meshgrid(lon, lat)[0]

            # Flag to grab entire array for AWS slicing
            if self.config_options.aws:
                self.lon_bounds = var_tmp
        else:
            var_tmp = None

        return var_tmp, "longitude_grid", self.config_options, False

    @cached_property
    def cosalpha_var(self) -> xr.DataArray:
        """Get the COSALPHA variable from the geospatial metadata file."""
        return self.get_geogrid_var(self.config_options.cosalpha_var)

    @scatter
    @cached_property
    def cosa_grid(self) -> np.ndarray:
        """Get the COSALPHA grid for the gridded domain."""
        if (
            self.config_options.cosalpha_var is not None
            and self.config_options.sinalpha_var is not None
        ):
            # Scatter the COSALPHA,SINALPHA grids to the processors.
            if self.mpi_config.rank == 0:
                if self.cosalpha_var.ndim == 3:
                    cosa = self.cosa_grid_from_geogrid_n3
                else:
                    cosa = self.cosalpha_var[:, :]

            else:
                cosa = None

            return cosa, "cosa", self.config_options, True

    @cached_property
    def sinalpha_var(self) -> xr.DataArray:
        """Get the SINALPHA variable from the geospatial metadata file."""
        return self.get_geogrid_var(self.config_options.sinalpha_var)

    @cached_property
    def sina_grid(self) -> np.ndarray:
        """Get the SINALPHA grid for the gridded domain."""
        if (
            self.config_options.cosalpha_var is not None
            and self.config_options.sinalpha_var is not None
        ):
            if self.mpi_config.rank == 0:
                if self.sinalpha_var.ndim == 3:
                    sina = self.sina_grid_from_geogrid_n3
                else:
                    sina = self.sinalpha_var[:, :]
            else:
                sina = None

            return sina, "sina", self.config_options, True

    @cached_property
    def hgt_var(self) -> xr.DataArray:
        """Get the HGT variable from the geospatial metadata file."""
        return self.get_geogrid_var(self.config_options.hgt_var)

    @scatter
    @cached_property
    def height(self) -> np.ndarray:
        """Get the height grid for the gridded domain.

        Used for downscaling purposes.
        """
        if self.config_options.hgt_var is not None:
            if self.mpi_config.rank == 0:
                if self.hgt_var.ndim == 3:
                    hgt = self.hgt_grid_from_geogrid_n3
                else:
                    hgt = self.hgt_var[:, :]
            else:
                hgt = None

            return hgt, "height", self.config_options, False

    @cached_property
    def slope_var(self) -> xr.DataArray:
        """Get the slope variable from the geospatial metadata file."""
        return self.get_geogrid_var(self.config_options.slope_var)

    @cached_property
    def slope_azimuth_var(self) -> xr.DataArray:
        """Get the slope azimuth variable from the geospatial metadata file."""
        return self.get_geogrid_var(self.config_options.slope_azimuth_var)

    @cached_property
    def dx(self) -> np.ndarray:
        """Calculate the dx distance in meters for the longitude variable."""
        dx = np.empty(
            (
                self.lat_var.shape[0],
                self.lon_var.shape[0],
            ),
            dtype=float,
        )
        dx[:] = self.lon_var.dx
        return dx

    @cached_property
    def dy(self) -> np.ndarray:
        """Calculate the dy distance in meters for the latitude variable."""
        dy = np.empty(
            (
                self.lat_var.shape[0],
                self.lon_var.shape[0],
            ),
            dtype=float,
        )
        dy[:] = self.lat_var.dy
        return dy

    @cached_property
    def dz(self) -> np.ndarray:
        """Calculate the dz distance in meters for the height variable."""
        dz_init = np.diff(self.hgt_var, axis=0)
        dz = np.empty(self.dx.shape, dtype=float)
        dz[0 : dz_init.shape[0], 0 : dz_init.shape[1]] = dz_init
        dz[dz_init.shape[0] :, :] = dz_init[-1, :]
        return dz

    @scatter
    @cached_property
    def slope(self) -> np.ndarray:
        """Calculate slope grids needed for incoming shortwave radiation downscaling.

        Calculate slope from sina_grid, cosa_grid, and height variables if they are
        present in the geogrid file, otherwise calculate slope from slope and slope
        azimuth variables, and if those are not present, calculate slope from height variable.


        Calculate grid coordinates dx distances in meters
        based on general geospatial formula approximations
        on a spherical grid.
        """
        if (
            self.config_options.cosalpha_var is not None
            and self.config_options.sinalpha_var is not None
        ):
            slope = self.slope_from_cosalpha_sinalpha
        elif (
            self.config_options.slope_var is not None
            and self.config_options.slope_azimuth_var is not None
        ):
            slope = self.slope_from_slope_azimuth
        elif self.config_options.hgt_var is not None:
            slope = self.slope_from_height
        else:
            raise Exception(
                "Unable to calculate slope grid for incoming shortwave radiation downscaling. No geospatial metadata variables provided to calculate slope."
            )
        return slope, "slope", self.config_options, True

    @scatter
    @cached_property
    def slp_azi(self) -> np.ndarray:
        """Calculate slope azimuth grids needed for incoming shortwave radiation downscaling.

        Calculate slp_azi from sina_grid, cosa_grid, and height variables if they are
        present in the geogrid file, otherwise calculate slope from slope and slope
        azimuth variables, and if those are not present, calculate slope from height variable.
        """
        if (
            self.config_options.cosalpha_var is not None
            and self.config_options.sinalpha_var is not None
        ):
            slp_azi = self.slp_azi_from_cosalpha_sinalpha
        elif (
            self.config_options.slope_var is not None
            and self.config_options.slope_azimuth_var is not None
        ):
            slp_azi = self.slp_azi_from_slope_azimuth

        elif self.config_options.hgt_var is not None:
            slp_azi = self.slp_azi_from_height
        else:
            raise Exception(
                "Unable to calculate slope azimuth grid for incoming shortwave radiation downscaling. No geospatial metadata variables provided to calculate slope azimuth."
            )

        return slp_azi, "slp_azi", self.config_options, True

    @cached_property
    def slp_azi_from_slope_azimuth(self) -> np.ndarray:
        """Calculate slope azimuth from slope and slope azimuth variables."""
        if self.mpi_config.rank == 0:
            if self.slope_azimuth_var.ndim == 3:
                return self.slope_azimuth_var[0, :, :]
            else:
                return self.slope_azimuth_var[:, :]

    @cached_property
    def slp_azi_from_height(self) -> np.ndarray:
        """Calculate slope azimuth from height variable."""
        if self.mpi_config.rank == 0:
            return (180 / np.pi) * np.arctan(self.dx / self.dy)

    @cached_property
    def slope_from_height(self) -> np.ndarray:
        """Calculate slope from height variable."""
        if self.mpi_config.rank == 0:
            return self.dz / np.sqrt((self.dx**2) + (self.dy**2))

    @cached_property
    def slope_from_slope_azimuth(self) -> np.ndarray:
        """Calculate slope from slope and slope azimuth variables."""
        if self.mpi_config.rank == 0:
            if self.slope_var.ndim == 3:
                return self.slope_var[0, :, :]
            else:
                return self.slope_var[:, :]

    @cached_property
    def slope_from_cosalpha_sinalpha(self) -> np.ndarray:
        """Calculate slope from COSALPHA and SINALPHA variables."""
        if self.mpi_config.rank == 0:
            slope_tmp = np.arctan(
                (self.hx[self.ind_orig] ** 2 + self.hy[self.ind_orig] ** 2) ** 0.5
            )
            slope_tmp[np.where(slope_tmp < 1e-4)] = 0.0
            return slope_tmp

    @cached_property
    def slp_azi_from_cosalpha_sinalpha(self) -> np.ndarray:
        """Calculate slope azimuth from COSALPHA and SINALPHA variables."""
        if self.mpi_config.rank == 0:
            slp_azi = np.empty([self.ny_global, self.nx_global], np.float32)
            slp_azi[np.where(self.slope_from_cosalpha_sinalpha < 1e-4)] = 0.0
            ind_valesmf_ds = np.where(self.slope_from_cosalpha_sinalpha >= 1e-4)
            slp_azi[ind_valesmf_ds] = (
                np.arctan2(self.hx[ind_valesmf_ds], self.hy[ind_valesmf_ds]) + math.pi
            )
            ind_valesmf_ds = np.where(self.cosa_grid_from_geogrid_n3 >= 0.0)
            slp_azi[ind_valesmf_ds] = slp_azi[ind_valesmf_ds] - np.arcsin(
                self.sina_grid_from_geogrid_n3[ind_valesmf_ds]
            )
            ind_valesmf_ds = np.where(self.cosa_grid_from_geogrid_n3 < 0.0)
            slp_azi[ind_valesmf_ds] = slp_azi[ind_valesmf_ds] - (
                math.pi - np.arcsin(self.sina_grid_from_geogrid_n3[ind_valesmf_ds])
            )
            return slp_azi

    @cached_property
    def ind_orig(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the indices of the original grid points for the height variable."""
        return np.where(self.hgt_grid_from_geogrid_n3 == self.hgt_grid_from_geogrid_n3)

    @cached_property
    def hx(self) -> np.ndarray:
        """Calculate the slope in the x direction from the height variable."""
        rdx = 1.0 / self.dx_meters
        msftx = 1.0
        toposlpx = np.empty([self.ny_global, self.nx_global], np.float32)
        ip_diff = np.empty([self.ny_global, self.nx_global], np.int32)
        hx = np.empty([self.ny_global, self.nx_global], np.float32)

        # Create index arrays that will be used to calculate slope.
        x_tmp = np.arange(self.nx_global)
        x_grid = np.tile(x_tmp[:], (self.ny_global, 1))
        ind_ip1 = ((self.ind_orig[0]), (self.ind_orig[1] + 1))
        ind_im1 = ((self.ind_orig[0]), (self.ind_orig[1] - 1))
        ind_ip1[1][np.where(ind_ip1[1] >= self.nx_global)] = self.nx_global - 1
        ind_im1[1][np.where(ind_im1[1] < 0)] = 0

        ip_diff[self.ind_orig] = x_grid[ind_ip1] - x_grid[ind_im1]
        toposlpx[self.ind_orig] = (
            (
                self.hgt_grid_from_geogrid_n3[ind_ip1]
                - self.hgt_grid_from_geogrid_n3[ind_im1]
            )
            * msftx
            * rdx
        ) / ip_diff[self.ind_orig]
        hx = np.empty([self.ny_global, self.nx_global], np.float32)
        hx[self.ind_orig] = toposlpx[self.ind_orig]
        return hx

    @cached_property
    def hy(self) -> np.ndarray:
        """Calculate the slope in the y direction from the height variable."""
        rdy = 1.0 / self.dy_meters
        msfty = 1.0
        toposlpy = np.empty([self.ny_global, self.nx_global], np.float32)
        jp_diff = np.empty([self.ny_global, self.nx_global], np.int32)
        hy = np.empty([self.ny_global, self.nx_global], np.float32)

        # Create index arrays that will be used to calculate slope.
        y_tmp = np.arange(self.ny_global)
        y_grid = np.repeat(y_tmp[:, np.newaxis], self.nx_global, axis=1)
        ind_jp1 = ((self.ind_orig[0] + 1), (self.ind_orig[1]))
        ind_jm1 = ((self.ind_orig[0] - 1), (self.ind_orig[1]))
        ind_jp1[0][np.where(ind_jp1[0] >= self.ny_global)] = self.ny_global - 1
        ind_jm1[0][np.where(ind_jm1[0] < 0)] = 0

        jp_diff[self.ind_orig] = y_grid[ind_jp1] - y_grid[ind_jm1]
        toposlpy[self.ind_orig] = (
            (
                self.hgt_grid_from_geogrid_n3[ind_jp1]
                - self.hgt_grid_from_geogrid_n3[ind_jm1]
            )
            * msfty
            * rdy
        ) / jp_diff[self.ind_orig]
        hy[self.ind_orig] = toposlpy[self.ind_orig]
        return hy

    @cached_property
    def x_lower_bound(self) -> float:
        """Get the local x lower bound for this processor."""
        return self.esmf_grid.lower_bounds[ESMF.StaggerLoc.CENTER][1]

    @cached_property
    def x_upper_bound(self) -> float:
        """Get the local x upper bound for this processor."""
        return self.esmf_grid.upper_bounds[ESMF.StaggerLoc.CENTER][1]

    @cached_property
    def y_lower_bound(self) -> float:
        """Get the local y lower bound for this processor."""
        return self.esmf_grid.lower_bounds[ESMF.StaggerLoc.CENTER][0]

    @cached_property
    def y_upper_bound(self) -> float:
        """Get the local y upper bound for this processor."""
        return self.esmf_grid.upper_bounds[ESMF.StaggerLoc.CENTER][0]

    @cached_property
    def nx_local(self) -> int:
        """Get the local x dimension size for this processor."""
        return self.x_upper_bound - self.x_lower_bound

    @cached_property
    def ny_local(self) -> int:
        """Get the local y dimension size for this processor."""
        return self.y_upper_bound - self.y_lower_bound

    @cached_property
    def sina_grid_from_geogrid_n3(self) -> np.ndarray:
        """Get the SINALPHA grid for the gridded domain directly from the geogrid file."""
        try:
            return self.check_grid(self.sinalpha_var[0, :, :])
        except Exception as e:
            self.config_options.errMsg = f"Unable to extract {self.config_options.sinalpha_var} from: {self.config_options.geogrid}"
            log_critical(self.config_options, self.mpi_config)
            raise e

    def check_grid(self, grid: np.ndarray) -> np.ndarray:
        """Check to make sure the grid dimensions match the expected dimensions for the gridded domain."""
        if grid.shape[0] != self.ny_global or grid.shape[1] != self.nx_global:
            self.config_options.errMsg = (
                f"Grid dimensions mismatch in: {self.config_options.geogrid}"
            )
            log_critical(self.config_options, self.mpi_config)
            raise ValueError(self.config_options.errMsg)
        return grid

    @cached_property
    def cosa_grid_from_geogrid_n3(self) -> np.ndarray:
        """Get the COSALPHA grid for the gridded domain directly from the geogrid file."""
        try:
            return self.check_grid(self.cosalpha_var[0, :, :])
        except Exception as e:
            self.config_options.errMsg = f"Unable to extract {self.config_options.cosalpha_var} from: {self.config_options.geogrid}"
            log_critical(self.config_options, self.mpi_config)
            raise e

    @cached_property
    def hgt_grid_from_geogrid_n3(self) -> np.ndarray:
        """Get the HGT_M grid for the gridded domain directly from the geogrid file."""
        try:
            return self.check_grid(self.hgt_var[0, :, :])
        except Exception as e:
            self.config_options.errMsg = f"Unable to extract {self.config_options.hgt_var} from: {self.config_options.geogrid}"
            log_critical(self.config_options, self.mpi_config)
            raise e


class HydrofabricGeoMeta(GeoMeta):
    """Class for handling information about the hydrofabric domain forcing."""

    def __init__(self, config_options: ConfigOptions, mpi_config: MpiConfig):
        """Initialize HydrofabricGeoMeta class variables.

        Initialization function to initialize ESMF through ESMPy,
        calculate the global parameters of the hydrofabric
        being processed to, along with the local parameters
        for this particular processor.
        :return:
        """
        super().__init__(config_options, mpi_config)
        for attr in GEOMOD[self.__class__.__name__]:
            setattr(self, attr, None)

    @cached_property
    def lat_bounds(self) -> np.ndarray:
        """Get the latitude bounds for the hydrofabric domain."""
        bounds = self.get_bound(1)
        if bounds is not None:
            return bounds.values

    @cached_property
    def lon_bounds(self) -> np.ndarray:
        """Get the longitude bounds for the hydrofabric domain."""
        bounds = self.get_bound(0)
        if bounds is not None:
            return bounds.values

    def get_bound(self, dim: int) -> np.ndarray:
        """Get the longitude or latitude bounds for the hydrofabric domain."""
        if self.mpi_config.rank == 0:
            if self.config_options.aws:
                return self.get_geogrid_var(self.config_options.nodecoords_var)[:, dim]

    @broadcast
    @cached_property
    def elementcoords_global(self) -> np.ndarray:
        """Get the global element coordinates for the hydrofabric domain."""
        return self.get_geogrid_var(self.config_options.elemcoords_var).values

    @broadcast
    @cached_property
    def nx_global(self) -> int:
        """Get the global x dimension size for the hydrofabric domain."""
        return self.elementcoords_global.shape[0]

    @broadcast
    @cached_property
    def ny_global(self) -> int:
        """Get the global y dimension size for the hydrofabric domain.

        Same as nx_global.
        """
        return self.nx_global

    @cached_property
    def esmf_grid(self) -> ESMF.Mesh:
        """Create the ESMF Mesh object for the hydrofabric domain."""
        try:
            return ESMF.Mesh(
                filename=self.config_options.geogrid, filetype=ESMF.FileFormat.ESMFMESH
            )
        except Exception as e:
            LOG.critical(
                f"Unable to create ESMF Mesh: {self.config_options.geogrid} "
                f"due to {str(e)}"
            )
            raise e

    @cached_property
    def latitude_grid(self) -> np.ndarray:
        """Get the latitude grid for the hydrofabric domain."""
        return self.esmf_grid.coords[1][1]

    @cached_property
    def longitude_grid(self) -> np.ndarray:
        """Get the longitude grid for the hydrofabric domain."""
        return self.esmf_grid.coords[1][0]

    @cached_property
    def pet_element_inds(self) -> np.ndarray:
        """Get the PET element indices for the hydrofabric domain."""
        try:
            tree = spatial.KDTree(self.elementcoords_global)
            return tree.query(
                np.column_stack([self.longitude_grid, self.latitude_grid])
            )[1]
        except Exception as e:
            LOG.critical(
                f"Failed to open mesh file: {self.config_options.geogrid} "
                f"due to {str(e)}"
            )
            raise e

    @cached_property
    def element_ids(self) -> np.ndarray:
        """Get the element IDs for the hydrofabric domain."""
        return self.element_ids_global[self.pet_element_inds]

    @broadcast
    @cached_property
    def element_ids_global(self) -> np.ndarray:
        """Get the global element IDs for the hydrofabric domain."""
        return self.get_geogrid_var(self.config_options.element_id_var).values

    @broadcast
    @cached_property
    def heights_global(self) -> np.ndarray:
        """Get the global heights for the hydrofabric domain."""
        return self.get_geogrid_var(self.config_options.hgt_var)

    @cached_property
    def height(self) -> np.ndarray:
        """Get the height grid for the hydrofabric domain."""
        if self.config_options.hgt_var is not None:
            return self.heights_global[self.pet_element_inds]

    @cached_property
    def slope(self) -> np.ndarray:
        """Get the slopes for the hydrofabric domain."""
        if self.slopes_global is not None:
            return self.slopes_global[self.pet_element_inds]

    @cached_property
    def slp_azi(self) -> np.ndarray:
        """Get the slope azimuths for the hydrofabric domain."""
        if self.slp_azi_global is not None:
            return self.slp_azi_global[self.pet_element_inds]

    @cached_property
    def mesh_inds(self) -> np.ndarray:
        """Get the mesh indices for the hydrofabric domain."""
        return self.pet_element_inds

    @broadcast
    @cached_property
    def slopes_global(self) -> np.ndarray:
        """Get the global slopes for the hydrofabric domain."""
        return self.get_geogrid_var(self.config_options.slope_var)

    @cached_property
    def slp_azi_global(self) -> np.ndarray:
        """Get the global slope azimuths for the hydrofabric domain."""
        return self.get_geogrid_var(self.config_options.slope_azimuth_var)

    @cached_property
    def nx_local(self) -> int:
        """Get the local x dimension size for this processor."""
        return len(self.esmf_grid.coords[1][1])

    @cached_property
    def ny_local(self) -> int:
        """Get the local y dimension size for this processor."""
        return len(self.esmf_grid.coords[1][1])


class UnstructuredGeoMeta(GeoMeta):
    """Class for handling information about the hydrofabric domain forcing."""

    def __init__(self, config_options: ConfigOptions, mpi_config: MpiConfig) -> None:
        """Initialize HydrofabricGeoMeta class variables.

        Initialization function to initialize ESMF through ESMPy,
        calculate the global parameters of the unstructured mesh
        being processed to, along with the local parameters
        for this particular processor.
        :return:
        """
        super().__init__(config_options, mpi_config)
        for attr in GEOMOD[self.__class__.__name__]:
            setattr(self, attr, None)

    @broadcast
    @cached_property
    def nx_global(self) -> int:
        """Get the global x dimension size for the unstructured domain."""
        return self.get_geogrid_var(self.config_options.nodecoords_var).shape[0]

    @broadcast
    @cached_property
    def ny_global(self) -> int:
        """Get the global y dimension size for the unstructured domain."""
        return self.get_geogrid_var(self.config_options.nodecoords_var).shape[0]

    @broadcast
    @cached_property
    def nx_global_elem(self) -> int:
        """Get the global x dimension size for the unstructured domain elements."""
        return self.get_esmf_var(self.config_options.elemcoords_var).shape[0]

    @broadcast
    @cached_property
    def ny_global_elem(self) -> int:
        """Get the global y dimension size for the unstructured domain elements."""
        return self.get_esmf_var(self.config_options.elemcoords_var).shape[0]

    @cached_property
    def lon_bounds(self) -> np.ndarray:
        """Get the longitude bounds for the unstructured domain."""
        bounds = self.get_bound(0)
        if bounds is not None:
            return bounds.values

    @cached_property
    def lat_bounds(self) -> np.ndarray:
        """Get the latitude bounds for the unstructured domain."""
        bounds = self.get_bound(1)
        if bounds is not None:
            return bounds.values

    def get_bound(self, dim: int) -> np.ndarray:
        """Get the longitude or latitude bounds for the unstructured domain."""
        if self.mpi_config.rank == 0:
            # Flag to grab entire array for AWS slicing
            if self.config_options.aws:
                return self.get_esmf_var(self.config_options.nodecoords_var)[:][:, dim]

    @cached_property
    def esmf_grid(self) -> ESMF.Mesh:
        """Create the ESMF grid object for the unstructured domain.

        Removed argument coord_sys=ESMF.CoordSys.SPH_DEG since we are always reading from a file
        From ESMF documentation
        If you create a mesh from a file (like NetCDF/ESMF-Mesh), coord_sys is ignored. The mesh’s coordinate system should be embedded in the file or inferred.
        """
        try:
            return ESMF.Mesh(
                filename=self.config_options.geogrid, filetype=ESMF.FileFormat.ESMFMESH
            )
        except Exception as e:
            self.config_options.errMsg = f"Unable to create ESMF Mesh from geogrid file: {self.config_options.geogrid}"
            log_critical(self.config_options, self.mpi_config)
            raise e

    @cached_property
    def latitude_grid(self) -> np.ndarray:
        """Get the latitude grid for the unstructured domain.

        Place the local lat/lon grid slices from the parent geogrid file into
        the ESMF lat/lon grids that have already been seperated by processors.
        """
        return self.esmf_grid.coords[0][1]

    @cached_property
    def latitude_grid_elem(self) -> np.ndarray:
        """Get the latitude grid for the unstructured domain elements.

        Place the local lat/lon grid slices from the parent geogrid file into
        the ESMF lat/lon grids that have already been seperated by processors.
        """
        return self.esmf_grid.coords[1][1]

    @cached_property
    def longitude_grid(self) -> np.ndarray:
        """Get the longitude grid for the unstructured domain.

        Place the local lat/lon grid slices from the parent geogrid file into
        the ESMF lat/lon grids that have already been seperated by processors.
        """
        return self.esmf_grid.coords[0][0]

    @cached_property
    def longitude_grid_elem(self) -> np.ndarray:
        """Get the longitude grid for the unstructured domain elements.

        Place the local lat/lon grid slices from the parent geogrid file into
        the ESMF lat/lon grids that have already been seperated by processors.
        """
        return self.esmf_grid.coords[1][0]

    @cached_property
    def pet_element_inds(self) -> np.ndarray:
        """Get the local node indices for the unstructured domain elements."""
        # Get lat and lon global variables for pet extraction of indices
        elementcoords_global = self.get_var(
            self.geogrid_ds, self.config_options.elemcoords_var
        )[:].data
        # Find the corresponding local indices to slice global heights and slope
        # variables that are based on the partitioning on the unstructured mesh
        pet_elementcoords = np.empty((len(self.latitude_grid_elem), 2), dtype=float)
        pet_elementcoords[:, 0] = self.longitude_grid_elem
        pet_elementcoords[:, 1] = self.latitude_grid_elem
        return spatial.KDTree(elementcoords_global).query(pet_elementcoords)[1]

    @cached_property
    def pet_node_inds(self) -> np.ndarray:
        """Get the local node indices for the unstructured domain nodes."""
        # Get lat and lon global variables for pet extraction of indices
        nodecoords_global = self.get_var(
            self.geogrid_ds, self.config_options.nodecoords_var
        )[:].data
        # Find the corresponding local indices to slice global heights and slope
        # variables that are based on the partitioning on the unstructured mesh
        pet_nodecoords = np.empty((len(self.latitude_grid), 2), dtype=float)
        pet_nodecoords[:, 0] = self.longitude_grid
        pet_nodecoords[:, 1] = self.latitude_grid

        return spatial.KDTree(nodecoords_global).query(pet_nodecoords)[1]

    @cached_property
    def slope(self) -> np.ndarray:
        """Get the slope grid for the unstructured domain."""
        # NOTE this is a note/commented out code from before refactor on 2/19/2026.
        # Not accepting cosalpha and sinalpha at this time for unstructured meshes, only
        # accepting the pre-calculated slope and slope azmiuth variables if available,
        # otherwise calculate slope from height estimates
        # if(config_options.cosalpha_var != None and config_options.sinalpha_var != None):
        # self.cosa_grid = esmf_ds.variables[config_options.cosalpha_var][:].data[pet_node_inds]
        # self.sina_grid = esmf_ds.variables[config_options.sinalpha_var][:].data[pet_node_inds]
        # slope_tmp, slp_azi_tmp = self.calc_slope(esmf_ds,config_options)
        # self.slope = slope_node_tmp[pet_node_inds]
        # self.slp_azi = slp_azi_node_tmp[pet_node_inds]
        if (
            self.config_options.slope_var is not None
            and self.config_options.slp_azi_var is not None
        ):
            return self.get_geogrid_var(self.config_options.slope_var)[
                self.pet_node_inds
            ]
        elif self.config_options.hgt_var is not None:
            return (
                self.dz_node
                / np.sqrt((self.dx_node**2) + (self.dy_node**2))[self.pet_node_inds]
            )
        else:
            raise ValueError(
                "Unable to calculate slope grid for incoming shortwave radiation downscaling. No geospatial metadata variables provided to calculate slope."
            )

    @cached_property
    def slp_azi(self) -> np.ndarray:
        """Get the slope azimuth grid for the unstructured domain."""
        if (
            self.config_options.slope_var is not None
            and self.config_options.slp_azi_var is not None
        ):
            return self.get_geogrid_var(self.config_options.slope_azimuth_var)[
                self.pet_node_inds
            ]
        elif self.config_options.hgt_var is not None:
            return (180 / np.pi) * np.arctan(self.dx_node / self.dy_node)[
                self.pet_node_inds
            ]

    @cached_property
    def slope_elem(self) -> np.ndarray:
        """Get the slope grid for the unstructured domain elements."""
        if (
            self.config_options.slope_var is not None
            and self.config_options.slp_azi_var is not None
        ):
            return self.get_geogrid_var(self.config_options.slope_var_elem)[:].data[
                self.pet_element_inds
            ]
        elif self.config_options.hgt_var is not None:
            return (
                self.dz_elem
                / np.sqrt((self.dx_elem**2) + (self.dy_elem**2))[self.pet_element_inds]
            )

    @cached_property
    def slp_azi_elem(self) -> np.ndarray:
        """Get the slope azimuth grid for the unstructured domain elements."""
        if (
            self.config_options.slope_var is not None
            and self.config_options.slp_azi_var is not None
        ):
            return self.get_var(
                self.geogrid_ds, self.config_options.slope_azimuth_var_elem
            )[:].data[self.pet_element_inds]
        elif self.config_options.hgt_var is not None:
            return (180 / np.pi) * np.arctan(self.dx_elem / self.dy_elem)[
                self.pet_element_inds
            ]

    @cached_property
    def height(self) -> np.ndarray:
        """Get the height grid for the unstructured domain nodes."""
        if (
            self.config_options.slope_var is not None
            and self.config_options.slp_azi_var is not None
        ):
            return self.get_geogrid_var(self.config_options.hgt_var)[:].data[
                self.pet_node_inds
            ]
        elif self.config_options.hgt_var is not None:
            return self.self.get_geogrid_var(self.config_options.hgt_var)[:].data[
                self.pet_node_inds
            ]

    @cached_property
    def height_elem(self) -> np.ndarray:
        """Get the height grid for the unstructured domain elements."""
        if (
            self.config_options.slope_var is not None
            and self.config_options.slp_azi_var is not None
        ):
            return self.get_geogrid_var(self.config_options.hgt_elem_var)[:].data[
                self.pet_element_inds
            ]
        elif self.config_options.hgt_var is not None:
            return self.get_geogrid_var(self.config_options.hgt_elem_var)[:].data[
                self.pet_element_inds
            ]

    @cached_property
    def node_lons(self) -> np.ndarray:
        """Get the longitude grid for the unstructured domain nodes."""
        return self.get_geogrid_var(self.config_options.nodecoords_var)[:][:, 0]

    @cached_property
    def node_lats(self) -> np.ndarray:
        """Get the latitude grid for the unstructured domain nodes."""
        return self.get_geogrid_var(self.config_options.nodecoords_var)[:][:, 1]

    @cached_property
    def elem_lons(self) -> np.ndarray:
        """Get the longitude grid for the unstructured domain elements."""
        return self.get_geogrid_var(self.config_options.elemcoords_var)[:][:, 0]

    @cached_property
    def elem_lats(self) -> np.ndarray:
        """Get the latitude grid for the unstructured domain elements."""
        return self.get_geogrid_var(self.config_options.elemcoords_var)[:][:, 1]

    @cached_property
    def elem_conn(self) -> np.ndarray:
        """Get the element connectivity for the unstructured domain."""
        return self.get_geogrid_var(self.config_options.elemconn_var)[:][:, 0]

    @cached_property
    def node_heights(self) -> np.ndarray:
        """Get the height grid for the unstructured domain nodes."""
        node_heights = self.get_geogrid_var(self.config_options.hgt_var)[:]

        if node_heights.shape[0] != self.ny_global:
            self.config_options.errMsg = (
                f"HGT_M dimension mismatch in: {self.config_options.geogrid}"
            )
            log_critical(self.config_options, self.mpi_config)
            raise Exception
        return node_heights

    @cached_property
    def elem_heights(self) -> np.ndarray:
        """Get the height grid for the unstructured domain elements."""
        elem_heights = self.get_var(self.geogrid_ds, self.config_options.hgt_elem_var)[
            :
        ]

        if elem_heights.shape[0] != len(self.elem_lons):
            self.config_options.errMsg = (
                f"HGT_M_ELEM dimension mismatch in: {self.config_options.geogrid}"
            )
            log_critical(self.config_options, self.mpi_config)
            raise Exception
        return elem_heights

    @cached_property
    def dx_elem(self) -> np.ndarray:
        """Calculate the dx distance in meters for the longitude variable for the unstructured domain elements."""
        dx = (
            np.diff(self.elem_lons)
            * 40075160
            * np.cos(self.elem_lats[0:-1] * np.pi / 180)
            / 360
        )
        return np.append(dx, dx[-1])

    @cached_property
    def dy_elem(self) -> np.ndarray:
        """Calculate the dy distance in meters for the latitude variable for the unstructured domain elements."""
        dy = np.diff(self.elem_lats) * 40008000 / 360
        return np.append(dy, dy[-1])

    @cached_property
    def dz_elem(self) -> np.ndarray:
        """Calculate the dz distance in meters for the height variable for the unstructured domain elements."""
        dz = np.diff(self.elem_heights)
        return np.append(dz, dz[-1])

    @cached_property
    def dx_node(self) -> np.ndarray:
        """Calculate the dx distance in meters for the longitude variable for the unstructured domain nodes."""
        dx = (
            np.diff(self.node_lons)
            * 40075160
            * np.cos(self.node_lats[0:-1] * np.pi / 180)
            / 360
        )
        return np.append(dx, dx[-1])

    @cached_property
    def dy_node(self) -> np.ndarray:
        """Calculate the dy distance in meters for the latitude variable for the unstructured domain nodes."""
        dy = np.diff(self.node_lats) * 40008000 / 360
        return np.append(dy, dy[-1])

    @cached_property
    def dz_node(self) -> np.ndarray:
        """Calculate the dz distance in meters for the height variable for the unstructured domain nodes."""
        dz = np.diff(self.node_heights)
        return np.append(dz, dz[-1])

    @cached_property
    def mesh_inds(self) -> np.ndarray:
        """Get the local mesh node indices for the unstructured domain."""
        return self.pet_node_inds

    @cached_property
    def mesh_inds_elem(self) -> np.ndarray:
        """Get the local mesh element indices for the unstructured domain."""
        return self.pet_element_inds

    @cached_property
    def nx_local(self) -> int:
        """Get the local x dimension size for this processor."""
        return len(self.esmf_grid.coords[0][1])

    @cached_property
    def ny_local(self) -> int:
        """Get the local y dimension size for this processor."""
        return len(self.esmf_grid.coords[0][1])

    @cached_property
    def nx_local_elem(self) -> int:
        """Get the local x dimension size for this processor."""
        return len(self.esmf_grid.coords[1][1])

    @cached_property
    def ny_local_elem(self) -> int:
        """Get the local y dimension size for this processor."""
        return len(self.esmf_grid.coords[1][1])
