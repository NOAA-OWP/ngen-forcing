import math
from time import time

import numpy as np

# For ESMF + shapely 2.x, shapely must be imported first, to avoid segfault "address not mapped to object" stemming from calls such as:
# /usr/local/esmf/lib/libO/Linux.gfortran.64.openmpi.default/libesmf_fullylinked.so(get_geom+0x36)
import shapely
import netCDF4
from scipy import spatial
from . import err_handler
from .. import esmf_utils
from .. import nc_utils


try:
    import esmpy as ESMF
except ImportError:
    import ESMF

import logging

from nextgen_forcings_ewts import MODULE_NAME

LOG = logging.getLogger(MODULE_NAME)


class GeoMetaWrfHydro:
    """Abstract class for handling information about the WRF-Hydro domain we are processing forcings too."""

    def __init__(self):
        """Initialize GeoMetaWrfHydro class variables."""
        self.nx_global = None
        self.ny_global = None
        self.nx_global_elem = None
        self.ny_global_elem = None
        self.dx_meters = None
        self.dy_meters = None
        self.nx_local = None
        self.ny_local = None
        self.nx_local_elem = None
        self.ny_local_elem = None
        self.x_lower_bound = None
        self.x_upper_bound = None
        self.y_lower_bound = None
        self.y_upper_bound = None
        self.latitude_grid = None
        self.longitude_grid = None
        self.element_ids = None
        self.element_ids_global = None
        self.latitude_grid_elem = None
        self.longitude_grid_elem = None
        self.lat_bounds = None
        self.lon_bounds = None
        self.mesh_inds = None
        self.mesh_inds_elem = None
        self.height = None
        self.height_elem = None
        self.sina_grid = None
        self.cosa_grid = None
        self.nodeCoords = None
        self.centerCoords = None
        self.inds = None
        self.slope = None
        self.slp_azi = None
        self.slope_elem = None
        self.slp_azi_elem = None
        self.esmf_grid = None
        self.esmf_lat = None
        self.esmf_lon = None
        self.crs_atts = None
        self.x_coord_atts = None
        self.x_coords = None
        self.y_coord_atts = None
        self.y_coords = None
        self.spatial_global_atts = None

    def get_processor_bounds(self, config_options):
        """Calculate the local grid boundaries for this processor.

        ESMF operates under the hood and the boundary values
        are calculated within the ESMF software.
        :return:
        """
        if config_options.grid_type == "gridded":
            self.x_lower_bound = self.esmf_grid.lower_bounds[ESMF.StaggerLoc.CENTER][1]
            self.x_upper_bound = self.esmf_grid.upper_bounds[ESMF.StaggerLoc.CENTER][1]
            self.y_lower_bound = self.esmf_grid.lower_bounds[ESMF.StaggerLoc.CENTER][0]
            self.y_upper_bound = self.esmf_grid.upper_bounds[ESMF.StaggerLoc.CENTER][0]
            self.nx_local = self.x_upper_bound - self.x_lower_bound
            self.ny_local = self.y_upper_bound - self.y_lower_bound
        elif config_options.grid_type == "unstructured":
            self.nx_local = len(self.esmf_grid.coords[0][1])
            self.ny_local = len(self.esmf_grid.coords[0][1])
            self.nx_local_elem = len(self.esmf_grid.coords[1][1])
            self.ny_local_elem = len(self.esmf_grid.coords[1][1])
            # LOG.debug("ESMF Mesh nx local node is " + str(self.nx_local))
            # LOG.debug("ESMF Mesh nx local elem is " + str(self.nx_local_elem))
        elif config_options.grid_type == "hydrofabric":
            self.nx_local = len(self.esmf_grid.coords[1][1])
            self.ny_local = len(self.esmf_grid.coords[1][1])
            # self.nx_local_poly = len(self.esmf_poly_coords)
            # self.ny_local_poly = len(self.esmf_poly_coords)
            # LOG.debug("ESMF Mesh nx local elem is " + str(self.nx_local))
            # LOG.debug("ESMF Mesh nx local poly is " + str(self.nx_local_poly))
        # LOG.debug("WRF-HYDRO LOCAL X BOUND 1 = " + str(self.x_lower_bound))
        # LOG.debug("WRF-HYDRO LOCAL X BOUND 2 = " + str(self.x_upper_bound))
        # LOG.debug("WRF-HYDRO LOCAL Y BOUND 1 = " + str(self.y_lower_bound))
        # LOG.debug("WRF-HYDRO LOCAL Y BOUND 2 = " + str(self.y_upper_bound))

    def initialize_destination_geo_gridded(self, config_options, mpi_config):
        """Initialize GeoMetaWrfHydro class variables.

        Initialization function to initialize ESMF through ESMPy,
        calculate the global parameters of the WRF-Hydro grid
        being processed to, along with the local parameters
        for this particular processor.
        :return:
        """
        # Open the geogrid file and extract necessary information
        # to create ESMF fields.
        if mpi_config.rank == 0:
            try:
                idTmp = netCDF4.Dataset(config_options.geogrid, "r")
            except Exception as e:
                config_options.errMsg = (
                    "Unable to open the WRF-Hydro geogrid file: "
                    + config_options.geogrid
                )
                raise Exception
            if idTmp.variables[config_options.lat_var].ndim == 3:
                try:
                    self.nx_global = idTmp.variables[config_options.lat_var].shape[2]
                except Exception as e:
                    config_options.errMsg = (
                        "Unable to extract X dimension size from latitude variable in: "
                        + config_options.geogrid
                    )
                    raise Exception

                try:
                    self.ny_global = idTmp.variables[config_options.lat_var].shape[1]
                except Exception as e:
                    config_options.errMsg = (
                        "Unable to extract Y dimension size from latitude in: "
                        + config_options.geogrid
                    )
                    raise Exception

                try:
                    self.dx_meters = idTmp.DX
                except Exception as e:
                    config_options.errMsg = (
                        "Unable to extract DX global attribute in: "
                        + config_options.geogrid
                    )
                    raise Exception

                try:
                    self.dy_meters = idTmp.DY
                except Exception as e:
                    config_options.errMsg = (
                        "Unable to extract DY global attribute in: "
                        + config_options.geogrid
                    )
                    raise Exception
            elif idTmp.variables[config_options.lat_var].ndim == 2:
                try:
                    self.nx_global = idTmp.variables[config_options.lat_var].shape[1]
                except Exception as e:
                    config_options.errMsg = (
                        "Unable to extract X dimension size from latitude variable in: "
                        + config_options.geogrid
                    )
                    raise Exception

                try:
                    self.ny_global = idTmp.variables[config_options.lat_var].shape[0]
                except Exception as e:
                    config_options.errMsg = (
                        "Unable to extract Y dimension size from latitude in: "
                        + config_options.geogrid
                    )
                    raise Exception

                try:
                    self.dx_meters = idTmp.variables[config_options.lon_var].dx
                except Exception as e:
                    config_options.errMsg = (
                        "Unable to extract DX global attribute in: "
                        + config_options.geogrid
                    )
                    raise Exception

                try:
                    self.dy_meters = idTmp.variables[config_options.lat_var].dy
                except Exception as e:
                    config_options.errMsg = (
                        "Unable to extract DY global attribute in: "
                        + config_options.geogrid
                    )
                    raise Exception

            else:
                try:
                    self.nx_global = idTmp.variables[config_options.lon_var].shape[0]
                except Exception as e:
                    config_options.errMsg = (
                        "Unable to extract X dimension size from longitude variable in: "
                        + config_options.geogrid
                    )
                    raise Exception

                try:
                    self.ny_global = idTmp.variables[config_options.lat_var].shape[0]
                except Exception as e:
                    config_options.errMsg = (
                        "Unable to extract Y dimension size from latitude in: "
                        + config_options.geogrid
                    )
                    raise Exception
                if config_options.input_forcings[0] != 23:
                    try:
                        self.dx_meters = idTmp.variables[config_options.lon_var].dx
                    except Exception as e:
                        config_options.errMsg = (
                            "Unable to extract dx metadata attribute in: "
                            + config_options.geogrid
                        )
                        raise Exception

                    try:
                        self.dy_meters = idTmp.variables[config_options.lat_var].dy
                    except Exception as e:
                        config_options.errMsg = (
                            "Unable to extract dy metadata attribute in: "
                            + config_options.geogrid
                        )
                        raise Exception
                else:
                    # Manually input the grid spacing since ERA5-Interim does not
                    # internally have this geospatial information within the netcdf file
                    self.dx_meters = 31000
                    self.dy_meters = 31000

        # mpi_config.comm.barrier()

        # Broadcast global dimensions to the other processors.
        self.nx_global = mpi_config.broadcast_parameter(
            self.nx_global, config_options, param_type=int
        )
        self.ny_global = mpi_config.broadcast_parameter(
            self.ny_global, config_options, param_type=int
        )
        self.dx_meters = mpi_config.broadcast_parameter(
            self.dx_meters, config_options, param_type=float
        )
        self.dy_meters = mpi_config.broadcast_parameter(
            self.dy_meters, config_options, param_type=float
        )

        # mpi_config.comm.barrier()

        try:
            self.esmf_grid = ESMF.Grid(
                np.array([self.ny_global, self.nx_global]),
                staggerloc=ESMF.StaggerLoc.CENTER,
                coord_sys=ESMF.CoordSys.SPH_DEG,
            )
        except Exception as e:
            config_options.errMsg = (
                "Unable to create ESMF grid for WRF-Hydro geogrid: "
                + config_options.geogrid
            )
            raise Exception

        # mpi_config.comm.barrier()

        self.esmf_lat = self.esmf_grid.get_coords(1)
        self.esmf_lon = self.esmf_grid.get_coords(0)

        # mpi_config.comm.barrier()

        # Obtain the local boundaries for this processor.
        self.get_processor_bounds(config_options)

        # Scatter global XLAT_M grid to processors..
        if mpi_config.rank == 0:
            if idTmp.variables[config_options.lat_var].ndim == 3:
                varTmp = idTmp.variables[config_options.lat_var][0, :, :]
            elif idTmp.variables[config_options.lat_var].ndim == 2:
                varTmp = idTmp.variables[config_options.lat_var][:, :]
            elif idTmp.variables[config_options.lat_var].ndim == 1:
                lat = idTmp.variables[config_options.lat_var][:]
                lon = idTmp.variables[config_options.lon_var][:]
                varTmp = np.meshgrid(lon, lat)[1]
                lat = None
                lon = None
            # Flag to grab entire array for AWS slicing
            if config_options.aws:
                self.lat_bounds = varTmp
        else:
            varTmp = None

        # mpi_config.comm.barrier()

        varSubTmp = mpi_config.scatter_array(self, varTmp, config_options)

        # mpi_config.comm.barrier()

        # Place the local lat/lon grid slices from the parent geogrid file into
        # the ESMF lat/lon grids.
        try:
            self.esmf_lat[:, :] = varSubTmp
            self.latitude_grid = varSubTmp
            varSubTmp = None
            varTmp = None
        except Exception as e:
            config_options.errMsg = (
                "Unable to subset latitude from geogrid file into ESMF object"
            )
            raise Exception

        # mpi_config.comm.barrier()

        # Scatter global XLONG_M grid to processors..
        if mpi_config.rank == 0:
            if idTmp.variables[config_options.lat_var].ndim == 3:
                varTmp = idTmp.variables[config_options.lon_var][0, :, :]
            elif idTmp.variables[config_options.lon_var].ndim == 2:
                varTmp = idTmp.variables[config_options.lon_var][:, :]
            elif idTmp.variables[config_options.lon_var].ndim == 1:
                lat = idTmp.variables[config_options.lat_var][:]
                lon = idTmp.variables[config_options.lon_var][:]
                varTmp = np.meshgrid(lon, lat)[0]
                lat = None
                lon = None
            # Flag to grab entire array for AWS slicing
            if config_options.aws:
                self.lon_bounds = varTmp
        else:
            varTmp = None

        # mpi_config.comm.barrier()

        varSubTmp = mpi_config.scatter_array(self, varTmp, config_options)

        # mpi_config.comm.barrier()

        try:
            self.esmf_lon[:, :] = varSubTmp
            self.longitude_grid = varSubTmp
            varSubTmp = None
            varTmp = None
        except Exception as e:
            config_options.errMsg = (
                "Unable to subset longitude from geogrid file into ESMF object"
            )
            raise Exception

        # mpi_config.comm.barrier()

        if (
            config_options.cosalpha_var is not None
            and config_options.sinalpha_var is not None
        ):
            # Scatter the COSALPHA,SINALPHA grids to the processors.
            if mpi_config.rank == 0:
                if idTmp.variables[config_options.cosalpha_var].ndim == 3:
                    varTmp = idTmp.variables[config_options.cosalpha_var][0, :, :]
                else:
                    varTmp = idTmp.variables[config_options.cosalpha_var][:, :]

            else:
                varTmp = None
            # mpi_config.comm.barrier()

            varSubTmp = mpi_config.scatter_array(self, varTmp, config_options)
            # mpi_config.comm.barrier()

            self.cosa_grid = varSubTmp[:, :]
            varSubTmp = None
            varTmp = None

            if mpi_config.rank == 0:
                if idTmp.variables[config_options.sinalpha_var].ndim == 3:
                    varTmp = idTmp.variables[config_options.sinalpha_var][0, :, :]
                else:
                    varTmp = idTmp.variables[config_options.sinalpha_var][:, :]
            else:
                varTmp = None
            # mpi_config.comm.barrier()

            varSubTmp = mpi_config.scatter_array(self, varTmp, config_options)
            # mpi_config.comm.barrier()
            self.sina_grid = varSubTmp[:, :]
            varSubTmp = None
            varTmp = None

        if config_options.hgt_var is not None:
            # Read in a scatter the WRF-Hydro elevation, which is used for downscaling
            # purposes.
            if mpi_config.rank == 0:
                if idTmp.variables[config_options.hgt_var].ndim == 3:
                    varTmp = idTmp.variables[config_options.hgt_var][0, :, :]
                else:
                    varTmp = idTmp.variables[config_options.hgt_var][:, :]
            else:
                varTmp = None
            # mpi_config.comm.barrier()

            varSubTmp = mpi_config.scatter_array(self, varTmp, config_options)
            # mpi_config.comm.barrier()
            self.height = varSubTmp
            varSubTmp = None
            varTmp = None

        if (
            config_options.cosalpha_var is not None
            and config_options.sinalpha_var is not None
        ):
            # Calculate the slope from the domain using elevation on the WRF-Hydro domain. This will
            # be used for downscaling purposes.
            if mpi_config.rank == 0:
                try:
                    slopeTmp, slp_azi_tmp = self.calc_slope(idTmp, config_options)
                except Exception:
                    raise Exception
            else:
                slopeTmp = None
                slp_azi_tmp = None
            # mpi_config.comm.barrier()

            slopeSubTmp = mpi_config.scatter_array(self, slopeTmp, config_options)
            self.slope = slopeSubTmp[:, :]
            slopeSubTmp = None

            slp_azi_sub = mpi_config.scatter_array(self, slp_azi_tmp, config_options)
            self.slp_azi = slp_azi_sub[:, :]
            slp_azi_tmp = None

        elif (
            config_options.slope_var is not None
            and config_options.slope_azimuth_var is not None
        ):
            if mpi_config.rank == 0:
                if idTmp.variables[config_options.slope_var].ndim == 3:
                    varTmp = idTmp.variables[config_options.slope_var][0, :, :]
                else:
                    varTmp = idTmp.variables[config_options.slope_var][:, :]
            else:
                varTmp = None

            slopeSubTmp = mpi_config.scatter_array(self, varTmp, config_options)
            self.slope = slopeSubTmp
            varTmp = None

            if mpi_config.rank == 0:
                if idTmp.variables[config_options.slope_azimuth_var].ndim == 3:
                    varTmp = idTmp.variables[config_options.slope_azimuth_var][0, :, :]
                else:
                    varTmp = idTmp.variables[config_options.slope_azimuth_var][:, :]
            else:
                varTmp = None

            slp_azi_sub = mpi_config.scatter_array(self, varTmp, config_options)
            self.slp_azi = slp_azi_sub[:, :]
            varTmp = None

        elif config_options.hgt_var is not None:
            # Calculate the slope from the domain using elevation of the gridded model and other approximations
            if mpi_config.rank == 0:
                try:
                    slopeTmp, slp_azi_tmp = self.calc_slope_gridded(
                        idTmp, config_options
                    )
                except Exception:
                    raise Exception
            else:
                slopeTmp = None
                slp_azi_tmp = None
            # mpi_config.comm.barrier()

            slopeSubTmp = mpi_config.scatter_array(self, slopeTmp, config_options)
            self.slope = slopeSubTmp[:, :]
            slopeSubTmp = None

            slp_azi_sub = mpi_config.scatter_array(self, slp_azi_tmp, config_options)
            self.slp_azi = slp_azi_sub[:, :]
            slp_azi_tmp = None

        if mpi_config.rank == 0:
            # Close the geogrid file
            try:
                idTmp.close()
            except Exception as e:
                config_options.errMsg = (
                    "Unable to close geogrid file: " + config_options.geogrid
                )
                raise Exception

        # Reset temporary variables to free up memory
        slopeTmp = None
        slp_azi_tmp = None
        varTmp = None

    def initialize_geospatial_metadata(self, config_options, mpi_config):
        """Initialize GeoMetaWrfHydro class variables.

        Function that will read in crs/x/y geospatial metadata and coordinates
        from the optional geospatial metadata file IF it was specified by the user in
        the configuration file.
        :param config_options:
        :return:
        """
        # We will only read information on processor 0. This data is not necessary for the
        # other processors, and is only used in the output routines.
        if mpi_config.rank == 0:
            # Open the geospatial metadata file.
            try:
                idTmp = netCDF4.Dataset(config_options.spatial_meta, "r")
            except Exception as e:
                config_options.errMsg = (
                    "Unable to open spatial metadata file: "
                    + config_options.spatial_meta
                )
                raise Exception

            # Make sure the expected variables are present in the file.
            if "crs" not in idTmp.variables.keys():
                config_options.errMsg = (
                    "Unable to locate crs variable in: " + config_options.spatial_meta
                )
                raise Exception
            if "x" not in idTmp.variables.keys():
                config_options.errMsg = (
                    "Unable to locate x variable in: " + config_options.spatial_meta
                )
                raise Exception
            if "y" not in idTmp.variables.keys():
                config_options.errMsg = (
                    "Unable to locate y variable in: " + config_options.spatial_meta
                )
                raise Exception
            # Extract names of variable attributes from each of the input geospatial variables. These
            # can change, so we are making this as flexible as possible to accomodate future changes.
            try:
                crs_att_names = idTmp.variables["crs"].ncattrs()
            except Exception as e:
                config_options.errMsg = (
                    "Unable to extract crs attribute names from: "
                    + config_options.spatial_meta
                )
                raise Exception
            try:
                x_coord_att_names = idTmp.variables["x"].ncattrs()
            except Exception as e:
                config_options.errMsg = (
                    "Unable to extract x attribute names from: "
                    + config_options.spatial_meta
                )
                raise Exception
            try:
                y_coord_att_names = idTmp.variables["y"].ncattrs()
            except Exception as e:
                config_options.errMsg = (
                    "Unable to extract y attribute names from: "
                    + config_options.spatial_meta
                )
                raise Exception
            # Extract attribute values
            try:
                self.x_coord_atts = {
                    item: idTmp.variables["x"].getncattr(item)
                    for item in x_coord_att_names
                }
            except Exception as e:
                config_options.errMsg = (
                    "Unable to extract x coordinate attributes from: "
                    + config_options.spatial_meta
                )
                raise Exception
            try:
                self.y_coord_atts = {
                    item: idTmp.variables["y"].getncattr(item)
                    for item in y_coord_att_names
                }
            except Exception as e:
                config_options.errMsg = (
                    "Unable to extract y coordinate attributes from: "
                    + config_options.spatial_meta
                )
                raise Exception
            try:
                self.crs_atts = {
                    item: idTmp.variables["crs"].getncattr(item)
                    for item in crs_att_names
                }
            except Exception as e:
                config_options.errMsg = (
                    "Unable to extract crs coordinate attributes from: "
                    + config_options.spatial_meta
                )
                raise Exception

            # Extract global attributes
            try:
                global_att_names = idTmp.ncattrs()
            except Exception as e:
                config_options.errMsg = (
                    "Unable to extract global attribute names from: "
                    + config_options.spatial_meta
                )
                raise Exception
            try:
                self.spatial_global_atts = {
                    item: idTmp.getncattr(item) for item in global_att_names
                }
            except Exception as e:
                config_options.errMsg = (
                    "Unable to extract global attributes from: "
                    + config_options.spatial_meta
                )
                raise Exception

            # Extract x/y coordinate values
            if len(idTmp.variables["x"].shape) == 1:
                try:
                    self.x_coords = idTmp.variables["x"][:].data
                except Exception as e:
                    config_options.errMsg = (
                        "Unable to extract x coordinate values from: "
                        + config_options.spatial_meta
                    )
                    raise Exception
                try:
                    self.y_coords = idTmp.variables["y"][:].data
                except Exception as e:
                    config_options.errMsg = (
                        "Unable to extract y coordinate values from: "
                        + config_options.spatial_meta
                    )
                    raise Exception
                # Check to see if the Y coordinates are North-South. If so, flip them.
                if self.y_coords[1] < self.y_coords[0]:
                    self.y_coords[:] = np.flip(self.y_coords[:], axis=0)

            if len(idTmp.variables["x"].shape) == 2:
                try:
                    self.x_coords = idTmp.variables["x"][:, :].data
                except Exception as e:
                    config_options.errMsg = (
                        "Unable to extract x coordinate values from: "
                        + config_options.spatial_meta
                    )
                    raise Exception
                try:
                    self.y_coords = idTmp.variables["y"][:, :].data
                except Exception as e:
                    config_options.errMsg = (
                        "Unable to extract y coordinate values from: "
                        + config_options.spatial_meta
                    )
                    raise Exception
                # Check to see if the Y coordinates are North-South. If so, flip them.
                if self.y_coords[1, 0] > self.y_coords[0, 0]:
                    self.y_coords[:, :] = np.flipud(self.y_coords[:, :])

            # Close the geospatial metadata file.
            try:
                idTmp.close()
            except Exception as e:
                config_options.errMsg = (
                    "Unable to close spatial metadata file: "
                    + config_options.spatial_meta
                )
                raise Exception

        # mpi_config.comm.barrier()

    def calc_slope(self, idTmp, config_options):
        """Calculate slope grids needed for incoming shortwave radiation downscaling.

        Function to calculate slope grids needed for incoming shortwave radiation downscaling
        later during the program.
        :param idTmp:
        :param config_options:
        :return:
        """
        # First extract the sina,cosa, and elevation variables from the geogrid file.
        try:
            sinaGrid = idTmp.variables[config_options.sinalpha_var][0, :, :]
        except Exception as e:
            config_options.errMsg = (
                "Unable to extract SINALPHA from: " + config_options.geogrid
            )
            raise

        try:
            cosaGrid = idTmp.variables[config_options.cosalpha_var][0, :, :]
        except Exception as e:
            config_options.errMsg = (
                "Unable to extract COSALPHA from: " + config_options.geogrid
            )
            raise

        try:
            heightDest = idTmp.variables[config_options.hgt_var][0, :, :]
        except Exception as e:
            config_options.errMsg = (
                "Unable to extract HGT_M from: " + config_options.geogrid
            )
            raise

        # Ensure cosa/sina are correct dimensions
        if sinaGrid.shape[0] != self.ny_global or sinaGrid.shape[1] != self.nx_global:
            config_options.errMsg = (
                "SINALPHA dimensions mismatch in: " + config_options.geogrid
            )
            raise Exception
        if cosaGrid.shape[0] != self.ny_global or cosaGrid.shape[1] != self.nx_global:
            config_options.errMsg = (
                "COSALPHA dimensions mismatch in: " + config_options.geogrid
            )
            raise Exception
        if (
            heightDest.shape[0] != self.ny_global
            or heightDest.shape[1] != self.nx_global
        ):
            config_options.errMsg = (
                "HGT_M dimension mismatch in: " + config_options.geogrid
            )
            raise Exception

        # Establish constants
        rdx = 1.0 / self.dx_meters
        rdy = 1.0 / self.dy_meters
        msftx = 1.0
        msfty = 1.0

        slopeOut = np.empty([self.ny_global, self.nx_global], np.float32)
        toposlpx = np.empty([self.ny_global, self.nx_global], np.float32)
        toposlpy = np.empty([self.ny_global, self.nx_global], np.float32)
        slp_azi = np.empty([self.ny_global, self.nx_global], np.float32)
        ipDiff = np.empty([self.ny_global, self.nx_global], np.int32)
        jpDiff = np.empty([self.ny_global, self.nx_global], np.int32)
        hx = np.empty([self.ny_global, self.nx_global], np.float32)
        hy = np.empty([self.ny_global, self.nx_global], np.float32)

        # Create index arrays that will be used to calculate slope.
        xTmp = np.arange(self.nx_global)
        yTmp = np.arange(self.ny_global)
        xGrid = np.tile(xTmp[:], (self.ny_global, 1))
        yGrid = np.repeat(yTmp[:, np.newaxis], self.nx_global, axis=1)
        indOrig = np.where(heightDest == heightDest)
        indIp1 = ((indOrig[0]), (indOrig[1] + 1))
        indIm1 = ((indOrig[0]), (indOrig[1] - 1))
        indJp1 = ((indOrig[0] + 1), (indOrig[1]))
        indJm1 = ((indOrig[0] - 1), (indOrig[1]))
        indIp1[1][np.where(indIp1[1] >= self.nx_global)] = self.nx_global - 1
        indJp1[0][np.where(indJp1[0] >= self.ny_global)] = self.ny_global - 1
        indIm1[1][np.where(indIm1[1] < 0)] = 0
        indJm1[0][np.where(indJm1[0] < 0)] = 0

        ipDiff[indOrig] = xGrid[indIp1] - xGrid[indIm1]
        jpDiff[indOrig] = yGrid[indJp1] - yGrid[indJm1]

        toposlpx[indOrig] = (
            (heightDest[indIp1] - heightDest[indIm1]) * msftx * rdx
        ) / ipDiff[indOrig]
        toposlpy[indOrig] = (
            (heightDest[indJp1] - heightDest[indJm1]) * msfty * rdy
        ) / jpDiff[indOrig]
        hx[indOrig] = toposlpx[indOrig]
        hy[indOrig] = toposlpy[indOrig]
        slopeOut[indOrig] = np.arctan((hx[indOrig] ** 2 + hy[indOrig] ** 2) ** 0.5)
        slopeOut[np.where(slopeOut < 1e-4)] = 0.0
        slp_azi[np.where(slopeOut < 1e-4)] = 0.0
        indValidTmp = np.where(slopeOut >= 1e-4)
        slp_azi[indValidTmp] = np.arctan2(hx[indValidTmp], hy[indValidTmp]) + math.pi
        indValidTmp = np.where(cosaGrid >= 0.0)
        slp_azi[indValidTmp] = slp_azi[indValidTmp] - np.arcsin(sinaGrid[indValidTmp])
        indValidTmp = np.where(cosaGrid < 0.0)
        slp_azi[indValidTmp] = slp_azi[indValidTmp] - (
            math.pi - np.arcsin(sinaGrid[indValidTmp])
        )

        # Reset temporary arrays to None to free up memory
        toposlpx = None
        toposlpy = None
        heightDest = None
        sinaGrid = None
        cosaGrid = None
        indValidTmp = None
        xTmp = None
        yTmp = None
        xGrid = None
        ipDiff = None
        jpDiff = None
        indOrig = None
        indJm1 = None
        indJp1 = None
        indIm1 = None
        indIp1 = None
        hx = None
        hy = None

        return slopeOut, slp_azi

    def calc_slope_gridded(self, idTmp, config_options):
        """Calculate slope grids needed for incoming shortwave radiation downscaling.

        Function to calculate slope grids needed for incoming shortwave radiation downscaling
        later during the program. This calculates the slopes for grid cells
        :param idTmp:
        :param config_options:
        :return:
        """
        idTmp = netCDF4.Dataset(config_options.geogrid, "r")

        try:
            lons = idTmp.variables[config_options.lon_var][:]
            lats = idTmp.variables[config_options.lat_var][:]
        except Exception as e:
            config_options.errMsg = (
                "Unable to extract gridded coordinates in " + config_options.geogrid
            )
            raise Exception
        try:
            dx = np.empty(
                (
                    idTmp.variables[config_options.lat_var].shape[0],
                    idTmp.variables[config_options.lon_var].shape[0],
                ),
                dtype=float,
            )
            dy = np.empty(
                (
                    idTmp.variables[config_options.lat_var].shape[0],
                    idTmp.variables[config_options.lon_var].shape[0],
                ),
                dtype=float,
            )
            dx[:] = idTmp.variables[config_options.lon_var].dx
            dy[:] = idTmp.variables[config_options.lat_var].dy
        except Exception as e:
            config_options.errMsg = (
                "Unable to extract dx and dy distances in " + config_options.geogrid
            )
            raise Exception
        try:
            heights = idTmp.variables[config_options.hgt_var][:]
        except Exception as e:
            config_options.errMsg = (
                "Unable to extract heights of grid cells in " + config_options.geogrid
            )
            raise Exception

        idTmp.close()

        # calculate grid coordinates dx distances in meters
        # based on general geospatial formula approximations
        # on a spherical grid
        dz_init = np.diff(heights, axis=0)
        dz = np.empty(dx.shape, dtype=float)
        dz[0 : dz_init.shape[0], 0 : dz_init.shape[1]] = dz_init
        dz[dz_init.shape[0] :, :] = dz_init[-1, :]

        slope = dz / np.sqrt((dx**2) + (dy**2))
        slp_azi = (180 / np.pi) * np.arctan(dx / dy)

        # Reset temporary arrays to None to free up memory
        lons = None
        lats = None
        heights = None
        dx = None
        dy = None
        dz = None

        return slope, slp_azi

    def initialize_destination_geo_unstructured(self, config_options, mpi_config):
        """Initialize GeoMetaWrfHydro class variables.

        Initialization function to initialize ESMF through ESMPy,
        calculate the global parameters of the WRF-Hydro grid
        being processed to, along with the local parameters
        for this particular processor.
        :return:
        """
        # Open the geogrid file and extract necessary information
        # to create ESMF fields.
        if mpi_config.rank == 0:
            try:
                idTmp = netCDF4.Dataset(config_options.geogrid, "r")
            except Exception as e:
                config_options.errMsg = (
                    "Unable to open the unstructured mesh file: "
                    + config_options.geogrid
                )
                raise Exception

            try:
                self.nx_global = idTmp.variables[config_options.nodecoords_var].shape[0]
            except Exception as e:
                config_options.errMsg = (
                    "Unable to extract X dimension size in " + config_options.geogrid
                )
                raise Exception

            try:
                self.ny_global = idTmp.variables[config_options.nodecoords_var].shape[0]
            except Exception as e:
                config_options.errMsg = (
                    "Unable to extract Y dimension size in " + config_options.geogrid
                )
                raise Exception

            try:
                self.nx_global_elem = idTmp.variables[
                    config_options.elemcoords_var
                ].shape[0]
            except Exception as e:
                config_options.errMsg = (
                    "Unable to extract X dimension size in " + config_options.geogrid
                )
                raise Exception

            try:
                self.ny_global_elem = idTmp.variables[
                    config_options.elemcoords_var
                ].shape[0]
            except Exception as e:
                config_options.errMsg = (
                    "Unable to extract Y dimension size in " + config_options.geogrid
                )
                raise Exception

            # Flag to grab entire array for AWS slicing
            if config_options.aws:
                self.lat_bounds = idTmp.variables[config_options.nodecoords_var][:][
                    :, 1
                ]
                self.lon_bounds = idTmp.variables[config_options.nodecoords_var][:][
                    :, 0
                ]

        # mpi_config.comm.barrier()

        # Broadcast global dimensions to the other processors.
        self.nx_global = mpi_config.broadcast_parameter(
            self.nx_global, config_options, param_type=int
        )
        self.ny_global = mpi_config.broadcast_parameter(
            self.ny_global, config_options, param_type=int
        )
        self.nx_global_elem = mpi_config.broadcast_parameter(
            self.nx_global_elem, config_options, param_type=int
        )
        self.ny_global_elem = mpi_config.broadcast_parameter(
            self.ny_global_elem, config_options, param_type=int
        )

        # mpi_config.comm.barrier()

        if mpi_config.rank == 0:
            # Close the geogrid file
            try:
                idTmp.close()
            except Exception as e:
                config_options.errMsg = (
                    "Unable to close geogrid Mesh file: " + config_options.geogrid
                )
                raise Exception

        try:
            # Removed argument coord_sys=ESMF.CoordSys.SPH_DEG since we are always reading from a file
            # From ESMF documentation
            # If you create a mesh from a file (like NetCDF/ESMF-Mesh), coord_sys is ignored. The mesh’s coordinate system should be embedded in the file or inferred.
            self.esmf_grid = ESMF.Mesh(
                filename=config_options.geogrid, filetype=ESMF.FileFormat.ESMFMESH
            )
        except Exception as e:
            config_options.errMsg = (
                "Unable to create ESMF Mesh from geogrid file: "
                + config_options.geogrid
            )
            raise Exception

        # mpi_config.comm.barrier()

        # Obtain the local boundaries for this processor.
        self.get_processor_bounds(config_options)

        # Place the local lat/lon grid slices from the parent geogrid file into
        # the ESMF lat/lon grids that have already been seperated by processors.
        try:
            self.latitude_grid = self.esmf_grid.coords[0][1]
            self.latitude_grid_elem = self.esmf_grid.coords[1][1]
            varSubTmp = None
            varTmp = None
        except Exception as e:
            config_options.errMsg = (
                "Unable to subset node latitudes from ESMF Mesh object"
            )
            raise Exception
        try:
            self.longitude_grid = self.esmf_grid.coords[0][0]
            self.longitude_grid_elem = self.esmf_grid.coords[1][0]
            varSubTmp = None
            varTmp = None
        except Exception as e:
            config_options.errMsg = (
                "Unable to subset XLONG_M from geogrid file into ESMF Mesh object"
            )
            raise Exception

        idTmp = netCDF4.Dataset(config_options.geogrid, "r")

        # Get lat and lon global variables for pet extraction of indices
        nodecoords_global = idTmp.variables[config_options.nodecoords_var][:].data
        elementcoords_global = idTmp.variables[config_options.elemcoords_var][:].data

        # Find the corresponding local indices to slice global heights and slope
        # variables that are based on the partitioning on the unstructured mesh
        pet_nodecoords = np.empty((len(self.latitude_grid), 2), dtype=float)
        pet_elementcoords = np.empty((len(self.latitude_grid_elem), 2), dtype=float)
        pet_nodecoords[:, 0] = self.longitude_grid
        pet_nodecoords[:, 1] = self.latitude_grid
        pet_elementcoords[:, 0] = self.longitude_grid_elem
        pet_elementcoords[:, 1] = self.latitude_grid_elem

        distance, pet_node_inds = spatial.KDTree(nodecoords_global).query(
            pet_nodecoords
        )
        distance, pet_element_inds = spatial.KDTree(elementcoords_global).query(
            pet_elementcoords
        )

        # reset variables to free up memory
        nodecoords_global = None
        elementcoords_global = None
        pet_nodecoords = None
        pet_elementcoords = None
        distance = None

        # Not accepting cosalpha and sinalpha at this time for unstructured meshes, only
        # accepting the pre-calculated slope and slope azmiuth variables if available,
        # otherwise calculate slope from height estimates
        # if(config_options.cosalpha_var != None and config_options.sinalpha_var != None):
        # self.cosa_grid = idTmp.variables[config_options.cosalpha_var][:].data[pet_node_inds]
        # self.sina_grid = idTmp.variables[config_options.sinalpha_var][:].data[pet_node_inds]
        # slopeTmp, slp_azi_tmp = self.calc_slope(idTmp,config_options)
        # self.slope = slope_node_Tmp[pet_node_inds]
        # self.slp_azi = slp_azi_node_tmp[pet_node_inds]
        if (
            config_options.slope_var is not None
            and config_options.slp_azi_var is not None
        ):
            self.slope = idTmp.variables[config_options.slope_var][:].data[
                pet_node_inds
            ]
            self.slp_azi = idTmp.variables[config_options.slope_azimuth_var][:].data[
                pet_node_inds
            ]
            self.slope_elem = idTmp.variables[config_options.slope_var_elem][:].data[
                pet_element_inds
            ]
            self.slp_azi_elem = idTmp.variables[config_options.slope_azimuth_var_elem][
                :
            ].data[pet_element_inds]

            # Read in a scatter the mesh node elevation, which is used for downscaling purposes
            self.height = idTmp.variables[config_options.hgt_var][:].data[pet_node_inds]
            # Read in a scatter the mesh element elevation, which is used for downscaling purposes.
            self.height_elem = idTmp.variables[config_options.hgt_elem_var][:].data[
                pet_element_inds
            ]

        elif config_options.hgt_var is not None:
            # Read in a scatter the mesh node elevation, which is used for downscaling purposes
            self.height = idTmp.variables[config_options.hgt_var][:].data[pet_node_inds]

            # Read in a scatter the mesh element elevation, which is used for downscaling purposes.
            self.height_elem = idTmp.variables[config_options.hgt_elem_var][:].data[
                pet_element_inds
            ]

            # Calculate the slope from the domain using elevation on the WRF-Hydro domain. This will
            # be used for downscaling purposes.
            slope_node_Tmp, slp_azi_node_tmp, slope_elem_Tmp, slp_azi_elem_tmp = (
                self.calc_slope_unstructured(idTmp, config_options)
            )

            self.slope = slope_node_Tmp[pet_node_inds]
            slope_node_Tmp = None

            self.slp_azi = slp_azi_node_tmp[pet_node_inds]
            slp_azi__node_tmp = None

            self.slope_elem = slope_elem_Tmp[pet_element_inds]
            slope_elem_Tmp = None

            self.slp_azi_elem = slp_azi_elem_tmp[pet_element_inds]
            slp_azi_elem_tmp = None

        # save indices where mesh was partition for future scatter functions
        self.mesh_inds = pet_node_inds
        self.mesh_inds_elem = pet_element_inds

        # reset variables to free up memory
        pet_node_inds = None
        pet_element_inds = None

    def calc_slope_unstructured(self, idTmp, config_options):
        """Calculate slope grids needed for incoming shortwave radiation downscaling.

        Function to calculate slope grids needed for incoming shortwave radiation downscaling
        later during the program. This calculates the slopes for both nodes and elements
        :param idTmp:
        :param config_options:
        :return:
        """
        idTmp = netCDF4.Dataset(config_options.geogrid, "r")

        try:
            node_lons = idTmp.variables[config_options.nodecoords_var][:][:, 0]
            node_lats = idTmp.variables[config_options.nodecoords_var][:][:, 1]
        except Exception as e:
            config_options.errMsg = (
                "Unable to extract node coordinates in " + config_options.geogrid
            )
            raise Exception
        try:
            elem_lons = idTmp.variables[config_options.elemcoords_var][:][:, 0]
            elem_lats = idTmp.variables[config_options.elemcoords_var][:][:, 1]
        except Exception as e:
            config_options.errMsg = (
                "Unable to extract element coordinates in " + config_options.geogrid
            )
            raise Exception
        try:
            elem_conn = idTmp.variables[config_options.elemconn_var][:][:, 0]
        except Exception as e:
            config_options.errMsg = (
                "Unable to extract element connectivity in " + config_options.geogrid
            )
            raise Exception
        try:
            node_heights = idTmp.variables[config_options.hgt_var][:]
        except Exception as e:
            config_options.errMsg = (
                "Unable to extract HGT_M from: " + config_options.geogrid
            )
            raise

        if node_heights.shape[0] != self.ny_global:
            config_options.errMsg = (
                "HGT_M dimension mismatch in: " + config_options.geogrid
            )
            raise Exception

        try:
            elem_heights = idTmp.variables[config_options.hgt_elem_var][:]
        except Exception as e:
            config_options.errMsg = (
                "Unable to extract HGT_M_ELEM from: " + config_options.geogrid
            )
            raise

        if elem_heights.shape[0] != len(elem_lons):
            config_options.errMsg = (
                "HGT_M_ELEM dimension mismatch in: " + config_options.geogrid
            )
            raise Exception

        idTmp.close()

        # calculate node coordinate distances in meters
        # based on general geospatial formula approximations
        # on a spherical grid
        dx = np.diff(node_lons) * 40075160 * np.cos(node_lats[0:-1] * np.pi / 180) / 360
        dx = np.append(dx, dx[-1])
        dy = np.diff(node_lats) * 40008000 / 360
        dy = np.append(dy, dy[-1])
        dz = np.diff(node_heights)
        dz = np.append(dz, dz[-1])

        slope_nodes = dz / np.sqrt((dx**2) + (dy**2))
        slp_azi_nodes = (180 / np.pi) * np.arctan(dx / dy)

        # calculate element coordinate distances in meters
        # based on general geospatial formula approximations
        # on a spherical grid
        dx = np.diff(elem_lons) * 40075160 * np.cos(elem_lats[0:-1] * np.pi / 180) / 360
        dx = np.append(dx, dx[-1])
        dy = np.diff(elem_lats) * 40008000 / 360
        dy = np.append(dy, dy[-1])
        dz = np.diff(elem_heights)
        dz = np.append(dz, dz[-1])

        slope_elem = dz / np.sqrt((dx**2) + (dy**2))
        slp_azi_elem = (180 / np.pi) * np.arctan(dx / dy)

        # Reset temporary arrays to None to free up memory
        node_lons = None
        node_lats = None
        elem_lons = None
        elem_lats = None
        node_heights = None
        elem_heights = None
        dx = None
        dy = None
        dz = None

        return slope_nodes, slp_azi_nodes, slope_elem, slp_azi_elem

    def initialize_destination_geo_hydrofabric(self, config_options, mpi_config):
        """Initialize GeoMetaWrfHydro class variables.

        Initialization function to initialize ESMF through ESMPy,
        calculate the global parameters of the WRF-Hydro grid
        being processed to, along with the local parameters
        for this particular processor.
        :return:
        """

        if config_options.geogrid is not None:
            # Phase 1: Rank 0 extracts all needed global data
            if mpi_config.rank == 0:
                try:
                    idTmp = nc_utils.nc_Dataset_retry(
                        mpi_config,
                        config_options,
                        err_handler,
                        config_options.geogrid,
                        "r",
                    )

                    # Extract everything we need with retries
                    tmp_vars = idTmp.variables

                    if config_options.aws:
                        nodecoords_data = nc_utils.nc_read_var_retry(
                            mpi_config,
                            config_options,
                            err_handler,
                            tmp_vars[config_options.nodecoords_var],
                        )
                        self.lat_bounds = nodecoords_data[:, 1]
                        self.lon_bounds = nodecoords_data[:, 0]

                    # Store these for later broadcast/scatter
                    elementcoords_global = nc_utils.nc_read_var_retry(
                        mpi_config,
                        config_options,
                        err_handler,
                        tmp_vars[config_options.elemcoords_var],
                    )

                    self.nx_global = elementcoords_global.shape[0]
                    self.ny_global = self.nx_global

                    element_ids_global = nc_utils.nc_read_var_retry(
                        mpi_config,
                        config_options,
                        err_handler,
                        tmp_vars[config_options.element_id_var],
                    )

                    heights_global = None
                    if config_options.hgt_var is not None:
                        heights_global = nc_utils.nc_read_var_retry(
                            mpi_config,
                            config_options,
                            err_handler,
                            tmp_vars[config_options.hgt_var],
                        )
                    slopes_global = None
                    slp_azi_global = None
                    if config_options.slope_var is not None:
                        slopes_global = nc_utils.nc_read_var_retry(
                            mpi_config,
                            config_options,
                            err_handler,
                            tmp_vars[config_options.slope_var],
                        )
                    if config_options.slope_azimuth_var is not None:
                        slp_azi_global = nc_utils.nc_read_var_retry(
                            mpi_config,
                            config_options,
                            err_handler,
                            tmp_vars[config_options.slope_azimuth_var],
                        )

                except Exception as e:
                    LOG.critical(
                        f"Failed to open mesh file: {config_options.geogrid} "
                        f"due to {str(e)}"
                    )
                    raise
                finally:
                    idTmp.close()
            else:
                elementcoords_global = None
                element_ids_global = None
                heights_global = None
                slopes_global = None
                slp_azi_global = None

            # Broadcast dimensions
            self.nx_global = mpi_config.broadcast_parameter(
                self.nx_global, config_options, param_type=int
            )
            self.ny_global = mpi_config.broadcast_parameter(
                self.ny_global, config_options, param_type=int
            )

            mpi_config.comm.barrier()

            # Phase 2: Create ESMF Mesh (collective operation with retry)
            try:
                self.esmf_grid = esmf_utils.esmf_mesh_retry(
                    mpi_config,
                    config_options,
                    err_handler,
                    filename=config_options.geogrid,
                    filetype=ESMF.FileFormat.ESMFMESH,
                )
            except Exception as e:
                LOG.critical(
                    f"Unable to create ESMF Mesh: {config_options.geogrid} "
                    f"due to {str(e)}"
                )
                raise

            # Get processor bounds
            self.get_processor_bounds(config_options)

            # Extract local coordinates from ESMF mesh
            self.latitude_grid = self.esmf_grid.coords[1][1]
            self.longitude_grid = self.esmf_grid.coords[1][0]

            # Phase 3: Broadcast global arrays and compute local indices
            elementcoords_global = mpi_config.comm.bcast(elementcoords_global, root=0)
            element_ids_global = mpi_config.comm.bcast(element_ids_global, root=0)

            # Each rank computes its own local indices
            pet_elementcoords = np.column_stack(
                [self.longitude_grid, self.latitude_grid]
            )
            tree = spatial.KDTree(elementcoords_global)
            _, pet_element_inds = tree.query(pet_elementcoords)

            self.element_ids = element_ids_global[pet_element_inds]
            self.element_ids_global = element_ids_global

            # Broadcast and extract height/slope data
            if config_options.hgt_var is not None:
                heights_global = mpi_config.comm.bcast(heights_global, root=0)
                self.height = heights_global[pet_element_inds]

            if config_options.slope_var is not None:
                slopes_global = mpi_config.comm.bcast(slopes_global, root=0)
                slp_azi_global = mpi_config.comm.bcast(slp_azi_global, root=0)
                self.slope = slopes_global[pet_element_inds]
                self.slp_azi = slp_azi_global[pet_element_inds]

            self.mesh_inds = pet_element_inds
