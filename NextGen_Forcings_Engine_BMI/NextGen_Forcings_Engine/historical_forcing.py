"""Module for processing AORC and NWM data."""

import datetime
import gc
import os
import typing
from contextlib import contextmanager
from datetime import timedelta
from functools import cached_property
from time import perf_counter, sleep

import ewts
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import obstore
import pandas as pd
import s3fs
import xarray as xr
import zarr
from dotenv import find_dotenv, load_dotenv
from pyproj import CRS
from zarr.storage import ObjectStore

from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.config import (
    ConfigOptions,
)
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.parallel import MpiConfig
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.general_utils import rand_str

LOG = ewts.get_logger(ewts.FORCING_ID)

zarr.config.set({"async.concurrency": 100})


class BaseProcessor:
    """Base class for data processors."""

    def __init__(
        self,
        config_options: ConfigOptions,
        mpi_config: MpiConfig,
        wrf_hydro_geo_meta: dict,
    ):
        """Initialize base processor."""
        self.config_options = config_options
        self.mpi_config = mpi_config
        self.wrf_hydro_geo_meta = wrf_hydro_geo_meta
        self.dest_crs = CRS(4326)
        self.buffer = 0.02  # degree buffer around bounding box

    @cached_property
    def bounds(self) -> tuple[float, float, float, float]:
        """Get bounding box from geospatial dataframe.

        Apply buffer in known crs/units (degrees) and then convert back to src_crs.
        """
        return (
            self.gdf.to_crs(self.dest_crs)
            .buffer(self.buffer)
            .to_crs(self.src_crs)
            .total_bounds
        )

    @property
    def reprojected_xmin(self) -> float:
        """Minimum longitude in source CRS."""
        return self.bounds[0]

    @property
    def reprojected_ymin(self) -> float:
        """Minimum latitude in source CRS."""
        return self.bounds[1]

    @property
    def reprojected_xmax(self) -> float:
        """Maximum longitude in source CRS."""
        return self.bounds[2]

    @property
    def reprojected_ymax(self) -> float:
        """Maximum latitude in source CRS."""
        return self.bounds[3]

    @contextmanager
    def timing_block(self, step_str: str, log_callable: typing.Callable = None):
        """Context manager for timing code execution.

        Args:
            step_str: Description of the step being timed.
            log_callable: Callable used for sending the log message. Defaults to LOG.debug.

        """
        if log_callable is None:
            log_callable = LOG.debug
        start = perf_counter()
        log_callable(f"  Starting {step_str}")
        yield
        end = perf_counter()
        log_callable(
            f"  Execution time for {step_str}: {round(end - start, 2)} seconds"
        )

    @property
    def time_min(self) -> np.datetime64:
        """Calculate minimum time for forecast window.

        :return: Minimum time as np.datetime64
        """
        return np.datetime64(self.config_options.b_date_proc) + np.timedelta64(1, "h")

    @property
    def datetimes(self) -> pd.DatetimeIndex:
        """Date range for the forecast window."""
        return pd.date_range(start=self.time_min, end=self.time_max, freq="h")

    @property
    def years(self) -> list[int]:
        """List of years in the date range."""
        return sorted(list(set([date.year for date in self.datetimes])))

    @property
    def year_start_stop_dict(self) -> dict[int, tuple[pd.Timestamp, pd.Timestamp]]:
        """Dictionary of start and stop dates for each year in the date range."""
        year_dict = {}
        for year in self.years:
            year_dates = [date for date in self.datetimes if date.year == year]
            year_dict[year] = (year_dates[0], year_dates[-1])
        return year_dict

    @property
    def time_max(self) -> np.datetime64:
        """Calculate maximum time for forecast window.

        :return: Maximum time as np.datetime64
        """
        return (
            self.time_min
            + np.timedelta64(self.config_options.fcst_input_horizons[0], "m")
            + np.timedelta64(self.config_options.fcst_freq, "m")
        )

    @property
    def gpkg_name(self) -> str:
        """Return name of the geopackage."""
        return os.path.splitext(os.path.basename(self.config_options.geopackage))[0]

    @property
    def nc_path(self) -> str:
        """Construct file path for cached netcdf files."""
        return f"{os.environ.get('TMPDIR', '/tmp')}/{self.cache_filename}.nc"

    @property
    def cache_filename(self):
        """Cache filename."""
        return f"{self.dataset_name}_{self.gpkg_name}_{self.current_time_str}_{self.end_time_str}"

    @property
    def end_time_datetime(self) -> pd.Timestamp:
        """Datetime object for the end time step."""
        return self.start_end_datetimes.get(self.current_time)

    @property
    def end_time_str(self) -> str:
        """String representation of the end time step."""
        return self.end_time_datetime.strftime("%Y%m%d%H")

    @property
    def current_time_str(self) -> str:
        """String representation of the current time step."""
        return self.current_time.strftime("%Y%m%d%H")

    def update_dates(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Update start and end dates for caching."""
        start_date = end_date + timedelta(hours=1)
        end_date = start_date + self.cache_size
        if end_date > end:
            end_date = end
        return start_date, end_date

    @cached_property
    def start_end_datetimes(self) -> dict[pd.Timestamp, pd.Timestamp]:
        """Generate dictionary of start and end dates for caching.

        If the cache size exceeds the year boundary, it will create multiple
        start and end date pairs for each year. Otherwise, it will create
        start and end date pairs based on the cache size.
         :return: Dictionary of start and end dates as pd.Timestamp

         TODO for lru_cache / cached_property safety, confirm or enforce that these are never mutated:
            self.config_options.b_date_proc
            self.config_options.fcst_input_horizons
            self.config_options.fcst_freq
        """
        start_end_datetimes = {}
        for start, end in self.year_start_stop_dict.values():
            start_date = start
            end_date = start_date + self.cache_size
            if end_date > end:  # ensure end date does not exceed year boundary
                end_date = end
            while end_date <= end:  # loop until end date reaches year boundary
                start_end_datetimes[start_date] = end_date
                if end_date == end:
                    break
                start_date, end_date = self.update_dates(
                    start_date, end_date, end
                )  # update dates for next time window (based on cache size)
        return start_end_datetimes

    def process_historical_data(self, current_time: str) -> xr.Dataset:
        """Process forcing data for the given configuration and geospatial metadata.

        Test if the current time is in the start_end_datetimes keys (start of a time window).
        If so, compute the dataset. Otherwise, use the existing computed dataset.

        Then select the data for the current time from the computed dataset.

        :param current_time: Current time as string in format YYYYMMDDHH
        :return: xarray Dataset for the current time step
        :raises IndexError: If the current time is not in the dataset
        """
        self.current_time = current_time
        if self.current_time in self.start_end_datetimes.keys():
            self.computed_ds = self.compute_ds()
            gc.collect()  # reclaim potentially large previous computed_ds explicitly since memory management is an ongoing issue
        if self.current_time not in self.computed_ds.time.values:
            raise IndexError(
                f"The time provided ({self.current_time}) is not in the dataset. Please check that you have provided a time span that is valid for the given domain/dataset."
            )
        try:
            ds = self.computed_ds.sel(time=self.current_time)
        except KeyError:
            raise KeyError(
                f"The time provided ({self.current_time}) is not in the dataset. Please check that you have provided a time span that is valid for the given domain/dataset."
            )
        # if self.mpi_config.rank == 0:
        #     self.plot_precip(ds)
        # self.write_sum_tif(self.computed_ds)
        return ds

    def compute_ds(self) -> xr.Dataset:
        """Materialize lazy dask arrays into memory."""
        ds = None
        if self.mpi_config.rank == 0:
            with self.timing_block("computing dataset", LOG.info):
                ds = self.sliced_ds.rio.write_crs(self.src_crs)
        self.mpi_config.comm.barrier()
        ds = self.mpi_config.comm.bcast(ds, root=0)
        if self.mpi_config.rank == 0:
            if not os.path.exists(self.nc_path):
                tmp_file = (
                    f"{self.nc_path}.{rand_str(12)}{os.path.splitext(self.nc_path)[1]}"
                )
                LOG.info(f"Writing tmp file: {tmp_file}")
                try:
                    ds.to_netcdf(tmp_file, "w")
                    LOG.info(f"Renaming: {tmp_file} -> {self.nc_path}")
                    os.replace(tmp_file, self.nc_path)
                    LOG.info(f"Renamed: {tmp_file} -> {self.nc_path}")
                except Exception as e:
                    pass
        return ds

    @cached_property
    def gdf(self) -> gpd.GeoDataFrame:
        """Load and cache the geospatial dataframe."""
        gdf = gpd.read_file(self.config_options.geopackage, layer="divides")
        return gdf.to_crs(self.src_crs)

    def plot_precip(self, ds: xr.Dataset) -> None:
        """Plot precipitation field for the current time step."""
        qmesh = ds[self.precip_variable].plot()
        self.gdf.plot(ax=qmesh.axes, facecolor="none", edgecolor="black")

        plt.title(f"{self.precip_variable} at {str(ds.time.values)}")
        plt.savefig(
            f"{self.precip_variable}_{str(ds.time.values)}_{self.mpi_config.rank}.png"
        )
        plt.clf()

    def write_sum_tif(self, ds: xr.Dataset) -> None:
        """Write precip sum raster."""
        ds[self.precip_variable].sum("time").rio.write_crs(self.src_crs).rio.to_raster(
            f"{self.precip_variable}_sum.tif"
        )

    @cached_property
    def number_of_catchments(self) -> int:
        """Return number of catchments in the geospatial dataframe."""
        return len(self.gdf)

    @cached_property
    def cache_size(self) -> np.timedelta64:
        """Determine cache size based on number of catchments.

        20 in the numerator is an empirically derived value that balances memory usage
        and performance. It can be adjusted based on system capabilities and dataset size.
        24 and 365 are hours in  a day and days in a year, respectively.

        :return: Cache size as np.timedelta64
        """
        return np.timedelta64(
            max(round(24 * 365 * 20 / self.number_of_catchments), 12), "h"
        )

    def slice_ds(self, ds: xr.Dataset) -> xr.Dataset:
        """Subset dataset to spatial and temporal bounds.

        :return: Sliced Dataset
        """
        with self.timing_block("slicing dataset"):
            if self.current_time not in ds.time.values:
                raise IndexError(
                    f"The time provided is not in the dataset: {self.current_time}"
                )
            if self.end_time_datetime not in ds.time.values:
                raise IndexError(
                    f"The time provided is not in the dataset: {self.end_time_datetime}"
                )
            sliced_ds = ds.sel(
                {
                    self.x_label: slice(self.reprojected_xmin, self.reprojected_xmax),
                    self.y_label: slice(self.reprojected_ymin, self.reprojected_ymax),
                    self.time_label: slice(self.current_time, self.end_time_datetime),
                }
            )
            if sliced_ds[self.x_label].size == 0 or sliced_ds[self.y_label].size == 0:
                raise ValueError(
                    f"""Unable to find data for the specified input dataset, domain, 
                    and catchment locations. Check that the dataset is supported for 
                    the given domain. x-size: {sliced_ds[self.x_label].size} | 
                    y-size: {sliced_ds[self.y_label].size} | y-min: {self.reprojected_ymin} | 
                    y-max: {self.reprojected_ymax} | x-min: {self.reprojected_xmin} | 
                    x-max: {self.reprojected_xmax} | ds y-start coord: {ds[self.y_label].values[0]} | 
                    ds y-end coord: {ds[self.y_label].values[-1]} | ds x-start coord: {ds[self.x_label].values[0]} | 
                    ds x-end coord: {ds[self.x_label].values[-1]}"""
                )
        return sliced_ds

    def load_cache(self) -> xr.Dataset | None:
        """Load the cahed netcdf file.

        If unable to read (likely do to a locked file issue), try again after sleeping for 1 second.
        Tries 10 times. If it fails 10 times then try deleting the file.
        """
        if os.path.exists(self.nc_path):
            with self.timing_block(f"opening local dataset {self.nc_path}"):
                c = 0
                while c < 10:
                    try:
                        with xr.open_dataset(self.nc_path) as ds:
                            dataset = ds.load()
                        return dataset
                    except Exception as e:
                        LOG.warning(f"Lock on cache file; sleeping 1s({c}). Error: {e}")
                        sleep(1)
                        c += 1

                error_message = f"Exceeded number of attempts (10) to read local cache file for historical forcing data. File: {self.nc_path}. Deleteing the cache file and recreating from s3"
                LOG.warning(error_message)
                c = 0
                while c < 10:
                    try:
                        os.remove(self.nc_path)
                        break
                    except Exception as e:
                        LOG.warning(
                            f"Could not delete the locked cache file retrying in 1 second. Error: {e}"
                        )
                        sleep(1)
                        c += 1


class AORCConusProcessor(BaseProcessor):
    """Processor for CONUS AORC data."""

    def __init__(
        self,
        config_options: ConfigOptions,
        mpi_config: MpiConfig,
        wrf_hydro_geo_meta: dict,
    ):
        """Initialize AORC processor."""
        super().__init__(config_options, mpi_config, wrf_hydro_geo_meta)
        self.dataset_name = "AORC"
        self.precip_variable = "APCP_surface"
        self.x_label = "longitude"
        self.y_label = "latitude"
        self.time_label = "time"

    @cached_property
    def src_crs(self) -> CRS:
        """Get source CRS from dataset."""
        object_store = obstore.store.from_url(
            self.url(self.years[0]), skip_signature=True
        )
        return CRS(xr.open_zarr(ObjectStore(object_store)).rio.crs)

    def url(self, year: str) -> str:
        """Generate AORC S3 zarr URL for current year.

        :return: AORC S3 zarr URL
        """
        url = self.config_options.aorc_conus_year_url.format(
            source=self.config_options.aorc_conus_source, year=year
        )
        LOG.debug(f"AORC S3 URL: {url}\n")

        return url

    @property
    def sliced_ds(self) -> xr.Dataset:
        """Sliced dataset.

        :return: xarray Dataset
        :raises Exception: If zarr open fails
        """
        cached_data = self.load_cache()
        if cached_data is not None:
            return cached_data
        current_year = self.current_time.year
        try:
            object_store = obstore.store.from_url(
                self.url(current_year), skip_signature=True
            )
            with (
                xr.open_dataset(ObjectStore(object_store), engine="zarr") as ds,
                self.timing_block(f"Loading {self.dataset_name} data"),
            ):
                return (
                    self.slice_ds(ds)
                    .rename({self.x_label: "x", self.y_label: "y"})
                    .load()
                )
        except Exception as e:
            error_message = f"Error opening {self.dataset_name} data from {self.url(current_year)}: {e}\n"
            LOG.critical(error_message)
            raise ValueError(error_message)


class AORCAlaskaProcessor(BaseProcessor):
    """Processor for AORC Alaska data."""

    def __init__(
        self,
        config_options: ConfigOptions,
        mpi_config: MpiConfig,
        wrf_hydro_geo_meta: dict,
    ):
        """Initialize AORC Alaska processor."""
        raise NotImplementedError("AORC Alaska processor is not yet implemented.")
        super().__init__(config_options, mpi_config, wrf_hydro_geo_meta)
        self.dataset_name = "AORC"
        self.precip_variable = "APCP_surface"
        self.x_label = "longitude"
        self.y_label = "latitude"
        self.time_label = "time"

    @cached_property
    def src_crs(self):
        """Get source CRS from dataset."""
        object_store = obstore.store.from_url(
            self.url(self.years[0]), skip_signature=True
        )
        return CRS(xr.open_zarr(ObjectStore(object_store)).crs.attrs["spatial_ref"])

    def url(self, date: datetime) -> str:
        """Generate AORC S3 zarr URL for current year.

        :return: AORC S3 zarr URL
        """
        url = self.config_options.aorc_alaska_url.format(
            source=self.config_options.aorc_alaska_source,
            year=date.year,
            month=date.month,
            date=date.strftime("%Y%m%d%H"),
        )
        LOG.debug(f"AORC S3 URL: {url}\n")

        return url

    @property
    def sliced_ds(self) -> xr.Dataset:
        """Open dataset.

        :return: xarray Dataset
        :raises Exception: If zarr open fails
        """
        datasets = []
        for date in self.datetimes:
            try:
                with self.timing_block(f"lazy loading {self.dataset_name} data"):
                    load_dotenv(find_dotenv())
                    s3 = s3fs.S3FileSystem()
                    with s3.open(self.url(date)) as f:
                        with xr.open_dataset(f, engine="h5netcdf") as ds:
                            dataset = ds.load()
                        datasets.append(
                            self.slice_ds(dataset, date, date + np.timedelta64(1, "h"))
                        )
            except Exception as e:
                LOG.critical(
                    f"Error opening {self.dataset_name} data from {self.url(date)}: {e}\n"
                )

                raise e
        return xr.concat(datasets, dim="time").rename(
            {self.x_label: "x", self.y_label: "y"}
        )


class NWMV3Processor(BaseProcessor):
    """Processor for NWM data."""

    def __init__(
        self,
        config_options: ConfigOptions,
        mpi_config: MpiConfig,
        wrf_hydro_geo_meta: dict,
    ):
        """Initialize NWM processor."""
        super().__init__(config_options, mpi_config, wrf_hydro_geo_meta)
        self.dataset_name = "NWM"
        self.precip_variable = "RAINRATE"
        self.x_label = "x"
        self.y_label = "y"
        self.time_label = "time"

    @property
    def vars(
        self,
    ) -> list[str]:
        """List of NWM variables to extract."""
        return [
            "lwdown",
            "precip",
            "psfc",
            "q2d",
            "swdown",
            "t2d",
            "u2d",
            "v2d",
        ]


class NWMV3ConusProcessor(NWMV3Processor):
    """Processor for NWM CONUS data."""

    def __init__(
        self,
        config_options: ConfigOptions,
        mpi_config: MpiConfig,
        wrf_hydro_geo_meta: dict,
    ):
        """Initialize NWM CONUS processor."""
        super().__init__(config_options, mpi_config, wrf_hydro_geo_meta)

    def url(self, var: str) -> str:
        """Generate NWM S3 zarr URL for current variable.

        :return: NWM S3 zarr URL
        """
        url = self.config_options.nwm_url.format(
            source=self.config_options.nwm_source,
            domain=self.config_options.nwm_domain,
            var=var,
        )
        LOG.debug(f"NWM S3 URL: {url}\n")

        return url

    @cached_property
    def src_crs(self) -> CRS:
        """Get source CRS from dataset."""
        object_store = obstore.store.from_url(
            self.url(self.vars[0]), skip_signature=True
        )
        return CRS(xr.open_zarr(ObjectStore(object_store)).crs.attrs["spatial_ref"])

    @property
    def sliced_ds(self) -> xr.Dataset:
        """Open dataset.

        :return: xarray Dataset
        :raises Exception: If zarr open fails
        """
        datasets = []
        for var in self.vars:
            try:
                with self.timing_block(f"lazy loading {self.dataset_name} data"):
                    # TODO this object_store var is not used
                    object_store = obstore.store.from_url(
                        self.url(var), skip_signature=True
                    )
                    datasets.append(self.slice_ds(self.s3_lazy_ds[var]))
            except Exception as e:
                LOG.critical(
                    f"Error opening {self.dataset_name} data from {self.url(var)}: {e}\n"
                )
                raise e
        return xr.merge(datasets, compat="override").rename(
            {self.x_label: "x", self.y_label: "y"}
        )

    @cached_property
    def s3_lazy_ds(self) -> dict[str, xr.Dataset]:
        """Lazy load dataset from S3."""
        variables = {}
        for var in self.vars:
            object_store = obstore.store.from_url(self.url(var), skip_signature=True)
            variables[var] = xr.open_zarr(ObjectStore(object_store))
        return variables


class NWMV3OConusProcessor(NWMV3Processor):
    """Processor for NWM OCONUS data."""

    def __init__(
        self,
        config_options: ConfigOptions,
        mpi_config: MpiConfig,
        wrf_hydro_geo_meta: dict,
    ):
        """Initialize NWM OCONUS processor."""
        super().__init__(config_options, mpi_config, wrf_hydro_geo_meta)

    @cached_property
    def url(self) -> str:
        """Generate NWM S3 zarr URL.

        :return: NWM S3 zarr URL
        """
        url = self.config_options.nwm_url.format(
            source=self.config_options.nwm_source, domain=self.config_options.nwm_domain
        )
        LOG.debug(f"NWM S3 URL: {url}\n")

        return url

    @cached_property
    def src_crs(self) -> CRS:
        """Get source CRS from dataset."""
        object_store = obstore.store.from_url(self.url, skip_signature=True)
        return CRS(xr.open_zarr(ObjectStore(object_store)).crs.attrs["spatial_ref"])

    @property
    def sliced_ds(self) -> xr.Dataset:
        """Sliced dataset.

        :return: xarray Dataset
        :raises Exception: If zarr open fails
        """
        cached_data = self.load_cache()
        if cached_data is not None:
            return cached_data
        try:
            with self.timing_block(f"Loading {self.dataset_name} data"):
                return (
                    self.slice_ds(self.s3_lazy_ds)
                    .rename({self.x_label: "x", self.y_label: "y"})
                    .load()
                )
        except Exception as e:
            error_message = (
                f"Error opening {self.dataset_name} data from {self.url}: {e}\n"
            )
            LOG.critical(error_message)
            raise ValueError(error_message)

    @cached_property
    def s3_lazy_ds(self) -> xr.Dataset:
        """Lazy load dataset from S3."""
        object_store = obstore.store.from_url(self.url, skip_signature=True)
        return xr.open_zarr(ObjectStore(object_store))


class NWMV3AlaskaProcessor(NWMV3Processor):
    """Processor for NWM OCONUS data."""

    def __init__(
        self,
        config_options: ConfigOptions,
        mpi_config: MpiConfig,
        wrf_hydro_geo_meta: dict,
    ):
        """Initialize NWM OCONUS processor."""
        super().__init__(config_options, mpi_config, wrf_hydro_geo_meta)

    @cached_property
    def url(self) -> str:
        """Generate NWM S3 zarr URL.

        :return: NWM S3 zarr URL
        """
        url = self.config_options.nwm_url.format(
            source=self.config_options.nwm_source, domain=self.config_options.nwm_domain
        )
        LOG.debug(f"NWM S3 URL: {url}\n")

        return url

    @property
    def sliced_ds(self) -> xr.Dataset:
        """Sliced dataset.

        :return: xarray Dataset
        :raises Exception: If zarr open fails
        """
        cached_data = self.load_cache()
        if cached_data is not None:
            return cached_data
        try:
            with self.timing_block(f"Loading {self.dataset_name} data"):
                return (
                    self.slice_ds(self.s3_lazy_ds)
                    .rename({self.x_label: "x", self.y_label: "y"})
                    .load()
                )
        except Exception as e:
            error_message = f"Error opening {self.dataset_name} data from {self.url(self.current_time.year)}: {e}\n"
            LOG.critical(error_message)
            raise ValueError(error_message)

    @cached_property
    def src_crs(self) -> CRS:
        """Get source CRS from dataset."""
        return self.geo_grid["crs"].attrs["spatial_ref"]

    @property
    def geogrid_ldasout_spatial_metadata_path(self) -> str:
        """Path to geogrid spatial metadata file for assigning CRS.

        Currently assumed to exist as a sibling file to the nwm_geogrid file.
        TODO consider using forcing config attribute SpatialMetaIn, see: https://github.com/NGWPC/ngen-forcing/blob/0992b43391ba141717b7a80f10ef38478cef2eee/NextGen_Forcings_Engine_BMI/BMI_NextGen_Configs/README.md?plain=1#L136-L138
        """
        basename = "GEOGRID_LDASOUT_Spatial_Metadata_AK.nc"
        parent_dir = os.path.dirname(self.config_options.nwm_geogrid)
        p = os.path.join(parent_dir, basename)
        LOG.warning(
            f"For Alaska NWM forcing, not using SpatialMetaIn. Using this instead to define the mesh CRS (this path assumed to be sibling of NWM_Geogrid): {p}"
        )
        return p

    @property
    def geo_grid(self) -> xr.Dataset:
        """Load geogrid metadata."""
        geo_grid = xr.open_dataset(self.geogrid_ldasout_spatial_metadata_path)
        return geo_grid

    @cached_property
    def s3_lazy_ds(self) -> xr.Dataset:
        """Lazy load dataset from S3 with coordinates assigned.

        NOTE: The y coordinates need to be reversed for proper orientation.
        This is specific to the Alaska NWM Retrospective data due to the CRS and
        coordinates having to be pulled from the geo_grid and mapped to the actual
        data.
        """
        object_store = obstore.store.from_url(self.url, skip_signature=True)
        ds = xr.open_zarr(ObjectStore(object_store))
        ds = ds.assign_coords(
            {"x": self.geo_grid["x"].values, "y": self.geo_grid["y"].values[::-1]}
        )
        ds.rio.write_crs(self.src_crs, inplace=True)
        return ds
