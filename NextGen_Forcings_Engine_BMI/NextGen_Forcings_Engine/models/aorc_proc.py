import hashlib
import logging
import os
import time

import dask
import numpy as np
import xarray as xr
from mpi4py.futures import MPICommExecutor

from nextgen_forcings_ewts import MODULE_NAME

LOG = logging.getLogger(MODULE_NAME)

_aorc_cache = None


class AORCDataCache:
    """Cache for AORC data to avoid redundant loading and processing."""

    def __init__(self):
        """Initialize empty cache."""
        self.final_ds = None
        self.time_sel_ds = None


def _get_cache_path(ConfigOptions) -> str:
    """Generate disk cache file path based on configuration. Hash the file name.

    :param ConfigOptions: Configuration with b_date_proc, fcst_input_horizons, geogrid, current_time
    :return: Path to NetCDF cache file
    """
    cache_dir = "/tmp/"
    cache_key = f"{ConfigOptions.b_date_proc}_{ConfigOptions.fcst_input_horizons[0]}_{ConfigOptions.geogrid}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_path = os.path.join(
        cache_dir, f"aorc_{cache_hash}_{ConfigOptions.current_time.year}.nc"
    )

    # LOG.debug(f"cache_path: {cache_path}\n")

    return cache_path


# def _load_from_cache(ConfigOptions, cache_path: str) -> xr.Dataset:
#    """
#    Load time-selected dataset from disk cache.
#
#    :param ConfigOptions: Configuration with current_time
#    :param cache_path: Path to cached NetCDF file
#    :return: Time-selected Dataset if cache exists and is valid, None otherwise
#    """
#
#    try:
#        if os.path.exists(cache_path):
#            LOG.debug(f"Loading from cache: {cache_path}\n")
#            ds = xr.open_dataset(cache_path, engine='netcdf4')
#            c_time_np = np.datetime64(ConfigOptions.current_time)
#            time_sel_ds = ds.sel(time=c_time_np)
#            return time_sel_ds
#    except Exception as e:
#        LOG.warning(f"Cache load failed: {e}\n")
#    return None


def _save_to_cache(ds: xr.Dataset, cache_path: str) -> None:
    """Save dataset to disk cache as NetCDF.

    :param ds: Dataset to cache
    :param cache_path: Target cache file path
    """
    try:
        ds.to_netcdf(cache_path, engine="netcdf4")
        LOG.debug(f"Cache saved successfully: {cache_path}\n")
    except Exception as e:
        LOG.warning(f"Failed to cache AORC data: {e}")


def check_time(ConfigOptions) -> bool:
    """Check if current year differs from cached year.

    :param ConfigOptions: Configuration with current_time and aws_time
    :return: True if year changed, False unchanged
    """
    LOG.debug(f"ConfigOptions.current_time: {ConfigOptions.current_time}\n")
    if (
        ConfigOptions.aws_time is None
        or ConfigOptions.current_time.year != ConfigOptions.aws_time.year
    ):
        ConfigOptions.aws_time = ConfigOptions.current_time
        return True
    else:
        ConfigOptions.aws_time = ConfigOptions.current_time
        return False


def set_year(ConfigOptions) -> str:
    """Generate AORC S3 zarr URL for current year.

    :param ConfigOptions: Configuration with aws_time, aorc_year_url, aorc_source
    :return: AORC S3 zarr URL
    """
    year = ConfigOptions.aws_time.year
    url = ConfigOptions.aorc_year_url.format(
        source=ConfigOptions.aorc_source, year=year
    )
    LOG.debug(f"AORC S3 URL: {url}\n")

    return url


def get_bounds_quick(wrf_hydro_geo_meta) -> tuple:
    """Extract spatial bounds from geospatial metadata.

    :param wrf_hydro_geo_meta: Metadata with lon_bounds, lat_bounds
    :return: Tuple of (xmax, xmin, ymax, ymin)
    """
    xmax = np.max(wrf_hydro_geo_meta.lon_bounds)
    xmin = np.min(wrf_hydro_geo_meta.lon_bounds)
    ymax = np.max(wrf_hydro_geo_meta.lat_bounds)
    ymin = np.min(wrf_hydro_geo_meta.lat_bounds)

    LOG.debug(f"  xmin: {xmin}, xmax: {xmax}\n")
    LOG.debug(f"  ymin: {ymin}, ymax: {ymax}\n")

    return xmax, xmin, ymax, ymin


def create_dataset(url: str) -> xr.Dataset:
    """Open AORC zarr dataset from S3.

    :param url: S3 zarr URL
    :return: xarray Dataset
    :raises Exception: If zarr open fails
    """
    try:
        t0 = time.time()
        ds = xr.open_zarr(url, storage_options={"anon": True})
        t1 = time.time()
        LOG.debug(f"AORC AWS data opened in: {t1 - t0:.3f}s\n")

        return ds

    except Exception as e:
        LOG.critical(f"Error opening AORC AWS data from {url}: {e}\n")
        raise e


def slice_dataset(
    ds: xr.Dataset, xmax: float, xmin: float, ymax: float, ymin: float
) -> xr.Dataset:
    """Subset dataset to spatial bounds.

    :param ds: Input Dataset
    :param xmax: Maximum longitude
    :param xmin: Minimum longitude
    :param ymax: Maximum latitude
    :param ymin: Minimum latitude
    :return: Spatially subsetted Dataset
    """
    t0 = time.time()
    sliced_ds = ds.sel(longitude=slice(xmin, xmax), latitude=slice(ymin, ymax))
    t1 = time.time()
    LOG.debug(f"slice_dataset took: {t1 - t0:.3f}s\n")

    return sliced_ds


def time_slice(ConfigOptions, sliced_ds: xr.Dataset) -> xr.Dataset:
    """Subset dataset to forecast time window.

    :param ConfigOptions: Configuration with b_date_proc and fcst_input_horizons
    :param sliced_ds: Geospatially subsetted input Dataset
    :return: Temporally subsetted Dataset
    """
    try:
        time_min = np.datetime64(ConfigOptions.b_date_proc)
        time_max = time_min + np.timedelta64(ConfigOptions.fcst_input_horizons[0], "m")
        time_sliced_ds = sliced_ds.sel(time=slice(time_min, time_max))
        LOG.debug(f"time_min: {time_min}\n")
        LOG.debug(f"time_max: {time_max}\n")
    except Exception as e:
        LOG.warning(f"time_slice failed due to: {e}\n")

    return time_sliced_ds


def compute_dataset(sliced_ds: xr.Dataset) -> xr.Dataset:
    """Materialize lazy dask arrays into memory.

    :param sliced_ds: Dask-backed Dataset
    :return: Computed Dataset
    """
    LOG.info("starting compute\n")
    t0 = time.time()
    year_ds = sliced_ds.compute()
    t1 = time.time()
    LOG.info(f"finished compute: {t1 - t0:.3f}s\n")

    return year_ds


def proc_aorc(ConfigOptions, MpiConfig, wrf_hydro_geo_meta):
    """Process AORC data for the given configuration and geospatial metadata."""
    global _aorc_cache

    if _aorc_cache is None:
        _aorc_cache = AORCDataCache()

    with MPICommExecutor(comm=MpiConfig.comm, root=0) as executor:
        with dask.config.set(scheduler=executor):
            if MpiConfig.rank == 0:
                try:
                    c_time_np = np.datetime64(ConfigOptions.current_time)
                    check_bool = check_time(ConfigOptions)

                    if check_bool:
                        # Year changed - need to load new dataset
                        xmax, xmin, ymax, ymin = get_bounds_quick(wrf_hydro_geo_meta)
                        cache_path = _get_cache_path(ConfigOptions)

                        if os.path.exists(cache_path):
                            # Load from disk cache (once per year)
                            LOG.debug(
                                f"Loading year dataset from cache: {cache_path}\n"
                            )
                            _aorc_cache.final_ds = xr.open_dataset(
                                cache_path, engine="netcdf4"
                            )
                        else:
                            # Cache miss - fetch from AWS
                            url = set_year(ConfigOptions)
                            ds = create_dataset(url)
                            sliced_ds = slice_dataset(ds, xmax, xmin, ymax, ymin)
                            time_sliced_ds = time_slice(ConfigOptions, sliced_ds)
                            _aorc_cache.final_ds = compute_dataset(time_sliced_ds)

                            # Save to disk cache
                            _save_to_cache(_aorc_cache.final_ds, cache_path)

                        LOG.debug("final_ds updated for new year\n")

                    # Always select from in-memory/open dataset
                    _aorc_cache.time_sel_ds = _aorc_cache.final_ds.sel(time=c_time_np)

                except Exception as e:
                    LOG.critical(f"Error with AORC processing: {e}")

    MpiConfig.comm.barrier()
    aorc_ds = MpiConfig.comm.bcast(_aorc_cache.time_sel_ds, root=0)

    return aorc_ds
