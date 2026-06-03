import logging

import dask.array as da
import dask.config
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
from mpi4py.futures import MPICommExecutor

from nextgen_forcings_ewts import MODULE_NAME

LOG = logging.getLogger(MODULE_NAME)

# TODO expand for other domains


def get_domain_bounds_quick(ConfigOptions):
    """Quick extraction of domain bounds without full ESMF initialization."""
    try:
        from netCDF4 import Dataset

        idTmp = Dataset(ConfigOptions.geogrid, "r")
        lat_bounds = idTmp.variables[ConfigOptions.nodecoords_var][:, 1]
        lon_bounds = idTmp.variables[ConfigOptions.nodecoords_var][:, 0]
        idTmp.close()

        lat_min, lat_max = lat_bounds.min(), lat_bounds.max()
        lon_min, lon_max = lon_bounds.min(), lon_bounds.max()

        return lat_min, lat_max, lon_min, lon_max
    except Exception as e:
        LOG.critical(f"Could not extract bounds: {e}")
        return None


def check_time(ConfigOptions):
    """Set aws_time and get target_time from current_time."""
    if ConfigOptions.aws_time is None:
        ConfigOptions.aws_time = ConfigOptions.current_time

    target_time = ConfigOptions.current_time
    return target_time


def s3_access(ConfigOptions):
    """Use s3fs to access S3 data."""
    _s3 = s3fs.S3FileSystem(anon=True)

    if ConfigOptions.nwm_domain == "CONUS":
        nwm_vars = ["lwdown", "precip", "psfc", "q2d", "swdown", "t2d", "u2d", "v2d"]
        files = [
            s3fs.S3Map(
                root=ConfigOptions.nwm_url.format(
                    source=ConfigOptions.nwm_source,
                    domain=ConfigOptions.nwm_domain,
                    var=var,
                ),
                s3=_s3,
                check=False,
            )
            for var in nwm_vars
        ]
    else:  # TODO work on other domains
        files = [
            s3fs.S3Map(
                root=ConfigOptions.nwm_url.format(
                    source=ConfigOptions.nwm_source, domain=ConfigOptions.nwm_domain
                ),
                s3=_s3,
                check=False,
            )
        ]
    return files


def get_zarr_sample(files):
    """Create a sample from one zarr file."""
    zarr_sample = zarr.open_group(files[0], mode="r")
    return zarr_sample


def get_time_index(zarr_sample, target_time):
    """Convert times to indices for use in subsetting zarr data temporally."""
    time_attrs = dict(zarr_sample["time"].attrs)

    # Gets reference time from time.attrs - assumes units is hours since reference time
    reference_time = pd.Timestamp(
        zarr_sample["time"].attrs["units"].split("since ")[-1]
    )
    # reference_time = pd.Timestamp('1979-02-01')
    hours_diff = (target_time - reference_time).total_seconds() / 3600
    target_index = int(round(hours_diff))

    # Verify bounds
    time_length = zarr_sample["time"].shape[0]
    if target_index < 0 or target_index >= time_length:
        LOG.warning(
            f"Calculated index {target_index} is out of bounds [0, {time_length - 1}]"
        )
        target_index = max(0, min(target_index, time_length - 1))

    actual_time = reference_time + pd.Timedelta(hours=target_index)

    return target_index, actual_time


def get_spatial_bounds(zarr_sample, domain_bounds):
    """Use domain_bounds to get spatial bounds for subsetting zarr data geospatially"""

    spatial_bounds = None
    if domain_bounds is not None:
        lat_min, lat_max, lon_min, lon_max = domain_bounds

        # Add buffer (optional)
        buffer = 0.1  # degrees
        lat_min -= buffer
        lat_max += buffer
        lon_min -= buffer
        lon_max += buffer

        # Read coordinate arrays from zarr (only once)
        x_coords = zarr_sample["x"][:]
        y_coords = zarr_sample["y"][:]

        try:
            from pyproj import Transformer

            nwm_crs = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"

            # Transform domain bounds from geographic to projected
            transformer = Transformer.from_crs("EPSG:4326", nwm_crs, always_xy=True)

            # Transform corner points
            x_min_proj, y_min_proj = transformer.transform(lon_min, lat_min)
            x_max_proj, y_max_proj = transformer.transform(lon_max, lat_max)

            # Find spatial indices using projected coordinates
            x_mask = (x_coords >= x_min_proj) & (x_coords <= x_max_proj)
            y_mask = (y_coords >= y_min_proj) & (y_coords <= y_max_proj)

        except ImportError:
            LOG.error("pyproj not available, trying simple coordinate matching...")
            # Fallback without transformation - later steps will likely fail
            x_mask = (x_coords >= lon_min) & (x_coords <= lon_max)
            y_mask = (y_coords >= lat_min) & (y_coords <= lat_max)
        except Exception as e:
            LOG.error(f"Coordinate transformation failed: {e}")
            # Fallback without transformation - later steps will likely fail
            x_mask = (x_coords >= lon_min) & (x_coords <= lon_max)
            y_mask = (y_coords >= lat_min) & (y_coords <= lat_max)

        x_indices = np.where(x_mask)[0]
        y_indices = np.where(y_mask)[0]

        try:
            if len(x_indices) > 0 and len(y_indices) > 0:
                x_min_idx, x_max_idx = x_indices.min(), x_indices.max()
                y_min_idx, y_max_idx = y_indices.min(), y_indices.max()

                subset_size = (x_max_idx - x_min_idx) * (y_max_idx - y_min_idx)
                total_size = len(x_coords) * len(y_coords)

                spatial_bounds = (y_min_idx, y_max_idx, x_min_idx, x_max_idx)
            else:
                raise Exception("One or more indices not > 0")
        except Exception as e:
            LOG.error(f"{e}")
            spatial_bounds = None

    return spatial_bounds, x_coords, y_coords


def read_zarr_data(
    files, spatial_bounds, target_index, actual_time, x_coords, y_coords
):
    """Read zarr variables from subset of zarr data."""
    data_vars = {}
    coords = {}
    attrs = {}
    var_name_mapping = {
        "lwdown": "LWDOWN",
        "precip": "RAINRATE",
        "psfc": "PSFC",
        "q2d": "Q2D",
        "swdown": "SWDOWN",
        "t2d": "T2D",
        "u2d": "U2D",
        "v2d": "V2D",
    }
    LOG.debug("Reading zarr variables individually due to zarr structure.")
    # TODO: Investigate performance optimizations.
    # NWM Retrospective zarr data structure: a separate zarr file for each variable
    for i, file_map in enumerate(files):
        LOG.debug(f"Reading variable {i + 1}/{len(files)}...")

        zarr_group = zarr.open_group(file_map, mode="r")

        # Read coordinates (only once, with spatial subsetting)
        if i == 0:
            if spatial_bounds is not None:
                y_min_idx, y_max_idx, x_min_idx, x_max_idx = spatial_bounds
                coords["x"] = (["x"], x_coords[x_min_idx:x_max_idx])
                coords["y"] = (["y"], y_coords[y_min_idx:y_max_idx])
            else:
                # Fallback to full domain or default subset
                coords["x"] = (["x"], zarr_group["x"][:])
                coords["y"] = (["y"], zarr_group["y"][:])
            coords["time"] = (["time"], [actual_time])
            attrs = dict(zarr_group.attrs)

        # Extract variable name and read data
        url_parts = file_map.root.split("/")
        url_var = url_parts[-1].replace(".zarr", "")
        zarr_var_name = var_name_mapping.get(url_var, url_var.upper())

        if zarr_var_name in zarr_group:
            var_zarr = zarr_group[zarr_var_name]

            # Read with spatial and temporal subsetting
            if spatial_bounds is not None:
                y_min_idx, y_max_idx, x_min_idx, x_max_idx = spatial_bounds
                var_data = var_zarr[
                    target_index, y_min_idx:y_max_idx, x_min_idx:x_max_idx
                ]
            else:
                var_data = var_zarr[target_index, :, :]  # Full domain fallback

            # Read scale and offset from zarr attributes
            var_attrs = dict(var_zarr.attrs)
            scale_factor = var_attrs.get("scale_factor", 1.0)
            add_offset = var_attrs.get("add_offset", 0.0)
            fill_value = var_attrs.get("fill_value", None)

            # Adjust for scale and offset
            # Handle fill values first (convert to NaN)
            if fill_value is not None:
                var_data = np.where(var_data == fill_value, np.nan, var_data)

            # Apply scale factor and offset
            var_data_scaled = (var_data.astype(np.float32) * scale_factor) + add_offset

            # Add time dimension and convert to dask array
            var_data_with_time = np.expand_dims(
                var_data_scaled, axis=0
            )  # Add time dimension at front
            var_dask = da.from_array(var_data_with_time, chunks=(1, 350, 350))

            # remove scale/offset since we've applied them
            clean_attrs = var_attrs.copy()
            clean_attrs.pop("scale_factor", None)
            clean_attrs.pop("add_offset", None)

            # Store the processed data
            data_vars[zarr_var_name] = (["time", "y", "x"], var_dask, clean_attrs)

        else:
            LOG.warning(f"Variable {zarr_var_name} not found in zarr group")
    return data_vars, coords, attrs


def proc_nwm(ConfigOptions, MpiConfig):
    """Orchestrate processing functions and create output dataset."""
    ds_subset = None

    with MPICommExecutor(comm=MpiConfig.comm, root=0) as executor:
        with dask.config.set(scheduler=executor):
            if MpiConfig.rank == 0:
                try:
                    domain_bounds = get_domain_bounds_quick(ConfigOptions)
                    target_time = check_time(ConfigOptions)
                    files = s3_access(ConfigOptions)
                    zarr_sample = get_zarr_sample(files)
                    target_index, actual_time = get_time_index(zarr_sample, target_time)
                    spatial_bounds, x_coords, y_coords = get_spatial_bounds(
                        zarr_sample, domain_bounds
                    )
                    data_vars, coords, attrs = read_zarr_data(
                        files,
                        spatial_bounds,
                        target_index,
                        actual_time,
                        x_coords,
                        y_coords,
                    )

                    # Create xarray dataset from the subset data
                    ds_subset = xr.Dataset(
                        data_vars=data_vars, coords=coords, attrs=attrs
                    )

                except Exception as e:
                    LOG.critical(f"Error with direct zarr access: {e}")
                    import traceback

                    traceback.print_exc()
                    ds_subset = None

    MpiConfig.comm.barrier()
    ds_subset = MpiConfig.comm.bcast(ds_subset, root=0)
    return ds_subset
