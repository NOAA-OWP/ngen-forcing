import types

import netCDF4

from . import retry_utils
from .core.config import ConfigOptions
from .core.parallel import MpiConfig


@retry_utils.retry_w_mpi_context(
    abort=True, num_retries=3, sleep_start=1, sleep_factor=3
)
def nc_Dataset_retry(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    err_handler: types.ModuleType,
    *args,
    **kwargs,
):
    """netCDF4 Dataset open with MPI retry logic."""
    return netCDF4.Dataset(*args, **kwargs)


@retry_utils.retry_w_mpi_context(
    abort=True, num_retries=3, sleep_start=1, sleep_factor=3
)
def nc_read_var_retry(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    err_handler: types.ModuleType,
    nc_var: netCDF4.Variable,
    slices=None,
):
    """Read NetCDF variable data with MPI retry logic."""
    if slices is None:
        return nc_var[:].data
    return nc_var[slices].data
