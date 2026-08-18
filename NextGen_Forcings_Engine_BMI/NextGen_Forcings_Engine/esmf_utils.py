import types

import esmpy as ESMF

### TODO fix circular import raised during `from .bmi_model import ESMF`.
### shapely must be imported before ESMF to avoid segfault with shapely 2+
import shapely

from . import retry_utils
from .core.config import ConfigOptions
from .core.parallel import MpiConfig


@retry_utils.retry_w_mpi_context(
    abort=True, num_retries=3, sleep_start=1, sleep_factor=3
)
def esmf_field_retry(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    err_handler: types.ModuleType,
    *esmf_args,
    **esmf_kwargs,
):
    """ESMF.Field() call, wrapped by MPI-aware retry decorator."""
    return ESMF.Field(*esmf_args, **esmf_kwargs)


@retry_utils.retry_w_mpi_context(
    abort=True, num_retries=3, sleep_start=1, sleep_factor=3
)
def esmf_grid_retry(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    err_handler: types.ModuleType,
    *esmf_args,
    **esmf_kwargs,
):
    """ESMF.Grid() call, wrapped by MPI-aware retry decorator."""
    return ESMF.Grid(*esmf_args, **esmf_kwargs)


@retry_utils.retry_w_mpi_context(
    abort=True, num_retries=3, sleep_start=1, sleep_factor=3
)
def esmf_mesh_retry(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    err_handler: types.ModuleType,
    *esmf_args,
    **esmf_kwargs,
):
    """ESMF.Mesh() call, wrapped by MPI-aware retry decorator."""
    return ESMF.Mesh(*esmf_args, **esmf_kwargs)


@retry_utils.retry_w_mpi_context(
    abort=True, num_retries=3, sleep_start=1, sleep_factor=3
)
def esmf_regrid_retry(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    err_handler: types.ModuleType,
    *esmf_args,
    **esmf_kwargs,
):
    """ESMF.Regrid() call, wrapped by MPI-aware retry decorator."""
    return ESMF.Regrid(*esmf_args, **esmf_kwargs)


@retry_utils.retry_w_mpi_context(
    abort=True, num_retries=3, sleep_start=1, sleep_factor=3
)
def esmf_regridfromfile_retry(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    err_handler: types.ModuleType,
    *esmf_args,
    **esmf_kwargs,
):
    """ESMF.RegridFromFile() call, wrapped by MPI-aware retry decorator."""
    return ESMF.RegridFromFile(*esmf_args, **esmf_kwargs)


@retry_utils.retry_w_mpi_context(
    abort=True, num_retries=3, sleep_start=1, sleep_factor=3
)
def esmf_regridobj_call_retry(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    err_handler: types.ModuleType,
    regridObj: ESMF.api.regrid.Regrid | ESMF.api.regrid.RegridFromFile,
    *esmf_args,
    **esmf_kwargs,
):
    """Call to provided regridObj (or regridObj_elem) object, wrapped by MPI-aware retry decorator.

    These objects are attrs of class .core.forcingInputMod.input_forcings.
    """
    return regridObj(*esmf_args, **esmf_kwargs)
