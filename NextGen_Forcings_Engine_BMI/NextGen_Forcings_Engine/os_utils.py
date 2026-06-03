from . import retry_utils
import types
from .core.parallel import MpiConfig
from .core.config import ConfigOptions
import os


@retry_utils.retry_w_mpi_context(
    abort=True, num_retries=3, sleep_start=1, sleep_factor=3
)
def assert_path_exists_retry(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    err_handler: types.ModuleType,
    path: str,
):
    """Raise FileNotFoundError if the path does not exist. Wrapped by MPI-aware retry decorator."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)


@retry_utils.retry_simple(num_retries=3, sleep_start=1, sleep_factor=3)
def os_remove_retry(
    path: str,
):
    """os.remove() call, wrapped by simple retry decorator."""
    os.remove(path)
