from . import retry_utils
import traceback
import types
import typing
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
    ignore_filenotfound: bool = False,
):
    """os.remove() call, wrapped by simple retry decorator."""
    try:
        os.remove(path)
    except FileNotFoundError as e:
        if not ignore_filenotfound:
            raise e


def os_remove_rank_0(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    err_handler: types.ModuleType,
    *args,
    **kwargs,
) -> None:
    """If rank 0, remove the file if it exists. Ignore FileNotFoundErrors.
    Collective, must be called by all ranks."""
    if mpi_config.rank == 0:
        _os_remove(mpi_config, config_options, err_handler, *args, **kwargs)
    err_handler.check_program_status(config_options, mpi_config)


def os_remove(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    err_handler: types.ModuleType,
    *args,
    **kwargs,
) -> None:
    """Remove the file if it exists. Ignore FileNotFoundErrors.
    Collective, must be called by all ranks."""
    _os_remove(mpi_config, config_options, err_handler, *args, **kwargs)
    err_handler.check_program_status(config_options, mpi_config)


def _os_remove(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    err_handler: types.ModuleType,
    file_path: str,
    msg_prefix: str = "",
) -> None:
    """Remove the file if it exists. Ignore FileNotFoundErrors.
    Does not make collective call."""
    err_handler.log_msg(
        config_options,
        mpi_config,
        debug=True,
        msg=f"{msg_prefix}Removing file if exists: {file_path}",
    )
    os_remove_retry(file_path, ignore_filenotfound=True)


def close_rank_0(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    err_handler: types.ModuleType,
    *args,
    **kwargs,
) -> None:
    """If rank 0, close the file handle. Wraps _close.
    Collective, must be called by all ranks.
    file_handle must have a close() method or be None.
    If rank != 0 or file_handle is None, do nothing except the error handler collective call."""
    if mpi_config.rank == 0:
        _close(mpi_config, config_options, err_handler, *args, **kwargs)
    err_handler.check_program_status(config_options, mpi_config)


def close(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    err_handler: types.ModuleType,
    *args,
    **kwargs,
) -> None:
    """Close the file handle. Wraps _close.
    Collective, must be called by all ranks.
    file_handle must have a close() method or be None.
    If file_handle is None, do nothing except the error handler collective call."""
    _close(mpi_config, config_options, err_handler, *args, **kwargs)
    err_handler.check_program_status(config_options, mpi_config)


def _close(
    mpi_config: MpiConfig,
    config_options: ConfigOptions,
    err_handler: types.ModuleType,
    file_handle: typing.Any | None,
    msg_prefix: str = "",
) -> None:
    """Close the file handle.
    file_handle must have a close() method or be None.
    If file_handle is None, do nothing.
    Does not make collective call."""
    if file_handle is None:
        return
    if not hasattr(file_handle, "close"):
        raise RuntimeError(
            f"Provided object for file_handle does not have a close method: {file_handle}"
        )
    # Get file name from the handle
    if hasattr(file_handle, "filepath"):
        fn = getattr(file_handle, "filepath")
        if not isinstance(fn, str):
            fn = fn()  # `filepath` is often a method rather than an attribute, e.g. for NetCDF files
    elif hasattr(file_handle, "name"):
        fn = getattr(file_handle, "name")
    else:
        fn = "(UNKNOWN)"
    if not isinstance(fn, str):
        raise TypeError(
            f"Expected fn to resolve to a string for file_handle {file_handle}, got: {type(fn)}"
        )
    # Close
    err_handler.log_msg(
        config_options, mpi_config, debug=True, msg=f"Closing file: {fn}"
    )
    try:
        file_handle.close()
    except Exception as e:
        msg = f"{msg_prefix}Could not close file object: {file_handle}. File name: {fn}. Exception: {e}. Traceback: {traceback.format_exc()}"
        err_handler.log_critical(config_options, mpi_config, msg)
        raise RuntimeError(msg) from e
