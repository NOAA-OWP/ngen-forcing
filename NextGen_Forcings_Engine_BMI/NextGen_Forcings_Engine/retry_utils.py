import functools
import time
import traceback
import types

from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.parallel import MpiConfig
from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.config import ConfigOptions


def retry_w_mpi_context(
    abort: bool, num_retries: int, sleep_start: float, sleep_factor: float
):
    """Retry with MPI context.

    Decorator intended to retry functions in MPI context. In the event of a fail out (after multiple attempts),
    this decorator will either MPI abort directly, or reraise the final exception.  In order to allow this to
    be called from only some ranks and not others (such as rank 0 only), this does not call
    err_handler.check_program_status, since that includes a MPI barrier.

    Causes any/all ranks to call their own MPI Abort(), rather than only rank 0 calling MPI Abort().

    May only wrap functions that include the following parameters as their first three arguments:
            mpi_config: MpiConfig,
            config_options: ConfigOptions,
            err_handler: types.ModuleType,

    :param abort: If True, on fail-out, MPI Abort() (system exit all ranks) without reraising the exeption. If False, on fail-out, reraise the exception of the final attempt.
    :param num_retries: The number of retries to perform. Must be >= 0.
    :param sleep_start: The sleep duration in seconds, between the first and second attempts.
    :param sleep_factor: With each attempt prior to fail-out, the sleep duration is multiplied by this amount.
    :return: On success, the decorated function returns its normally returned value. On fail-out, either an exception is raised, or MPI Abort() is called (system exit).
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(
            mpi_config: MpiConfig,
            config_options: ConfigOptions,
            err_handler: types.ModuleType,
            *args,
            **kwargs,
        ):
            if not isinstance(mpi_config, MpiConfig):
                raise TypeError(
                    f"Expected type {MpiConfig} for mpi_config, got: {type(mpi_config)}"
                )
            if not isinstance(config_options, ConfigOptions):
                raise TypeError(
                    f"Expected type {ConfigOptions} for config_options, got: {type(config_options)}"
                )
            if not isinstance(err_handler, types.ModuleType):
                raise TypeError(
                    f"Expected type {types.ModuleType} for err_handler, got: {type(err_handler)}"
                )
            if num_retries < 0:
                raise ValueError(f"Expected num_retries >= 0, got: {num_retries}")

            sleep_sec = sleep_start
            attempt = 0
            while True:
                attempt += 1
                msg = f"Starting attempt {attempt} of {num_retries + 1} for func: {func.__name__}."
                err_handler.log_msg(config_options, mpi_config, debug=True, msg=msg)
                try:
                    # if attempt < 2: raise RuntimeError("Testing one retry")
                    # if True: raise RuntimeError("Testing retry-failout")
                    # if mpi_config.rank == 0:  raise RuntimeError(f"Testing retry-failout on rank 0")
                    # if mpi_config.rank == 1:  raise RuntimeError(f"Testing retry-failout on rank 1")
                    ret = func(mpi_config, config_options, err_handler, *args, **kwargs)
                except Exception as e:
                    # Fail
                    msg = f"Attempt {attempt} of {num_retries + 1} for func: {func.__name__} failed with error: {repr(e)}."
                    if attempt < num_retries + 1:
                        # Retry
                        msg += f" Retrying in {sleep_sec} seconds."
                        err_handler.log_warning(config_options, mpi_config, msg=msg)
                        time.sleep(sleep_sec)
                        sleep_sec *= sleep_factor
                    else:
                        # Fail out
                        msg += " Attempts exceeded limit."
                        if abort:
                            msg += f" Will MPI Abort(). Traceback:\n{traceback.format_exc()}"
                            err_handler.log_critical(
                                config_options, mpi_config, msg=msg
                            )
                            mpi_config.abort_with_cleanup(1)
                        else:
                            msg += " Reraising exception."
                            err_handler.log_critical(
                                config_options, mpi_config, msg=msg
                            )
                            e.args = (msg,) + e.args
                            raise e
                        raise RuntimeError("Should not get here.")
                else:
                    msg = f"func {func.__name__} finished after {attempt} attempts."
                    err_handler.log_msg(config_options, mpi_config, debug=True, msg=msg)
                    return ret

        return wrapper

    return decorator


def retry_simple(num_retries: int, sleep_start: float, sleep_factor: float):
    """Retry simple decorator.

    Decorator intended to retry functions in non-MPI context, that do not involve collective / barrier calls.

    May wrap any function.

    :param abort: If True, on fail-out, system exit without reraising the exeption. If False, on fail-out, reraise the exception of the final attempt.
    :param num_retries: The number of retries to perform. Must be >= 0.
    :param sleep_start: The sleep duration in seconds, between the first and second attempts.
    :param sleep_factor: With each attempt prior to fail-out, the sleep duration is multiplied by this amount.
    :return: On success, the decorated function returns its normally returned value. On fail-out, either an exception is raised, or system exit is called.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sleep_sec = sleep_start
            for i in range(num_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if i == num_retries:
                        raise
                    time.sleep(sleep_sec)
                    sleep_sec *= sleep_factor

        return wrapper

    return decorator
