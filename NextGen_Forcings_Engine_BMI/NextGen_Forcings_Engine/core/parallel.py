from __future__ import annotations

import atexit
import os
import signal
import sys
from functools import partial
from typing import TYPE_CHECKING, TypeVar

import mpi4py
import numpy as np

from . import err_handler, mpi_utils

mpi4py.rc.threads = False

from mpi4py import MPI  # noqa: E402

if TYPE_CHECKING:
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.config import (
        ConfigOptions,
    )
    from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.geoMod import (
        GriddedGeoMeta,
    )

# If MPI was initialized outside of python,
# disable initialization/finalization behavior
if MPI.Is_initialized():
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False


_T = TypeVar("_T")


class MpiConfig:
    """MPI config class.

    Class for defining the MPI parameters,
    along with initialization of the MPI communication
    handle from mpi4py.
    """

    comm: MPI.Intracomm
    rank: int
    """MPI rank of the process of this instance."""
    size: int
    """The number of MPI processes on this run."""
    uid64: str
    """Random 16 chars based on random uint64 shared between all processes in a run."""
    config_options: ConfigOptions
    """Forcing Engine configurations options. Messaging will use the instance of ConfigOptions passed to it in the constructor."""

    def __init__(self, config_options: ConfigOptions):
        """Initialize the MPI abstract class that will contain basic information and communication handles.

        NOTE: Temporary files that are created during a normal forcing run should be cleaned up using the `cleanup` method.
        """
        self.comm = None
        self.rank = None
        self.size = None
        self.uid64 = None
        self.config_options = config_options
        self.log_debug = partial(err_handler.log_msg, self.config_options, self, True)
        self.log_info = partial(err_handler.log_msg, self.config_options, self, False)
        self.log_warning = partial(err_handler.log_warning, self.config_options, self)
        self.__register_exit_handlers()

    def initialize_comm(self, comm: MPI.Intracomm | None = None) -> None:
        """Initialize MPI communication, including getting MPI rank and size.
        Also generates the UID for the run.

        Usage note: if an exception is thrown, an error messsage is added to the
        `config_options` object with the expectation the caller will handle
        logging the error message generated. The error will be reraised.
        """
        try:
            self.comm = comm if comm is not None else MPI.COMM_WORLD
            self.comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)
        except AttributeError:
            self.config_options.errMsg = (
                "Unable to initialize the MPI Communicator object"
            )
            raise

        try:
            self.size = self.comm.Get_size()
        except MPI.Exception:
            self.config_options.errMsg = "Unable to retrieve the MPI size."
            raise

        try:
            self.rank = self.comm.Get_rank()
        except MPI.Exception:
            self.config_options.errMsg = "Unable to retrieve the MPI processor rank."
            raise

        try:
            self.uid64 = mpi_utils.get_new_broadcasted_uid()
        except Exception:
            self.config_options.errMsg = "Unable to generate a global unique ID."
            raise

        wait_for_debug = os.getenv("WAIT_FOR_DEBUGPY", "")
        if wait_for_debug.lower() in ("true", "1"):
            self.wait_for_debugpy_client()

    def abort_with_cleanup(self, errorcode: int) -> None:
        """Call cleanup methods, before calling MPI Abort.
        Do not make direct calls to MPI Abort without this method.
        Use this method for all MPI abort needs."""
        if self.comm is None:
            raise RuntimeError("comm is not initialized")
        self.cleanup()
        self.log_debug("About to MPI Abort")
        self.comm.Abort(errorcode)

    def cleanup(self) -> None:
        """High-level cleanup routine called during BMI finalization."""
        self.log_debug("About to clean up")
        self._cleanup_scratch_dir()
        self._cleanup_geogrid()

    def _cleanup_scratch_dir(self) -> None:
        """Remove contents of scratch dir.
        TODO: Potentially, the scratch dir could be replaced with /tmp/,
        but /tmp/ files remain in the container and are not available outside of the container.
        So use cases such as self._output_obj.outPath in `bmi_model.py` would need to be updated
        to use a new configuration key for specifying the output directory for (optionally) storing permanent results.
        The process of writing to outPath could still leverage /tmp/ while the file is incomplete / in-process,
        using a OS rename to move the file to the shared location once writing has completed and the file handle has been closed.
        """
        self.log_debug("Cleanup: starting scratch dir cleanup")
        if self.config_options is None:
            self.log_debug("Cleanup: config_options is not set")
            return
        if not self.config_options.scratch_dir:
            self.log_debug("Cleanup: scratch dir is not set")
            return

        self.log_debug(
            f"Cleanup: listing scratch dir: {self.config_options.scratch_dir}"
        )
        contents = self.try_list_dir_no_reraise(self.config_options.scratch_dir)
        # NFS mounts may create temporary files to facilitate read-after-delete functionality on linux systems
        # these will be cleaned when the mount is removed but will throw an error if python tries to remove it
        # the file name is typically ".nfs" followed by numbers, so we'll just ignore files that start with it
        #
        # Only delete files that don't start with either of these
        skip_starts = (".nfs", "NextGen_Forcings_Engine")
        to_delete = [n for n in contents if not n.startswith(skip_starts)]
        for fn in to_delete:
            fp = os.path.join(self.config_options.scratch_dir, fn)
            if os.path.isfile(fp):
                self.try_delete_file_no_reraise(fp)
            elif os.path.isdir(fp):
                self.try_remove_empty_dir_no_reraise(fp)

        if self.config_options._scratch_dir_has_been_uniquefied:
            self.try_remove_empty_dir_no_reraise(self.config_options.scratch_dir)

    def _cleanup_geogrid(self) -> None:
        """Remove temporary geogrid file if it exists."""
        self.log_debug("Cleanup: starting geogrid cleanup")
        if self.config_options is None:
            self.log_debug("Cleanup: config_options is not set")
            return
        geogrid = getattr(self.config_options, "geogrid", None)
        if geogrid is not None:
            self.try_delete_file_no_reraise(geogrid)
        else:
            self.log_debug("Cleanup: config_options.geogrid is not set")

    def try_list_dir_no_reraise(self, dir_path: str) -> list[str]:
        """Try to list the directory and return a list of its contents.
        Do not reraise an exception if it fails due to FileNotFoundError or NotADirectoryError"""
        self.log_debug(f"Trying to list directory: {dir_path}")
        try:
            return os.listdir(dir_path)
        except (FileNotFoundError, NotADirectoryError) as e:
            self.log_debug(
                f"Could not list (it may have already been deleted): {dir_path}: {e}"
            )
            return []

    def try_delete_file_no_reraise(self, file_path: str) -> None:
        """Try to delete the file, do not reraise an exception if it fails due to OSError"""
        self.log_debug(f"Trying to delete file: {file_path}")
        try:
            os.remove(file_path)
        except OSError as e:
            self.log_debug(
                f"Could not delete file (it may have already been deleted): {file_path}: {e}"
            )
        else:
            self.log_info(f"Deleted file: {file_path}")

    def try_remove_empty_dir_no_reraise(self, dir_path: str) -> None:
        """Try to rmdir the path, do not reraise an exception if it fails due to OSError"""
        self.log_debug(f"Trying to remove directory: {dir_path}")
        try:
            os.rmdir(dir_path)
        except OSError as e:
            self.log_debug(
                f"Could not rmdir (it may have already been deleted): {dir_path}: {e}"
            )
        else:
            self.log_info(f"Removed directory: {dir_path}")

    def wait_for_debugpy_client(self):
        """Block until the debugpy clients have attached to cppdbg/gdb.

        This is for debugging concurrent ngen-forcing MPI ranks (processes).
        See `launch.json`, `devcontainer.json`, and `tasks.json` in the nwm-rte repository for details.
        """
        import debugpy

        debugpy.listen(("localhost", 5678 + self.rank))
        debugpy.wait_for_client()

    def broadcast_parameter(self, value_broadcast: _T) -> _T:
        """Broadcast a single parameter value to all processors.

        Generic function for sending a parameter value out to the processors.
        :param value_broadcast:
        :param config_options:
        :return:
        """
        if self.size == 1:
            return value_broadcast
        try:
            return self.comm.bcast(value_broadcast, root=0)
        except Exception as e:
            self.config_options.errMsg = f"Unable to broadcst single value {value_broadcast} from rank 0: {e.__class__.__name__} -- {e}"
            err_handler.log_critical(self.config_options, self)
            raise

    def scatter_array(
        self,
        geo_meta: GriddedGeoMeta,
        src_array: np.ndarray,
        config_options: ConfigOptions,
    ):
        """Scatter an array based on the input dataset type from rank 0 to all other ranks.

        Generic function for calling scatter functions based on
        the input dataset type.

        :param geo_meta: GriddedGeoMeta instance used to determine the extent of the data received.
        :param src_array: Data to be shared with other MPI ranks.
        :param config_options:
        :return: The results of the scattered data filtered to the extent of `geo_meta`
        """
        # Determine which type of input array we have based on the
        # type of numpy array.
        data_type_flag = -1
        if self.rank == 0:
            if src_array.dtype == np.float32:
                data_type_flag = 1
            if src_array.dtype == np.float64:
                data_type_flag = 2
            if src_array.dtype == bool:
                data_type_flag = 3

        # Broadcast the data_type_flag to other processors
        if self.rank == 0:
            data_type_buffer = np.array([data_type_flag], np.int32)
        else:
            data_type_buffer = np.empty(1, np.int32)

        try:
            self.comm.Bcast(data_type_buffer, root=0)
        except Exception as e:
            config_options.errMsg = f"Unable to broadcast numpy datatype value from rank 0: {e.__class__.__name__} -- {e}"
            err_handler.log_critical(config_options, self)
            raise

        data_type_flag = data_type_buffer[0]

        # gather buffer offsets and bounds to rank 0
        bounds = np.array(
            [
                np.int32(geo_meta.x_lower_bound),
                np.int32(geo_meta.y_lower_bound),
                np.int32(geo_meta.x_upper_bound),
                np.int32(geo_meta.y_upper_bound),
            ]
        )
        global_bounds = np.zeros((self.size * 4), np.int32)

        try:
            self.comm.Allgather([bounds, MPI.INTEGER], [global_bounds, MPI.INTEGER])
        except Exception as e:
            config_options.errMsg = f"Failed all gathering global bounds at rank {self.rank}: {e.__class__.__name__} -- {e}"
            err_handler.log_critical(config_options, self)
            raise

        # create slices for x and y bounds arrays
        x_lower = global_bounds[0 : (self.size * 4) + 0 : 4]
        y_lower = global_bounds[1 : (self.size * 4) + 1 : 4]
        x_upper = global_bounds[2 : (self.size * 4) + 2 : 4]
        y_upper = global_bounds[3 : (self.size * 4) + 3 : 4]

        # generate counts
        counts = [
            (y_upper[i] - y_lower[i]) * (x_upper[i] - x_lower[i])
            for i in range(0, self.size)
        ]

        # generate offsets:
        offsets = [0]
        for i in range(0, self.size - 1):
            offsets.append(offsets[i] + counts[i])

        # create the send buffer
        if self.rank == 0:
            sendbuf = np.empty([src_array.size], src_array.dtype)

            # fill the send buffer
            for i in range(0, self.size):
                start = offsets[i]
                stop = offsets[i] + counts[i]
                sendbuf[start:stop] = src_array[
                    y_lower[i] : y_upper[i], x_lower[i] : x_upper[i]
                ].flatten()
        else:
            sendbuf = None

        # create the recvbuffer
        if data_type_flag == 1:
            data_type = MPI.FLOAT
            recvbuf = np.empty([counts[self.rank]], np.float32)
        elif data_type_flag == 3:
            data_type = MPI.BOOL
            recvbuf = np.empty([counts[self.rank]], bool)
        else:
            data_type = MPI.DOUBLE
            recvbuf = np.empty([counts[self.rank]], np.float64)

        # scatter the data
        try:
            self.comm.Scatterv([sendbuf, counts, offsets, data_type], recvbuf, root=0)
        except Exception as e:
            config_options.errMsg = (
                f"Failed Scatterv from rank 0: {e.__class__.__name__} -- {e}"
            )
            err_handler.log_critical(config_options, self)
            raise

        subarray = np.reshape(
            recvbuf,
            [
                y_upper[self.rank] - y_lower[self.rank],
                x_upper[self.rank] - x_lower[self.rank],
            ],
        ).copy()
        return subarray

    def merge_slabs_gatherv(
        self, local_slab: np.ndarray, allgather: bool = False
    ) -> np.ndarray:
        """Gather arrays from all processes. The returned array will have the gathered data if `self.rank == 0` or `allgather` is `True`.

        The use of `allgather` is necessary for the hydrofabric case, to handle how ngen's hydrologic
        catchment partitionining differs from ESMF's arbitrary partitioning.

        :param local_slab: Data that will be gathered from all processes.
        :param allgather: Boolean on whether the gathered array should be broadcasted to all processes instead of just rank 0.
        :return: Numpy array of the data gathered from all processes.
        """
        # Filter based on dimensionality of array
        if len(local_slab.shape) == 2:
            # gather buffer offsets and bounds to rank 0 for 2d array
            shapes = np.array(
                [np.int32(local_slab.shape[0]), np.int32(local_slab.shape[1])]
            )
            global_shapes = np.zeros((self.size * 2), np.int32)
        else:
            # gather buffer offsets and bounds to rank 0 for 1d array
            shapes = np.array([np.int32(local_slab.shape[0])])
            global_shapes = np.zeros((self.size), np.int32)

        try:
            self.comm.Allgather([shapes, MPI.INTEGER], [global_shapes, MPI.INTEGER])
        except Exception:
            self.config_options.errMsg = (
                "Failed all gathering slab shapes at rank" + str(self.rank)
            )
            err_handler.log_critical(self.config_options, self)
            return None

        if len(local_slab.shape) == 2:
            # check that all slabes are the same width and sum the number of rows
            width = global_shapes[1]
            total_rows = 0
            for i in range(0, self.size):
                total_rows += global_shapes[2 * i]
                if global_shapes[(2 * i) + 1] != width:
                    self.config_options.errMsg = (
                        "Error: slabs with differing widths detected on slab for rank"
                        + str(i)
                    )
                    err_handler.log_critical(self.config_options, self)
                    return None

            # generate counts
            counts = [
                global_shapes[i * 2] * global_shapes[(i * 2) + 1]
                for i in range(0, self.size)
            ]

            # generate offsets:
            offsets = [0]
            for i in range(0, len(counts) - 1):
                offsets.append(offsets[i] + counts[i])

            # create the receive buffer
            if allgather or self.rank == 0:
                recvbuf = np.empty([total_rows, width], local_slab.dtype)
            else:
                recvbuf = None
        else:
            # generate counts
            counts = [global_shapes[i] for i in range(0, self.size)]

            # generate offsets:
            offsets = [0]
            for i in range(0, len(counts) - 1):
                offsets.append(offsets[i] + counts[i])

            # create the receive buffer
            if allgather or self.rank == 0:
                recvbuf = np.empty([sum(global_shapes)], local_slab.dtype)
            else:
                recvbuf = None

        # set the MPI data type
        data_type = MPI.BYTE
        if local_slab.dtype == np.float32:
            data_type = MPI.FLOAT
        elif local_slab.dtype == np.float64:
            data_type = MPI.DOUBLE
        elif data_type == np.int32:
            data_type = MPI.INT

        # get the data with Gatherv
        try:
            if allgather:
                self.comm.Allgatherv(
                    sendbuf=local_slab, recvbuf=[recvbuf, counts, offsets, data_type]
                )
            else:
                self.comm.Gatherv(
                    sendbuf=local_slab,
                    recvbuf=[recvbuf, counts, offsets, data_type],
                    root=0,
                )
        except Exception:
            self.config_options.errMsg = "Failed to Gatherv to rank 0 from rank " + str(
                self.rank
            )
            err_handler.log_critical(self.config_options, self)
            return None

        return recvbuf

    def __register_exit_handlers(self) -> None:
        """Register exit handlers for unhandled exceptions, signals, and regular exits.
        TODO: consider WCOSS gating.
        TODO: note that when non-0 ranks call Abort directly, the rank 0 exit handler is still invoked, at least in some cases,
        so there may be opportunities to streamline this further to have only rank 0 perform the cleanup. Would need to test
        against potential deadlock conditions to be sure (would need to confirm that a non-0 rank initiating an abort would cause
        rank 0 break out of a collective call if it happens to be waiting at one)."""
        # Exceptions
        sys.excepthook = self.__excepthook
        # Regular exits
        atexit.register(self.cleanup)
        # Signals
        for sig in self.__signals_handled():
            signal.signal(sig, self.__signal_handler)

    def __excepthook(self, ex_type, value, tb) -> None:
        """Custom excepthook which follows these steps:
        1. Call Python's built-in excepthook.
        2. Log .errMsg as CRITICAL (unless it is None).
        3. Cleanup.
        4. MPI Abort.

        To apply, set `sys.excepthook` to this method."""
        sys.__excepthook__(ex_type, value, tb)
        if self.config_options.errMsg is not None:
            err_handler.log_critical(
                self.config_options,
                self,
                msg=f"In excepthook, found errMsg = {repr(self.config_options.errMsg)}",
            )
        self.abort_with_cleanup(1)

    def __signal_handler(self, signum, frame) -> None:
        """Handle termination signals by cleaning up before exit."""
        ### Unregister the signal handler
        for s in self.__signals_handled():
            signal.signal(s, signal.SIG_DFL)
        ### Cleanup and re-send the original signal to itself
        # self.cleanup()
        # os.kill(os.getpid(), signum)
        ### Cleanup and abort directly
        self.abort_with_cleanup(signum)

    def __signals_handled(self) -> tuple[int]:
        """Return a tuple of signals to be handled by cleanup routine."""
        ### signal.valid_signals() contains many that are unrelated to stoppage / interruption / error.
        # sigs = [s for s in signal.valid_signals() if s not in (signal.SIGKILL, signal.SIGSTOP)]
        sigs = (
            signal.SIGINT,
            signal.SIGTERM,
            signal.SIGHUP,
            signal.SIGQUIT,
            signal.SIGSEGV,
            signal.SIGABRT,
            signal.SIGFPE,
            signal.SIGBUS,
            signal.SIGILL,
        )
        return sigs
