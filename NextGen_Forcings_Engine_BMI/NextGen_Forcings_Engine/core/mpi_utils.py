import uuid
import numpy as np
import mpi4py

mpi4py.rc.threads = False
from mpi4py import MPI  # noqa: E402


def get_new_broadcasted_uid(comm: MPI.Comm | None = None) -> str:
    """Broadcast a random uint64 then return the hash of that. Used for generating a random string shared among all ranks."""
    if comm is None:
        comm = MPI.COMM_WORLD
    rand_uint64 = None
    if comm.rank == 0:
        rng = np.random.default_rng()
        rand_uint64 = rng.integers(0, 2**64, dtype=np.uint64)

    rand_uint64 = comm.bcast(rand_uint64, root=0)

    # uuid.UUID expects a built-in Python int. Convert the NumPy uint64
    uid_64bit_hex = uuid.UUID(int=int(rand_uint64)).hex
    assert len(uid_64bit_hex) == 32
    return uid_64bit_hex[16:]
