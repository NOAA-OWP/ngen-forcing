import uuid

import mpi4py

mpi4py.rc.threads = False
from mpi4py import MPI

import numpy as np


# def get_new_broadcasted_uid(comm: MPI.Comm) -> str:
def get_new_broadcasted_uid() -> str:
    """Broadcast a random uint64 then return the hash of that. Used for generating a random string shared among all ranks."""
    # if not isinstance(comm, MPI.Comm):
    #     raise TypeError(f"Expected comm to be type MPI.Comm, got: {type(comm)}")
    rand_uint64 = None
    if MPI.COMM_WORLD.rank == 0:
        rng = np.random.default_rng()
        rand_uint64 = rng.integers(0, 2**64, dtype=np.uint64)
    else:
        rand_uint64 = None

    rand_uint64 = MPI.COMM_WORLD.bcast(rand_uint64, root=0)

    # Since based on 64-bit int, first 16 chars are 0, final 16 chars are random
    uid_64bit_hex = uuid.UUID(int=rand_uint64).hex
    assert len(uid_64bit_hex) == 32
    uid64 = uid_64bit_hex[16:]
    return uid64
