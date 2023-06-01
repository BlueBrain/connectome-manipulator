"""Main CLI entry point for parallel processing"""

import logging
import os
import socket

import dask
from dask_mpi import initialize

from .cli import app, manipulate_connectome

logging.getLogger("distributed").setLevel(logging.WARNING)

_TMP = os.environ.get("SHMDIR", os.environ.get("TMPDIR", os.getcwd()))

dask.config.set(
    {
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": False,
        "distributed.worker.memory.pause": 0.8,
        "distributed.worker.memory.terminate": 0.95,
        "distributed.worker.use-file-locking": False,
        "distributed.worker.profile.interval": "1s",
        "distributed.worker.profile.cycle": "10s",
        "distributed.admin.tick.limit": "1m",
        "temporary-directory": _TMP,
    }
)

initialize(
    dashboard_address=f"{socket.getfqdn()}:8787",
    worker_class="distributed.Worker",
    worker_options={
        "local_directory": _TMP,
        "nthreads": 1,  # Avoids issues with non-thread-save RNG etc
    },
)

# This is a bit hackish, but seems to be a good way to change the default for the
# application: we always want to be parallel when starting our own distributed Dask
for param in manipulate_connectome.params:
    if param.name == "parallel":
        param.default = True

if __name__ == "__main__":
    app()  # pylint: disable=no-value-for-parameter
