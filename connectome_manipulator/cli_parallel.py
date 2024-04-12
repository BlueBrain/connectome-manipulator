# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Main CLI entry point for parallel processing"""

import logging
import os
import resource
import socket

import dask
from dask_mpi import initialize

from .cli import app, manipulate_connectome

logging.getLogger("distributed").setLevel(logging.WARNING)


def setup():
    """Dask parallel initialization

    This allows us to bail out early if we are not the root MPI rank that will start the scheduler.
    """
    # Dask likes to eat file descriptors for breakfast, lunch, diner, so give it the most we
    # can get
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

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

    # Using 1 thread avoids issues with non-thread-save RNG etc, we are modifying the Dask
    # process pool in executors.py to use a process pool, not a thread pool.
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    dash_url = f"{socket.getfqdn()}:8787"

    is_client = initialize(
        dashboard_address=dash_url,
        worker_class="distributed.Worker",
        worker_options={
            "local_directory": _TMP,
            "nthreads": threads,
        },
        exit=True,
    )

    if is_client:
        print("Dashboard URL:", dash_url)

        # This is a bit hackish, but seems to be a good way to change the default for the
        # application: we always want to be parallel when starting our own distributed Dask
        for param in manipulate_connectome.params:
            if param.name == "parallel":
                param.default = True

        if __name__ == "__main__":
            app()  # pylint: disable=no-value-for-parameter


setup()
