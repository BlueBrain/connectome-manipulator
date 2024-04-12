# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""A module implementing several executor wrappers"""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
from distributed import as_completed, WorkerPlugin

from .. import log


class DaskExecutor:
    """A executor wrapper for Dask"""

    _PROCESS_INTERVAL = timedelta(minutes=1)

    def __init__(self, executor, result_hook=None) -> None:
        """Initializes the executor wrapper"""
        self._executor = executor
        self._result_hook = result_hook
        self._jobs = []
        self._last_processing = datetime.now()

    def _timed_process_jobs(self) -> bool:
        if datetime.now() - self._last_processing > self._PROCESS_INTERVAL:
            self.process_jobs(self._jobs)
            # Use the time after processing to get an evenly timed job submission window
            self._last_processing = datetime.now()

    def submit(self, func, args, extra_data):
        """Submits a new routine to be run by the distributed framework"""
        job = self._executor.submit(func, *args)
        job.extra_data = extra_data
        self._timed_process_jobs()
        self._jobs.append(job)

    def process_jobs(self, jobs=None):
        """Process completed jobs, blocking to wait for all jobs if none are passed"""
        if not jobs:
            jobs = as_completed(self._jobs)
        self._jobs = []
        for job in jobs:
            if job.done():
                if self._result_hook:
                    self._result_hook(job.result(), job.extra_data)
                job.release()
            else:
                self._jobs.append(job)


class SerialExecutor:
    """The serial executor wrapper, which immediately runs the user function"""

    def __init__(self, result_hook) -> None:
        """Initializes the serial executor wrapper"""
        self._result_hook = result_hook

    def submit(self, func, args, extra_data):
        """Submits a new routine (which is run immediately)"""
        result = func(*args)  # Run it inplace
        if self._result_hook:
            self._result_hook(result, extra_data)
        return result


@contextmanager
def serial_ctx(result_hook):
    """A plain serial executor, basically no-op"""
    yield SerialExecutor(result_hook)
    # DONE!


# This allegedly provides a bit safer support for separate processes, skip for now to keep
# dependencies at bay
# from loky import ProcessPoolExecutor
class AddProcessPool(WorkerPlugin):
    """Helper to avoid threading and use processes instead"""

    def setup(self, worker):
        """Configures the worker"""
        if worker.state.nthreads > 1:
            worker.executors["default"] = ProcessPoolExecutor(max_workers=worker.state.nthreads)


@contextmanager
def dask_ctx(result_hook, executor_params: dict):
    """An executor using the Dask system"""
    from dask.distributed import Client

    # Dask requires numeric params to go as the native type
    for k, val in executor_params.items():
        if val.isdecimal():
            try:
                executor_params[k] = int(val)
            except ValueError:
                executor_params[k] = float(val)

    with Client(**executor_params) as client:
        client.register_worker_plugin(AddProcessPool())
        executor_wrapper = DaskExecutor(client, result_hook)

        yield executor_wrapper

        log.info("Jobs submitted to DASK")

        executor_wrapper.process_jobs()

        log.info("DASK jobs finished")

        log.info("Shutting down DASK workers gracefully")
        client.retire_workers()
        # In some cases, we seem to need to shut down the client: while using dask-mpi
        # though, it will attempt to shut down the client itself. If the code below is
        # activated while dask-mpi is used, exceptions will be displayed.
        # time.sleep(1)
        # client.shutdown()


def in_context(options, params, result_hook=None):
    """An auto-selector of the executor context manager"""
    log.info("Starting Execution context")
    if options.parallel:
        return dask_ctx(result_hook, params)
    return serial_ctx(result_hook)
