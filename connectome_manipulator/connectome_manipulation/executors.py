"""A module implementing several executor wrappers"""
from contextlib import contextmanager

from .. import log


class ExecutorWrapper:
    """A generic executor wrapper, unifying submit() API"""

    def __init__(self, executor) -> None:
        """Initializes the executor wrapper"""
        self._executor = executor
        self.jobs = []

    def submit(self, func, args, extra_data):
        """Submits a new routine to be run by the distributed framework"""
        job = self._executor.submit(func, *args)
        job.extra_data = extra_data  # abuse the future object with custom data
        self.jobs.append(job)
        return job

    def await_results(self) -> list:
        """Yields results and respective metadata as jobs finish"""
        from dask.distributed import as_completed

        for job in as_completed(self.jobs):
            yield job.result(), job.extra_data


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


# NOTE: Interestingly, Dask and Submitit share the same fundamental API.
#       We may however want to specialize this later for better performance


class DaskExecutor(ExecutorWrapper):
    """An Executor wrapper for SubmitIt"""


class SubmititExecutor(ExecutorWrapper):
    """An Executor wrapper for SubmitIt"""


@contextmanager
def serial_ctx(result_hook):
    """A plain serial executor, basically no-op"""
    yield SerialExecutor(result_hook)
    # DONE!


@contextmanager
def dask_ctx(_options, result_hook, executor_params: dict, **_kw):
    """An executor using the Dask system"""
    from dask.distributed import Client

    # Dask requires numeric params to go as the native type
    for k, val in executor_params.items():
        if val.isdecimal():
            try:
                executor_params[k] = int(val)
            except ValueError:
                executor_params[k] = float(val)

    # By default use processes
    executor_params.setdefault("processes", True)

    client = Client(**executor_params)
    executor_wrapper = DaskExecutor(client)

    yield executor_wrapper

    log.info("Jobs submitted to DASK")

    if result_hook:
        # Immediately await and process results
        for result, info in executor_wrapper.await_results():
            result_hook(result, info)


@contextmanager
def submitit_ctx(options, result_hook, *, executor_params, **_kw):
    """An executor using the SubmitIt system"""
    import submitit

    job_logs = str(options.logging_path) + "/%j"

    executor = submitit.AutoExecutor(folder=job_logs)
    executor.update_parameters(
        slurm_array_parallelism=options.max_parallel_jobs,
        slurm_partition="prod",
        name="connectome_manipulator",
        timeout_min=120,
    )
    extra_args = {"slurm_" + k: v for k, v in executor_params.items()}
    executor.update_parameters(**extra_args)

    log.info("SLURM args: %s", extra_args)

    # FIXME: There is a known corner-case when 1 split is chosen and parallelization is enabled.
    # This should be addressed.

    executor_wrapper = SubmititExecutor(executor)

    with executor.batch():
        yield executor_wrapper  # wrapper has unified API

    log.info("Jobs submitted to Slurm")

    if result_hook:
        # Await and process results
        for result, info in executor_wrapper.await_results():
            result_hook(result, info)


def in_context(options, params, result_hook=None):
    """An auto-selector of the executor context manager"""
    log.info("Starting Execution context")
    if options.parallel is None:
        return serial_ctx(result_hook)
    if options.parallel == "Dask":
        return dask_ctx(options, result_hook, **params)
    if options.parallel == "Slurm":
        return submitit_ctx(options, result_hook, **params)

    raise NotImplementedError(f"No such executor: {options.parallel}")
