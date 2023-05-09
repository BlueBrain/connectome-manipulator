"""Customized profling"""
import resource
import time
import os
import logging
from contextlib import ContextDecorator
import pandas as pd


logger_profiling = logging.getLogger(__name__)


class _ResourceProfiler:
    """CPU/Memory profiling class"""

    def __init__(self, enabled=False):
        """Class initialization."""
        self._enabled = enabled

        self._start_mem = None
        self._diff_mem = None
        self.total_mem = 0

        self._start_time = None
        self._diff_time = None
        self.total_time = 0

        logger_profiling.setLevel(logging.INFO)

    def start(self):
        """Start profiling."""
        if not self._enabled:
            return

        self._start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        self._start_time = time.perf_counter()

    def stop(self):
        """Stop profiling."""
        if not self._enabled:
            return
        self._diff_time = time.perf_counter() - self._start_time
        self.total_time += self._diff_time
        self._start_time = None  # invalidate start time

        self._diff_mem = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2 - self._start_mem
        )
        self.total_mem += self._diff_mem
        self._start_mem = None  # invalidate start memory


class _ProfilerManager:
    """CPU/Memory profiling manager class"""

    def __init__(self):
        self.profilers = {}
        self._enable = False
        self._perf_table = None
        self._csv_file = None

    def init_perf_table(self, csv_file=None):
        """Initialize the performance table."""
        self._perf_table = pd.DataFrame(
            [],
            columns=["label", "time", "memory"],
        )
        self._perf_table.index.name = "id"
        if csv_file is None:
            return
        if not os.path.exists(os.path.split(csv_file)[0]):
            os.makedirs(os.path.split(csv_file)[0])
        self._csv_file = csv_file

    def init(self, name):
        """Starts profiling"""
        self.profilers.setdefault(name, _ResourceProfiler(enabled=self._enable))
        self.profilers[name].start()

    def update(self, name):
        """Stops profiling"""
        if name not in self.profilers:
            raise KeyError("{} not initialized in profilers dict".format(name))
        self.profilers[name].stop()

    def show_stats(self):
        """Logs profiling stats"""
        if not self._enable:
            return
        delim = "\u255a"
        stats_name = " PROFILER STATS "
        logger_profiling.info("+{:=^80s}+".format(stats_name))
        logger_profiling.info(
            "|{:^58s}|{:^10s}|{:^10s}|".format("Event Label", "Time (s)", "Mem. (MB)")
        )
        logger_profiling.info("+{:-^80s}+".format("-"))

        for name, pinfo in self.profilers.items():
            base_name = delim.join("  ") * name.count(delim) + name.split(delim)[-1]
            logger_profiling.info(
                "| {:<56s} | {:8.2f} | {:8.2f} |".format(
                    base_name, pinfo.total_time, pinfo.total_mem
                )
            )
            self._perf_table.loc[self._perf_table.shape[0]] = [
                name,
                pinfo.total_time,
                pinfo.total_mem,
            ]
        self.write_to_csv()
        logger_profiling.info("+{:-^80s}+".format("-"))

    def write_to_csv(self):
        """Add entry to performance table and write to .csv file."""
        if self._csv_file is None:
            return

        self._perf_table.to_csv(self._csv_file)

    def set_enabled(self, enable):
        """Enables profiling"""
        self._enable = enable

    def merge(self, profiler_manager):
        """Merges profile managers"""
        self.profilers.update(profiler_manager.profilers)


ProfilerManager = _ProfilerManager()  # singleon


# Can be used as context manager or decorator
class profileit(ContextDecorator):
    """Context manager or decorator of for profiling"""

    def __init__(self, name):
        """Initialization"""
        self._name = name

    def __enter__(self):
        """Enter the context and starts profiling"""
        ProfilerManager.init(self._name)

    def __exit__(self, exc_type, exc, exc_tb):
        """Leave the context and stops profiling"""
        ProfilerManager.update(self._name)
