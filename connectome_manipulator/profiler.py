# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

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
        self.diff_mem = None

        self._start_time = None
        self.diff_time = None

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
        self.diff_time = time.perf_counter() - self._start_time
        self._start_time = None  # invalidate start time

        self.diff_mem = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2 - self._start_mem
        )
        self._start_mem = None  # invalidate start memory


class _ProfilerManager:
    """CPU/Memory profiling manager class"""

    def __init__(self):
        self.profilers = {}
        self._enable = False
        self.perf_table = None
        self._csv_file = None
        self._parent_labels = []

        self.init_perf_table()
        logger_profiling.setLevel(logging.INFO)

    def init_perf_table(self):
        """Initialize the performance table."""
        self.perf_table = pd.DataFrame(
            [],
            columns=["label", "time", "memory", "parent_labels"],
        )
        self.perf_table.index.name = "id"

    def set_csv_file(self, csv_file=None):
        """Set the profiling output csv file"""
        if csv_file is None:
            return
        if not os.path.exists(os.path.split(csv_file)[0]):
            os.makedirs(os.path.split(csv_file)[0])
        self._csv_file = csv_file

    def start(self, name):
        """Starts profiling"""
        self._parent_labels.append(name)
        self.profilers.setdefault(name, []).append(_ResourceProfiler(enabled=self._enable))
        self.profilers[name][-1].start()

    def stop(self, name):
        """Stops profiling"""
        if name not in self.profilers:
            raise KeyError("{} not initialized in profilers dict".format(name))
        pinfo = self.profilers[name][-1]
        pinfo.stop()
        self._parent_labels.pop()
        # Update the performance table with the new row of profiling data
        self.perf_table.loc[self.perf_table.shape[0]] = [
            name,
            pinfo.diff_time,
            pinfo.diff_mem,
            ":".join(self._parent_labels),
        ]

    def show_stats(self):
        """Logs profiling stats"""
        if not self._enable:
            return

        # Simplify perf table and add min, max, avg
        self.perf_table = (
            self.perf_table.groupby("label")
            .agg(
                {
                    "time": ["min", "mean", "max"],
                    "memory": ["min", "mean", "max"],
                    "parent_labels": "first",
                }
            )
            .reset_index()
        )

        delim = "\u255a"
        stats_name = " PROFILER STATS "
        logger_profiling.info("+{:=^110s}+".format(stats_name))
        logger_profiling.info(
            "|{:^44s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|{:^10s}|".format(
                "Event Label", "Min.Time", "Avg.Time", "Max.Time", "Min.Mem", "Avg.Mem", "Max.Mem"
            )
        )
        logger_profiling.info("+{:-^110s}+".format("-"))

        # Loop through the labels in order
        for label_name in self.profilers:
            # Access the row in perf_table based on the label
            row = self.perf_table.loc[self.perf_table["label"] == label_name]
            label = row["label"].item().strip()
            time_min = row[("time", "min")].item()
            time_mean = row[("time", "mean")].item()
            time_max = row[("time", "max")].item()
            mem_min = row[("memory", "min")].item()
            mem_mean = row[("memory", "mean")].item()
            mem_max = row[("memory", "max")].item()
            parents = row[("parent_labels", "first")].item()
            num_tabs = len(parents.split(":")) if parents else 0
            base_name = (
                "  " * num_tabs + delim.join("  ") * label.count(delim) + label.split(delim)[-1]
            )
            logger_profiling.info(
                "| {:<42s} | {:8.2f} | {:8.2f} | {:8.2f} | {:8.2f} | {:8.2f} | {:8.2f} |".format(
                    base_name, time_min, time_mean, time_max, mem_min, mem_mean, mem_max
                )
            )
        self.write_to_csv()
        logger_profiling.info("+{:-^110s}+".format("-"))

    def write_to_csv(self):
        """Add entry to performance table and write to .csv file."""
        if self._csv_file is None:
            return

        self.perf_table.to_csv(self._csv_file, index=False)

    def set_enabled(self, enable):
        """Enables profiling"""
        self._enable = enable

    def merge(self, profiler_manager):
        """Merges profile managers"""
        if not self._enable:
            return

        self.profilers.update(profiler_manager.profilers)
        if self.perf_table is not None and profiler_manager.perf_table is not None:
            self.perf_table = pd.concat([self.perf_table, profiler_manager.perf_table])
        elif profiler_manager.perf_table is not None:
            self.perf_table = profiler_manager.perf_table


ProfilerManager = _ProfilerManager()  # singleon


# Can be used as context manager or decorator
class profileit(ContextDecorator):
    """Context manager or decorator of for profiling"""

    def __init__(self, name):
        """Initialization"""
        self._name = name

    def __enter__(self):
        """Enter the context and starts profiling"""
        ProfilerManager.start(self._name)

    def __exit__(self, exc_type, exc, exc_tb):
        """Leave the context and stops profiling"""
        ProfilerManager.stop(self._name)
