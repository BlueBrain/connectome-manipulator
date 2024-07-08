# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Customized logging"""

import logging
import os
import sys
from glob import glob
from time import strftime

import numpy as np

import connectome_manipulator

_LOG_FORMAT = "[%(levelname)s] %(message)s"
_LOG_FORMAT_WITH_DATE = "[%(levelname)s] (%(asctime)s) %(message)s"
_DATE_FORMAT = "%b.%d %H:%M:%S"

_start_time = None


def debug(msg, *args, **kwargs):  # pragma: no cover
    """Wrapper for debug logging"""
    return logging.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):  # pragma: no cover
    """Wrapper for info logging"""
    return logging.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):  # pragma: no cover
    """Wrapper for warning logging"""
    return logging.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):  # pragma: no cover
    """Wrapper for error logging"""
    return logging.error(msg, *args, **kwargs)


def exception(msg, *args, **kwargs):  # pragma: no cover
    """Wrapper for error logging"""
    return logging.exception(msg, *args, **kwargs)


def log_assert(cond, msg):
    """Assertion with logging"""
    if not cond:
        logging.log(logging.ERROR, msg)
    assert cond, msg


def data(filespec, **kwargs):
    """Data logging, i.e., writing data arrays given by kwargs to compressed .npz file

    WARNING: Existing files will be overwritten
    """
    file_handlers = [hdl for hdl in logging.root.handlers if hasattr(hdl, "baseFilename")]
    if len(file_handlers) == 0:
        warning("Data logging not possible!")
        return

    file_handler = file_handlers[0]
    base_name = os.path.splitext(file_handler.baseFilename)[0]
    if len(filespec) > 0:
        filespec = "." + filespec
    data_file = base_name + filespec + ".npz"
    if os.path.exists(data_file):
        # File may exist, e.g., in case of multiple rewiring functions.
        # Therefore, extend filename to prevent overwriting.
        p, df = os.path.split(data_file)
        fn, ext = os.path.splitext(df)
        file_list = list(glob(fn + ".*" + ext, root_dir=p))
        fidx = len(file_list) + 1
        data_file = os.path.join(p, fn + f".{fidx}" + ext)
        if os.path.exists(data_file):
            # Check again if for any reason this file still already exists
            warning(f'Data log file "{data_file}" already exists and will be overwritten!')
    np.savez_compressed(data_file, **kwargs)
    info(f'Data log ({", ".join(list(kwargs.keys()))}) written to "{data_file}"')


class LogLevel:
    """Class to select the log level."""

    ERROR_ONLY = 0
    DEFAULT = 1
    DEBUG = 2


def setup_logging(log_level=LogLevel.DEFAULT):
    """Features tabs and colors output to stdout

    Args:
      log_level (int): minimum log level for emitting messages
    """
    assert isinstance(log_level, int)
    log_level = min(log_level, 2)

    verbosity_levels = [
        logging.WARNING,  # pos 0: Minimum possible logging level
        logging.INFO,
        logging.DEBUG,
    ]

    # Stdout
    hdlr = logging.StreamHandler(sys.stdout)
    hdlr.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    hdlr.setLevel(verbosity_levels[log_level])
    logging.root.setLevel(verbosity_levels[log_level])

    del logging.root.handlers[:]
    logging.root.addHandler(hdlr)


def create_log_file(log_path, name):
    """Create the log file

    Args:
      log_path: The destination directory for log messages besides stdout
      name: Name of the module to log
    """
    global _start_time  # pylint: disable=global-statement
    os.makedirs(log_path, exist_ok=True)
    if not _start_time:
        _start_time = strftime("%Y-%m-%d_%Hh%M")
    logfile = os.path.join(log_path, "{}_{}.log".format(name, _start_time))
    fileh = logging.FileHandler(logfile, encoding="utf-8")
    fileh.setFormatter(logging.Formatter(_LOG_FORMAT_WITH_DATE, _DATE_FORMAT))
    logging.root.setLevel(logging.DEBUG)  # So that log files write everything
    remove = [h for h in logging.root.handlers if isinstance(h, logging.FileHandler)]
    for h in remove:
        logging.root.removeHandler(h)
    logging.root.addHandler(fileh)
    info(f'Log file "{logfile}" created!')
    info(f"Version: connectome_manipulator {connectome_manipulator.__version__}")
    return logfile
