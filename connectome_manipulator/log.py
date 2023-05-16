"""Customized logging"""
from time import strftime
import logging
import os
import sys

import numpy as np


_LOG_FORMAT = "[%(levelname)s] %(message)s"
_LOG_FORMAT_WITH_DATE = "[%(levelname)s] (%(asctime)s) %(message)s"
_DATE_FORMAT = "%b.%d %H:%M:%S"


def info(msg, *args, **kwargs):  # pragma: no cover
    """Wrapper for info logging"""
    return logging.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):  # pragma: no cover
    """Wrapper for warning logging"""
    return logging.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):  # pragma: no cover
    """Wrapper for error logging"""
    return logging.error(msg, *args, **kwargs)


def log_assert(cond, msg):
    """Assertion with logging"""
    if not cond:
        logging.log(logging.ERROR, msg)
    assert cond, msg


def data(filespec, **kwargs):
    """Data logging, i.e., writing data arrays given by kwargs to compressed .npz file

    WARNING: Existing files will be overwritten
    """
    if len(logging.root.handlers) == 0:
        warning("Data logging not possible!")
        return

    file_handler = logging.root.handlers[0]
    if not hasattr(file_handler, "baseFilename"):
        warning("Data logging not possible!")
        return

    base_name = os.path.splitext(file_handler.baseFilename)[0]
    if len(filespec) > 0:
        filespec = "." + filespec
    data_file = base_name + filespec + ".npz"
    if os.path.exists(data_file):
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
    os.makedirs(log_path, exist_ok=True)
    logfile = os.path.join(log_path, "{}_{}.log".format(name, strftime("%Y-%m-%d_%Hh%M")))
    fileh = logging.FileHandler(logfile, encoding="utf-8")
    fileh.setFormatter(logging.Formatter(_LOG_FORMAT_WITH_DATE, _DATE_FORMAT))
    logging.root.setLevel(logging.DEBUG)  # So that log files write everything
    logging.root.addHandler(fileh)
    return logfile
