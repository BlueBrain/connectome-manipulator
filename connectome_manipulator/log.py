"""Customized logging"""
from datetime import datetime
import importlib
import logging
import os
import sys

import numpy as np


PROFILING_LOG_LEVEL = logging.INFO + 5
ASSERTION_LOG_LEVEL = logging.ERROR + 5


def info(msg, *args, **kwargs):  # pragma: no cover
    """Wrapper for info logging"""
    return logging.info(msg, *args, **kwargs)


def profiling(msg, *args, **kwargs):  # pragma: no cover
    """Wrapper for profiling logging"""
    return logging.log(PROFILING_LOG_LEVEL, msg, *args, **kwargs)


def warning(msg, *args, **kwargs):  # pragma: no cover
    """Wrapper for warning logging"""
    return logging.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):  # pragma: no cover
    """Wrapper for error logging"""
    return logging.error(msg, *args, **kwargs)


def log_assert(cond, msg):
    """Assertion with logging"""
    if not cond:
        logging.log(ASSERTION_LOG_LEVEL, msg)
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


def logging_init(log_path, name):
    """Initialize logger (with custom log level for profiling and assert with logging)."""
    # Configure logging
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Reload logging module, in case it has already been initialized before
    # [In future versions: logging.basicConfig(..., force=True, ...) supported instead!]
    importlib.reload(logging)

    # Initialize logging
    log_file = os.path.join(log_path, f'{name}.{datetime.today().strftime("%Y%m%dT%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(module)s] %(levelname)s: %(message)s")
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])

    # Add custom log levels
    logging.addLevelName(PROFILING_LOG_LEVEL, "PROFILING")
    logging.addLevelName(ASSERTION_LOG_LEVEL, "ASSERTION")

    return log_file
