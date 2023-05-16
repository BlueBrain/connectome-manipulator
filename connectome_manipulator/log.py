"""Customized logging"""
from time import strftime
import logging
import os
import sys

import numpy as np
from connectome_manipulator import utils


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


class _LevelColorFormatter(logging.Formatter):
    COLORS = {
        logging.CRITICAL: utils.ConsoleColors.RED,
        logging.ERROR: utils.ConsoleColors.RED,
        logging.WARNING: utils.ConsoleColors.YELLOW,
        logging.INFO: utils.ConsoleColors.BLUE,
        logging.DEBUG: utils.ConsoleColors.DEFAULT + utils.ConsoleColors.DIM,
    }

    _logfmt = "[%(levelname)s] %(message)s"
    _datefmt = "%b.%d %H:%M:%S"

    def __init__(self, with_time=True, use_color=True, **kw):
        super().__init__(self._logfmt, self._datefmt, **kw)
        self._with_time = with_time
        self._use_color = use_color

    def format(self, record):
        if hasattr(record, "ulevel"):
            record.levelno = record.ulevel
            record.levelname = logging.getLevelName(record.levelno)
        style = self.COLORS.get(record.levelno)
        if style is not None:
            record.levelname = self._format_level(record, style)
            record.msg = self._format_msg(record, style)
        return super().format(record)

    def _format_level(self, record, style):
        if not self._use_color:
            return record.levelname
        return utils.ConsoleColors.format_text(record.levelname, style)

    def _format_msg(self, record, style):
        msg = ""
        if self._with_time:
            msg += "(%s) " % self.formatTime(record, self._datefmt) + msg

        levelno = record.levelno
        msg += record.msg

        if not self._use_color:
            return msg
        return (
            utils.ConsoleColors.format_text(msg, style)
            if levelno >= logging.WARNING
            else utils.ConsoleColors.format_text(msg, utils.ConsoleColors.DEFAULT, style)
        )


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
    use_color = True
    if os.environ.get("ENVIRONMENT") == "BATCH":
        use_color = False
    else:
        try:
            sys.stdout.tell()  # works only if it's file
            use_color = False
        except IOError:
            pass
    hdlr.setFormatter(_LevelColorFormatter(False, use_color))
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
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logfile = os.path.join(log_path, "{}_{}".format(name, strftime("%Y-%m-%d_%Hh%M")))
    fileh = logging.FileHandler(logfile + ".log", encoding="utf-8")
    fileh.setFormatter(_LevelColorFormatter(use_color=False))
    logging.root.setLevel(logging.DEBUG)  # So that log files write everything
    logging.root.addHandler(fileh)
    return logfile
