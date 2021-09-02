'''Customized logging'''
from datetime import datetime
import logging
import os
import sys


PROFILING_LOG_LEVEL = logging.INFO + 5

def profiling(msg, *args, **kwargs):
    '''Wrapper for profiling logging'''
    return logging.log(PROFILING_LOG_LEVEL, msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    '''Wrapper for warning logging'''
    return logging.warning(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    '''Wrapper for info logging'''
    return logging.info(msg, *args, **kwargs)


def log_assert(cond, msg):
    '''Add assert with logging'''
    if not cond:
        logging.error(msg)
    assert cond, msg


def logging_init(output_path, name):
    """Initialize logger (with custom log level for profiling and assert with logging)."""
    # Configure logging
    log_path = os.path.join(output_path, 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file = os.path.join(log_path, f'{name}.{datetime.today().strftime("%Y%m%dT%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(module)s] %(levelname)s: %(message)s'))
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])

    # Add custom log level for profiling
    logging.addLevelName(PROFILING_LOG_LEVEL, 'PROFILING')
