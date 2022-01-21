from contextlib import redirect_stdout
import io
import os
import re

import pytest
from mock import patch

from utils import setup_tempdir
import connectome_manipulator.log as test_module


def test_log_assert():
    msg = 'failing_assert_message'
    with patch('logging.log') as patched:
        with pytest.raises(AssertionError, match=msg):
            test_module.log_assert(False, msg)

        patched.assert_called()
        patched.reset_mock()

        test_module.log_assert(True, msg)
        patched.assert_not_called()


def test_logging_init():
    logname = 'chronicle'
    with setup_tempdir(__name__) as tempdir:
        test_module.logging_init(tempdir, logname)
        logdir = os.path.join(tempdir, 'logs')

        assert os.path.isdir(logdir)

        # check directory has a file
        dir_listing = os.listdir(logdir)
        assert len(dir_listing) == 1

        # check that the file matches the naming policy
        assert re.match(rf'^{logname}.*\.log$', dir_listing[0])

        # check that PROFILING log level exists
        assert test_module.logging.getLevelName(test_module.logging.INFO + 5) == 'PROFILING'
