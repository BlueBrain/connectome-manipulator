# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

from contextlib import redirect_stdout
import io
import os
import re
import logging

import pytest
from mock import patch

from utils import setup_tempdir
import connectome_manipulator.log as test_module


def test_log_assert():
    msg = "failing_assert_message"
    with patch("logging.log") as patched:
        with pytest.raises(AssertionError, match=msg):
            test_module.log_assert(False, msg)

        patched.assert_called()
        patched.reset_mock()

        test_module.log_assert(True, msg)
        patched.assert_not_called()


def test_logging_init():
    logname = "chronicle"
    with setup_tempdir(__name__) as tempdir:
        logdir = os.path.join(tempdir, "logs")
        test_module.setup_logging(logging.DEBUG)
        test_module.create_log_file(logdir, logname)

        assert os.path.isdir(logdir)

        # check directory has a file
        dir_listing = os.listdir(logdir)
        assert len(dir_listing) == 1

        # check that the file matches the naming policy
        assert re.match(rf"^{logname}.*\.log$", dir_listing[0])

        # check the log level is set correctly
        assert logging.getLogger().getEffectiveLevel() == logging.DEBUG
