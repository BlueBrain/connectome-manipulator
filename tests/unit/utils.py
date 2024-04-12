# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import os
import sys
import shutil
import tempfile
from contextlib import contextmanager
from types import ModuleType

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


@contextmanager
def setup_tempdir(prefix):
    """Create a temporary dir to be removed after use"""
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


def create_fake_module(module_name, code):
    """Used to test code that uses dynamically imported modules"""
    module = ModuleType(module_name)
    exec(code, module.__dict__)
    sys.modules[module_name] = module

    return module
