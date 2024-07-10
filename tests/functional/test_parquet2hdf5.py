# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import os
import subprocess


def test_parquet2hdf5():
    """Test if parquet2hdf5 function is available."""
    try:
        with subprocess.Popen(
            ["parquet2hdf5", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={"PATH": os.getenv("PATH", "")},
        ) as proc:
            print("parquet2hdf5 v" + proc.communicate()[0].decode())
    except FileNotFoundError as e:
        assert False, str(e)
