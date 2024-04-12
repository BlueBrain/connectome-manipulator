# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Testing helpers"""

import pytest


@pytest.fixture
def nodes():
    class FakeNode:
        config = {"morphologies_dir": "/foo/bar"}
        _population = None

    return (FakeNode(), FakeNode())
