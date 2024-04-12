# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Test Dask auxiliary functions"""

from connectome_manipulator.processing import BatchInfo


def test_grouping():
    """Test grouping of batches for target payload"""
    batches = [
        BatchInfo(3, {}, [0]),
        BatchInfo(29, {}, [1]),
        BatchInfo(10, {}, [2]),
        BatchInfo(10, {}, [3]),
        BatchInfo(10, {}, [4]),
        BatchInfo(3, {}, [5]),
    ]

    groups = BatchInfo.group_batches(batches, 30)
    assert len(groups) == 4

    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert len(groups[2]) == 3
    assert len(groups[3]) == 1

    assert [b.node_ids[0] for b in groups[0]] == [0]
    assert [b.node_ids[0] for b in groups[1]] == [1]
    assert [b.node_ids[0] for b in groups[2]] == [2, 3, 4]
    assert [b.node_ids[0] for b in groups[3]] == [5]
