# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Test that alternate morphologies work."""

import os
import pytest
import bluepysnap
from connectome_manipulator.connectome_manipulation.converters import EdgeWriter
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation
from utils import TEST_DATA_DIR


@pytest.fixture
def manipulation():
    m = Manipulation.get("conn_wiring")
    return m


def test_apply(manipulation):
    c = bluepysnap.Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata_h5.json"))
    edges = c.edges[c.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    tgt_ids = nodes[1].ids()
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)
    edges_table_empty = edges_table.loc[[]].copy()
    writer = EdgeWriter(None, edges_table_empty)
    m = manipulation(nodes, writer)
    with pytest.raises(bluepysnap.exceptions.BluepySnapError):
        m.morpho_helper.get(tgt_ids[0], extension="swc")
    m.morpho_helper.get(tgt_ids[0], extension="h5")
