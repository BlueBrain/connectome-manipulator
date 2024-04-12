# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import os
import pytest

from bluepysnap import Circuit

from utils import TEST_DATA_DIR
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation
from connectome_manipulator.connectome_manipulation.converters import EdgeWriter
from connectome_manipulator import log


@pytest.fixture
def manipulation():
    m = Manipulation.get("null_manipulation")
    return m


def test_apply(manipulation):
    log.setup_logging()  # To have data logging in a defined state

    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = circuit.edges[circuit.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    tgt_ids = nodes[1].ids()
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)

    writer = EdgeWriter(None, existing_edges=edges_table.copy())
    manipulation(nodes, writer).apply(tgt_ids)
    res = writer.to_pandas()
    assert res.equals(edges_table)
