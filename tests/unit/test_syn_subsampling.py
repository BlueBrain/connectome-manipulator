# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import numpy as np
import os
import pandas as pd
import pytest

from bluepysnap import Circuit

from utils import TEST_DATA_DIR
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation
from connectome_manipulator.connectome_manipulation.converters import EdgeWriter
from connectome_manipulator import log


@pytest.fixture
def manipulation():
    m = Manipulation.get("syn_subsampling")
    return m


def test_apply(manipulation):
    log.setup_logging()  # To have data logging in a defined state

    c = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = c.edges[c.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)

    for pct in [10.1, 50.99999, 0.666, 99.9999]:
        writer = EdgeWriter(None, existing_edges=edges_table.copy())
        manipulation(nodes, writer).apply(tgt_ids, keep_pct=pct)
        res = writer.to_pandas()

        assert res.shape[0] == np.round(pct * edges_table.shape[0] / 100)
        assert np.all(
            [np.any(np.all(res.iloc[i] == edges_table, 1)) for i in range(res.shape[0])]
        )  # Check if all rows are contained
