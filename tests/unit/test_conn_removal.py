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
    m = Manipulation.get("conn_removal")
    return m


def test_apply(manipulation):
    log.setup_logging()  # To have data logging in a defined state

    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = circuit.edges[circuit.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)

    # Test that given percentage of connections is removed
    total_conns = len(np.unique(edges_table[["@source_node", "@target_node"]], axis=0))
    for pct in range(0, 101, 25):
        writer = EdgeWriter(None, existing_edges=edges_table.copy())
        manipulation(nodes, writer).apply(tgt_ids, amount_pct=pct)
        res = writer.to_pandas()
        res_conns = len(np.unique(res[["@source_node", "@target_node"]], axis=0))
        assert res_conns == int(total_conns * (100 - pct) / 100)

    # Check that only given ids are considered
    src_id = 0
    writer = EdgeWriter(None, existing_edges=edges_table.copy())
    manipulation(nodes, writer).apply(tgt_ids, sel_src={*src_ids} - {src_id}, amount_pct=100)
    res = writer.to_pandas()
    assert np.all(res["@source_node"] == src_id)

    tgt_id = 9
    writer = EdgeWriter(None, existing_edges=edges_table.copy())
    manipulation(nodes, writer).apply(tgt_ids, sel_dest={*tgt_ids} - {tgt_id}, amount_pct=100)
    res = writer.to_pandas()
    assert np.all(res["@target_node"] == tgt_id)

    # Check when both sel_src and sel_dest are given
    writer = EdgeWriter(None, existing_edges=edges_table.copy())
    manipulation(nodes, writer).apply(
        tgt_ids, sel_src={*src_ids} - {src_id}, sel_dest={*tgt_ids} - {tgt_id}, amount_pct=100
    )
    res = writer.to_pandas()
    assert np.all(np.logical_or(res["@source_node"] == src_id, res["@target_node"] == tgt_id))

    # Check that connection size filtering works for minimum connection size
    writer = EdgeWriter(None, existing_edges=edges_table.copy())
    manipulation(nodes, writer).apply(tgt_ids, min_syn_per_conn=2, amount_pct=100)
    res = writer.to_pandas()
    _, n_syn_conn = np.unique(res[["@source_node", "@target_node"]], axis=0, return_counts=True)
    assert np.all(n_syn_conn < 2)

    # Check that connection size filtering works for maximum connection size
    writer = EdgeWriter(None, existing_edges=edges_table.copy())
    manipulation(nodes, writer).apply(tgt_ids, max_syn_per_conn=2, amount_pct=100)
    res = writer.to_pandas()
    _, n_syn_conn = np.unique(res[["@source_node", "@target_node"]], axis=0, return_counts=True)
    assert np.all(n_syn_conn > 2)

    # Check that connection size filtering works when both are enabled
    writer = EdgeWriter(None, existing_edges=edges_table.copy())
    manipulation(nodes, writer).apply(
        tgt_ids, min_syn_per_conn=2, max_syn_per_conn=2, amount_pct=100
    )
    res = writer.to_pandas()
    _, n_syn_conn = np.unique(res[["@source_node", "@target_node"]], axis=0, return_counts=True)
    assert np.all(n_syn_conn != 2)

    # Check with empty selection, i.e. nothing to be removed
    writer = EdgeWriter(None, existing_edges=edges_table.copy())
    manipulation(nodes, writer).apply(tgt_ids, min_syn_per_conn=np.inf, amount_pct=100)
    res = writer.to_pandas()
    _, n_syn_conn = np.unique(res[["@source_node", "@target_node"]], axis=0, return_counts=True)
    assert res.equals(edges_table)
