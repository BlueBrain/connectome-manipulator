# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import json
import numpy as np
import os
import pytest

from bluepysnap import Circuit

from utils import TEST_DATA_DIR
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation
from connectome_manipulator.connectome_manipulation.converters import EdgeWriter
from connectome_manipulator import log


@pytest.fixture
def manipulation():
    m = Manipulation.get("conn_extraction")
    return m


def test_apply(manipulation):
    log.setup_logging()  # To have data logging in a defined state

    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = circuit.edges[circuit.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    tgt_ids = nodes[1].ids()
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)

    # Test that given intrinsic cell target is extracted (no extra node sets file given)
    for tgt_name in ["LayerA", "RegionB"]:
        writer = EdgeWriter(None, existing_edges=edges_table.copy())
        manipulation(nodes, writer).apply(tgt_ids, target_name=tgt_name, node_sets_file=None)
        res = writer.to_pandas()
        src_ids = nodes[0].ids(tgt_name)
        tgt_ids = nodes[1].ids(tgt_name)
        assert np.all(np.isin(res["@source_node"], src_ids)) and np.all(
            np.isin(res["@target_node"], tgt_ids)
        )
        assert (
            np.sum(
                np.logical_and(
                    np.isin(edges_table["@source_node"], src_ids),
                    np.isin(edges_table["@target_node"], tgt_ids),
                )
            )
            == res.shape[0]
        )

    # Test that given external cell target is extracted (given by extra node sets file)
    node_sets_file = os.path.join(TEST_DATA_DIR, "node_sets_extra.json")
    with open(node_sets_file, "r") as f:
        node_sets = json.load(f)
    for tgt_name in ["NSet1", "NSet1"]:
        writer = EdgeWriter(None, existing_edges=edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids, target_name=tgt_name, node_sets_file=node_sets_file
        )
        res = writer.to_pandas()
        src_ids = node_sets[tgt_name]["node_id"]
        tgt_ids = node_sets[tgt_name]["node_id"]
        assert np.all(np.isin(res["@source_node"], src_ids)) and np.all(
            np.isin(res["@target_node"], tgt_ids)
        )
        assert (
            np.sum(
                np.logical_and(
                    np.isin(edges_table["@source_node"], src_ids),
                    np.isin(edges_table["@target_node"], tgt_ids),
                )
            )
            == res.shape[0]
        )

    # Test special case when no cell target specified (should return empty connectome)
    writer = EdgeWriter(None, existing_edges=edges_table.copy())
    manipulation(nodes, writer).apply(tgt_ids, target_name=None, node_sets_file=None)
    res = writer.to_pandas()
    assert res.empty
    assert np.all(res.columns == edges_table.columns)
