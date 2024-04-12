# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import os

import numpy as np
import pandas as pd
import pytest
import re
from bluepysnap import Circuit
from mock import Mock, patch
from numpy.testing import assert_array_equal
from scipy.sparse import csc_matrix

from utils import TEST_DATA_DIR, setup_tempdir
import connectome_manipulator.model_building.conn_prob_adj as test_module
from connectome_manipulator.access_functions import get_connections
from connectome_manipulator.model_building import model_types


def test_extract():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = [circuit.nodes["nodeA"]] * 2  # Src/tgt populations
    edges = circuit.edges["nodeA__nodeA__chemical"]

    # Case 1: Empty node selection
    with pytest.raises(AssertionError, match=re.escape("Empty src/tgt node selection(s)")):
        test_module.extract(circuit, sel_src=[], sel_dest=[])

    # Case 2: Non-empty nodes selection
    sel_src = {"layer": "LA"}
    sel_tgt = {"layer": "LB"}
    src_ids = nodes[0].ids(sel_src)
    tgt_ids = nodes[1].ids(sel_tgt)

    conns = get_connections(edges, src_ids, tgt_ids)
    adj_mat = np.zeros((len(src_ids), len(tgt_ids)), dtype=bool)
    for _s, _t in conns:
        adj_mat[np.where(src_ids == _s)[0], np.where(tgt_ids == _t)[0]] = True

    res = test_module.extract(circuit, sel_src=sel_src, sel_dest=sel_tgt)
    assert_array_equal(src_ids, res["src_node_ids"])
    assert_array_equal(tgt_ids, res["tgt_node_ids"])
    assert res["adj_mat"].format.lower() == "csc"
    assert res["adj_mat"].dtype == bool
    assert_array_equal(res["adj_mat"].toarray(), adj_mat)


def test_build():
    # Define (random) adjacency matrix
    np.random.seed(0)
    src_node_ids = np.arange(5, 20)
    tgt_node_ids = np.arange(50, 100)
    adj_mat = csc_matrix(np.random.rand(len(src_node_ids), len(tgt_node_ids)) > 0.75)

    # Case 1: Non-inverted model
    inverted = False
    model = test_module.build(adj_mat, src_node_ids, tgt_node_ids, inverted=inverted)
    # Check internal representation
    assert_array_equal(src_node_ids, model.get_src_nids())
    assert_array_equal(tgt_node_ids, model.get_tgt_nids())
    assert_array_equal(adj_mat.toarray(), model.get_adj_matrix().toarray())
    assert not model.is_inverted()
    # Check model output
    assert_array_equal(
        adj_mat.toarray().astype(float), model.apply(src_nid=src_node_ids, tgt_nid=tgt_node_ids)
    )

    # Case 2: Inverted model
    inverted = True
    model = test_module.build(adj_mat, src_node_ids, tgt_node_ids, inverted=inverted)
    # Check internal representation
    assert_array_equal(src_node_ids, model.get_src_nids())
    assert_array_equal(tgt_node_ids, model.get_tgt_nids())
    assert_array_equal(adj_mat.toarray(), model.get_adj_matrix().toarray())
    assert model.is_inverted()
    # Check (inverted) model output
    assert_array_equal(
        adj_mat.toarray().astype(float),
        1.0 - model.apply(src_nid=src_node_ids, tgt_nid=tgt_node_ids),
    )
