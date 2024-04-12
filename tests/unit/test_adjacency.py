# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import os

import numpy as np
import pytest
import re
from bluepysnap import Circuit
from numpy.testing import assert_array_equal
from scipy.sparse import csc_matrix

from utils import TEST_DATA_DIR

import connectome_manipulator.connectome_comparison.adjacency as test_module


def test_compute():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = [circuit.nodes["nodeA"]] * 2  # Src/tgt populations
    edges = circuit.edges["nodeA__nodeA__chemical"]
    edges_table = edges.afferent_edges(nodes[1].ids(), properties=edges.property_names)

    def get_adj(edges_table, src_ids, tgt_ids):
        """Extract adjacency matrix from edges table"""
        conns, cnts = np.unique(
            edges_table[["@source_node", "@target_node"]], axis=0, return_counts=True
        )
        adj_mat = np.zeros((len(src_ids), len(tgt_ids)), dtype=bool)
        cnt_mat = np.zeros((len(src_ids), len(tgt_ids)), dtype=int)
        for (_s, _t), _c in zip(conns, cnts):
            adj_mat[np.where(src_ids == _s)[0], np.where(tgt_ids == _t)[0]] = True
            cnt_mat[np.where(src_ids == _s)[0], np.where(tgt_ids == _t)[0]] = _c
        return adj_mat, cnt_mat

    def check_adj(res, ref_adj, ref_cnt, src_ids, tgt_ids):
        for _key in ["adj", "adj_cnt", "common"]:
            assert _key in res, f'ERROR: Results key "{_key}" missing!'
        for _key in ["src_gids", "tgt_gids"]:
            assert _key in res["common"], f'ERROR: Results key "{_key}" in "common" missing!'
        for _key in ["adj", "adj_cnt"]:
            assert "data" in res[_key], f'ERROR: Results key "data" in "{_key}" missing!'

        assert_array_equal(res["common"]["src_gids"], src_ids, "ERROR: Source IDs mismatch!")
        assert_array_equal(res["common"]["tgt_gids"], tgt_ids, "ERROR: Target IDs mismatch!")
        assert isinstance(res["adj"]["data"], csc_matrix), "ERROR: CSC matrix expected!"
        assert isinstance(res["adj_cnt"]["data"], csc_matrix), "ERROR: CSC matrix expected!"
        assert_array_equal(
            res["adj"]["data"].toarray(), ref_adj, "ERROR: Adjacency matrix mismatch!"
        )
        assert_array_equal(
            res["adj_cnt"]["data"].toarray(), ref_cnt, "ERROR: Count matrix mismatch!"
        )

    # Case 1: Invalid inputs
    ## (a) Invalid population name
    popul_name = "INVALID_POPULATION_NAME"
    with pytest.raises(
        AssertionError, match=re.escape(f'Population "{popul_name}" not found in edges file')
    ):
        res = test_module.compute(circuit, sel_src=None, sel_dest=None, edges_popul_name=popul_name)

    ## (b) Empty node sets
    popul_name = "nodeA__nodeA__chemical"
    for _src, _tgt in [(None, []), ([], None), ([], [])]:
        with pytest.raises(AssertionError, match=re.escape("Empty src/tgt node selection(s)")):
            res = test_module.compute(
                circuit, sel_src=_src, sel_dest=_tgt, edges_popul_name=popul_name
            )

    # Case 2: Full circuit
    res = test_module.compute(circuit, sel_src=None, sel_dest=None)
    ref_adj, ref_cnt = get_adj(edges_table, nodes[0].ids(), nodes[1].ids())
    check_adj(res, ref_adj, ref_cnt, nodes[0].ids(), nodes[1].ids())

    # Case 3: Partial circuit (layer by layer)
    for _src_lay in nodes[0].property_values("layer"):
        for _tgt_lay in nodes[1].property_values("layer"):
            sel_src = {"layer": _src_lay}
            sel_tgt = {"layer": _tgt_lay}
            res = test_module.compute(circuit, sel_src=sel_src, sel_dest=sel_tgt)
            ref_adj, ref_cnt = get_adj(edges_table, nodes[0].ids(sel_src), nodes[1].ids(sel_tgt))
            check_adj(res, ref_adj, ref_cnt, nodes[0].ids(sel_src), nodes[1].ids(sel_tgt))
